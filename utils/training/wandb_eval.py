# =============================================================================
# utils/training/wandb_eval.py
#
# Standalone post-training evaluation pipeline.
# Called from scripts/evaluate.py.
#
# Covers three dataset types:
#   silver — held-out 20% eval set (teacher-labeled, S* >= 0.75)
#   fresh  — unseen test set (teacher-labeled, same pipeline)
#   gold   — 50-ticket expert-labeled gold standard
#
# Metrics per the evaluation framework:
#   Classification : agreement_rate, macro_f1, weighted_f1, per_class_l1_f1,
#                    accuracy_level1/2/3, fallback_rate, confusion_matrix
#   Generation     : format_compliance_rate, reasoning_length_mean/median,
#                    repetition_score, rouge_l, bertscore_f1
#   General        : student_confidence (mean Tag logprob), calibration_curve,
#                    spearman_correlation, mean_latency_ms
#   Gold           : student vs human agreement / F1 (label col = human tag directly)
# =============================================================================

import logging
import math
import os
import re
import time
from typing import (  # Optional kept for _find_subseq return type
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score

import wandb as _wandb

wandb: Any = _wandb
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional dependencies — gracefully skip if unavailable
# ---------------------------------------------------------------------------

try:
    from rouge_score import rouge_scorer as _rouge_scorer

    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False
    logger.warning("rouge-score not installed — ROUGE-L will be skipped.")

try:
    from bert_score import score as _bertscore_fn

    _BERTSCORE_AVAILABLE = True
except ImportError:
    _BERTSCORE_AVAILABLE = False
    logger.warning("bert-score not installed — BERTScore will be skipped.")


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------


def _extract_label(text: str) -> str:
    """Extract the Tag value from a full model output."""
    text = str(text).strip()
    match = re.search(r"Tag:\s*(.*)", text, re.IGNORECASE)
    if match:
        return match.group(1).splitlines()[0].strip()
    # Fallback: first non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text


def _extract_teacher_reasoning(label: str) -> str:
    """Extract the Reasoning portion from a teacher label string.
    Label format: 'Reasoning: ...\nTag: ...'
    """
    match = re.search(
        r"Reasoning:\s*(.*?)(?:\nTag:|$)", label, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    # Fallback: everything before Tag:
    if "Tag:" in label:
        return label[: label.index("Tag:")].replace("Reasoning:", "").strip()
    return ""


def _extract_student_reasoning(output: str) -> str:
    """Extract the Reasoning portion from a student output string."""
    match = re.search(
        r"Reasoning:\s*(.*?)(?:\nTag:|$)", output, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    if "Tag:" in output:
        return output[: output.index("Tag:")].replace("Reasoning:", "").strip()
    return ""


def _parse_hierarchical(tag: str) -> Tuple[str, str, str]:
    """
    Split a full tag into its three taxonomy levels.

    Example:
        '(1g Self service, 2g Mit YouSee), 3 Login issues'
        -> ('1g Self service', '2g Mit YouSee', '3 Login issues')

    Falls back to the full tag at all levels for malformed outputs.
    """
    match = re.match(r"\(([^,]+),\s*([^)]+)\),\s*(.+)", tag.strip())
    if match:
        return match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
    return tag, tag, tag


def _bigram_repetition(text: str) -> float:
    """Fraction of bigrams that are repeated (0 = no repetition, 1 = full repetition)."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < 2:
        return 0.0
    bigrams = list(zip(words, words[1:]))
    return 1.0 - len(set(bigrams)) / len(bigrams)


# ---------------------------------------------------------------------------
# Token-level confidence extraction
# ---------------------------------------------------------------------------


def _find_subseq(haystack: List[int], needle: List[int]) -> Optional[int]:
    """Return start index of first occurrence of needle in haystack, or None."""
    n, h = len(needle), len(haystack)
    for i in range(h - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return None


def _compute_tag_logprob(
    scores: Tuple[torch.Tensor, ...],
    generated_ids: torch.Tensor,
    tokenizer,
) -> float:
    """
    Compute mean log-probability over the Tag value tokens only.
    Excludes the Reasoning chain so the score reflects classification confidence.
    Returns nan if the Tag line cannot be located in the output.
    """
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    tag_match = re.search(r"Tag:\s*(.+?)(?:\n|$)", full_text, re.IGNORECASE)
    if not tag_match:
        return float("nan")

    tag_value = tag_match.group(1).strip()
    if not tag_value:
        return float("nan")

    tag_ids = tokenizer.encode(tag_value, add_special_tokens=False)
    gen_list = generated_ids.tolist()

    start_idx = _find_subseq(gen_list, tag_ids)
    if start_idx is None:
        # Approximate using prefix length
        prefix = full_text[: tag_match.start(1)]
        start_idx = len(tokenizer.encode(prefix, add_special_tokens=False))

    end_idx = min(start_idx + len(tag_ids), len(scores))
    if start_idx >= end_idx:
        return float("nan")

    logprobs = []
    for i in range(start_idx, end_idx):
        logits = scores[i]
        if logits.dim() == 2:
            logits = logits[0]
        lp = F.log_softmax(logits.float(), dim=-1)[gen_list[i]].item()
        logprobs.append(lp)

    return sum(logprobs) / len(logprobs) if logprobs else float("nan")


# ---------------------------------------------------------------------------
# Inference prompt builder (handles Mistral's no-system-role constraint)
# ---------------------------------------------------------------------------


def _build_inference_prompt(ticket_text: str, tokenizer, labels_cfg: dict) -> str:
    """Build inference prompt with model-specific chat template handling."""
    from utils.training.llm_handler import build_system_prompt

    system_content = build_system_prompt(labels_cfg)
    user_content = f"Input Description: {ticket_text}\nOutput Tag:"

    model_type = getattr(tokenizer, "name_or_path", "").lower()
    if "mistral" in model_type or "ministral" in model_type:
        messages = [{"role": "user", "content": f"{system_content}\n\n{user_content}"}]
    else:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Single-ticket inference with logprobs and latency
# ---------------------------------------------------------------------------


def _classify_single(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 250,
) -> Tuple[str, float, float]:
    """
    Run inference on one prompt.

    Returns:
        full_output    — decoded generation (Reasoning + Tag)
        tag_logprob    — mean log-prob over Tag value tokens (nan if unavailable)
        latency_ms     — wall-clock inference time in milliseconds
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        result = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    generated_ids = result.sequences[0][inputs["input_ids"].shape[1] :]
    full_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    tag_logprob = _compute_tag_logprob(result.scores, generated_ids, tokenizer)

    return full_output, tag_logprob, latency_ms


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------


def run_inference_batch(
    model,
    tokenizer,
    df: pd.DataFrame,
    labels_cfg: dict,
    max_new_tokens: int = 250,
) -> pd.DataFrame:
    """
    Run inference on every row of df (must have a 'text' column).

    Returns df with added columns:
        full_output       — raw model generation
        predicted_label   — extracted Tag value
        format_compliant  — bool: output contains both Reasoning: and Tag:
        is_fallback       — bool: predicted tag is the Misc fallback
        tag_logprob       — mean log-prob over Tag tokens (nan if unavailable)
        student_confidence— exp(tag_logprob) clipped to [0, 1]; confidence proxy
        latency_ms        — inference time per ticket
    """
    rows = []
    total = len(df)
    logger.info(f"Running batch inference on {total} tickets...")

    for i, row in enumerate(df.itertuples(index=False), 1):
        ticket_text = str(row.text)
        prompt = _build_inference_prompt(ticket_text, tokenizer, labels_cfg)
        full_output, tag_logprob, latency_ms = _classify_single(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens
        )
        pred_tag = _extract_label(full_output)
        confidence = (
            math.exp(tag_logprob) if not math.isnan(tag_logprob) else float("nan")
        )

        rows.append(
            {
                "full_output": full_output,
                "predicted_label": pred_tag,
                "format_compliant": (
                    "Reasoning:" in full_output and "Tag:" in full_output
                ),
                "is_fallback": "1 Misc incidents" in pred_tag,
                "tag_logprob": tag_logprob,
                "student_confidence": (
                    min(max(confidence, 0.0), 1.0)
                    if not math.isnan(confidence)
                    else float("nan")
                ),
                "latency_ms": latency_ms,
            }
        )

        if i % 50 == 0 or i == total:
            logger.info(f"  [{i}/{total}] last latency: {latency_ms:.0f} ms")

    result_df = df.copy().reset_index(drop=True)
    result_df = pd.concat([result_df, pd.DataFrame(rows)], axis=1)
    return result_df


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def _compute_classification_metrics(
    pred_df: pd.DataFrame,
    pred_col: str = "predicted_label",
    true_col: str = "true_label",
    prefix: str = "eval",
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """
    Compute all classification metrics for one dataset.
    Returns a flat dict of metric_name -> value.
    """
    preds = pred_df[pred_col].tolist()
    trues = pred_df[true_col].tolist()

    agreement_rate = sum(p == t for p, t in zip(preds, trues)) / len(trues)
    macro_f1 = f1_score(trues, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(trues, preds, average="weighted", zero_division=0)
    macro_precision = precision_score(trues, preds, average="macro", zero_division=0)
    macro_recall = recall_score(trues, preds, average="macro", zero_division=0)

    # Hierarchical accuracy
    pred_levels = pd.DataFrame(
        pred_df[pred_col].apply(_parse_hierarchical).tolist(),
        columns=["pred_l1", "pred_l2", "pred_l3"],
    )
    true_levels = pd.DataFrame(
        pred_df[true_col].apply(_parse_hierarchical).tolist(),
        columns=["true_l1", "true_l2", "true_l3"],
    )
    acc_l1 = (pred_levels["pred_l1"] == true_levels["true_l1"]).mean()
    acc_l2 = (pred_levels["pred_l2"] == true_levels["true_l2"]).mean()
    acc_l3 = (pred_levels["pred_l3"] == true_levels["true_l3"]).mean()

    # Per-L1-class F1
    unique_l1 = sorted(
        set(true_levels["true_l1"].tolist() + pred_levels["pred_l1"].tolist())
    )
    per_l1_f1 = f1_score(
        true_levels["true_l1"].tolist(),
        pred_levels["pred_l1"].tolist(),
        labels=unique_l1,
        average=None,
        zero_division=0,
    )
    per_l1_f1_dict = {
        f"{prefix}/per_l1_f1/{lbl}": v for lbl, v in zip(unique_l1, per_l1_f1)
    }

    fallback_rate = (
        pred_df["is_fallback"].mean()
        if "is_fallback" in pred_df.columns
        else float("nan")
    )
    format_compliance = (
        pred_df["format_compliant"].mean()
        if "format_compliant" in pred_df.columns
        else float("nan")
    )

    metrics = {
        f"{prefix}/agreement_rate": agreement_rate,
        f"{prefix}/macro_f1": macro_f1,
        f"{prefix}/weighted_f1": weighted_f1,
        f"{prefix}/macro_precision": macro_precision,
        f"{prefix}/macro_recall": macro_recall,
        f"{prefix}/accuracy_level1": acc_l1,
        f"{prefix}/accuracy_level2": acc_l2,
        f"{prefix}/accuracy_level3": acc_l3,
        f"{prefix}/fallback_rate": fallback_rate,
        f"{prefix}/format_compliance_rate": format_compliance,
        **per_l1_f1_dict,
    }
    return metrics, preds, trues


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------


def _compute_generation_metrics(
    pred_df: pd.DataFrame,
    tokenizer,
    prefix: str = "eval",
) -> Dict[str, Any]:
    """
    Compute generation quality metrics comparing student vs teacher reasoning.

    Requires 'full_output' and 'label' columns in pred_df.
    Teacher reasoning is extracted from the 'label' column.
    ROUGE-L and BERTScore are skipped if libraries are unavailable.
    """
    metrics: Dict[str, Any] = {}

    if "label" not in pred_df.columns or "full_output" not in pred_df.columns:
        logger.warning(
            "Skipping generation metrics — 'label' or 'full_output' column missing."
        )
        return metrics

    teacher_reasonings = pred_df["label"].apply(_extract_teacher_reasoning).tolist()
    student_reasonings = (
        pred_df["full_output"].apply(_extract_student_reasoning).tolist()
    )

    # Reasoning length (token count via tokenizer)
    student_lengths = [
        len(tokenizer.encode(r, add_special_tokens=False)) for r in student_reasonings
    ]
    metrics[f"{prefix}/reasoning_length_mean"] = sum(student_lengths) / len(
        student_lengths
    )
    metrics[f"{prefix}/reasoning_length_median"] = float(
        pd.Series(student_lengths).median()
    )

    # Repetition score (bigram repetition rate in student outputs)
    rep_scores = [_bigram_repetition(r) for r in student_reasonings]
    metrics[f"{prefix}/repetition_score_mean"] = sum(rep_scores) / len(rep_scores)

    # ROUGE-L: student vs teacher reasoning
    if _ROUGE_AVAILABLE:
        scorer = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        rouge_scores = []
        for ref, hyp in zip(teacher_reasonings, student_reasonings):
            if ref and hyp:
                rouge_scores.append(scorer.score(ref, hyp)["rougeL"].fmeasure)
        if rouge_scores:
            metrics[f"{prefix}/rouge_l"] = sum(rouge_scores) / len(rouge_scores)
    else:
        logger.info("ROUGE-L skipped (rouge-score not installed).")

    # BERTScore: student vs teacher reasoning
    if _BERTSCORE_AVAILABLE:
        valid_pairs = [
            (r, h) for r, h in zip(teacher_reasonings, student_reasonings) if r and h
        ]
        if valid_pairs:
            refs, hyps = zip(*valid_pairs)
            try:
                _, _, F1 = _bertscore_fn(
                    list(hyps), list(refs), lang="en", verbose=False, device=None
                )
                metrics[f"{prefix}/bertscore_f1"] = F1.mean().item()
            except Exception as exc:
                logger.warning(f"BERTScore computation failed: {exc}")
    else:
        logger.info("BERTScore skipped (bert-score not installed).")

    return metrics


# ---------------------------------------------------------------------------
# General / cross-cutting metrics
# ---------------------------------------------------------------------------


def _compute_general_metrics(
    pred_df: pd.DataFrame,
    prefix: str = "eval",
) -> Dict[str, Any]:
    """
    Compute calibration, Spearman correlation, and latency metrics.

    Calibration requires 'student_confidence' and 'correct' columns.
    Spearman requires 'student_confidence' and the 's_star' column (silver/fresh only).
    """
    metrics: Dict[str, Any] = {}

    # Latency
    if "latency_ms" in pred_df.columns:
        metrics[f"{prefix}/mean_latency_ms"] = pred_df["latency_ms"].mean()
        metrics[f"{prefix}/p95_latency_ms"] = pred_df["latency_ms"].quantile(0.95)

    # Calibration curve (student_confidence vs accuracy)
    if "student_confidence" in pred_df.columns and "correct" in pred_df.columns:
        cal_rows = []
        bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
        for lo, hi in bins:
            mask = (pred_df["student_confidence"] >= lo) & (
                pred_df["student_confidence"] < hi
            )
            bucket = pred_df[mask]
            if len(bucket) > 0:
                cal_rows.append(
                    {
                        "confidence_bucket": f"{int(lo*100)}-{int(hi*100)}%",
                        "accuracy": bucket["correct"].mean(),
                        "count": len(bucket),
                    }
                )
        if cal_rows:
            metrics[f"{prefix}/calibration_table"] = wandb.Table(
                dataframe=pd.DataFrame(cal_rows)
            )

    # Spearman correlation: student_confidence vs teacher S* score
    # s_star is a dedicated numeric column in silver/fresh CSVs (not embedded in label text).
    # Gold CSVs have no s_star column — this block is silently skipped for gold.
    if "student_confidence" in pred_df.columns and "s_star" in pred_df.columns:
        valid = pred_df["student_confidence"].notna() & pred_df["s_star"].notna()
        if valid.sum() >= 10:
            rho, pval = stats.spearmanr(
                pred_df.loc[valid, "student_confidence"], pred_df.loc[valid, "s_star"]
            )
            metrics[f"{prefix}/spearman_rho"] = rho
            metrics[f"{prefix}/spearman_pval"] = pval
        else:
            logger.info("Spearman skipped — insufficient non-null s_star pairs.")

    return metrics


# ---------------------------------------------------------------------------
# W&B logging orchestrator
# ---------------------------------------------------------------------------


def log_evaluation_to_wandb(
    pred_df: pd.DataFrame,
    run: Any,
    dataset_type: str,
    tokenizer,
    output_dir: str = "./outputs/eval",
) -> None:
    """
    Compute and log all evaluation metrics for one dataset to W&B.

    For silver/fresh: true_label is extracted from the teacher label column
      (format: 'Reasoning: ...\nTag: ...').  Generation metrics and Spearman
      (vs s_star column) are computed.
    For gold: true_label IS the label column directly (bare human tag, no reasoning).
      Generation metrics and Spearman are skipped automatically (no teacher
      reasoning and no s_star column in gold CSV).

    Args:
        pred_df      — DataFrame from run_inference_batch, must have 'label' column
        run          — active wandb.Run
        dataset_type — "silver", "fresh", or "gold"
        tokenizer    — used for token-length computation in generation metrics
        output_dir   — where to write predictions CSV
    """
    prefix = f"eval/{dataset_type}"

    # Establish ground-truth label column
    if "true_label" not in pred_df.columns:
        pred_df = pred_df.copy()
        pred_df["true_label"] = pred_df["label"].apply(_extract_label)

    if "correct" not in pred_df.columns:
        pred_df["correct"] = pred_df["predicted_label"] == pred_df["true_label"]

    # Save predictions to disk
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"predictions_{dataset_type}.csv")
    pred_df.to_csv(csv_path, index=False)
    logger.info(f"Saved predictions -> {csv_path}")

    # --- Collect all metrics ---
    all_metrics: Dict[str, Any] = {}

    # Classification (student vs teacher for silver/fresh; student vs human for gold)
    clf_metrics, preds, trues = _compute_classification_metrics(pred_df, prefix=prefix)
    all_metrics.update(clf_metrics)

    # Generation (only meaningful when teacher reasoning is available)
    if dataset_type in ("silver", "fresh"):
        gen_metrics = _compute_generation_metrics(
            pred_df, tokenizer=tokenizer, prefix=prefix
        )
        all_metrics.update(gen_metrics)

    # General (calibration, spearman if s_star present, latency)
    gen_general = _compute_general_metrics(pred_df, prefix=prefix)
    all_metrics.update(gen_general)

    # Log scalar metrics
    scalar_metrics = {
        k: v for k, v in all_metrics.items() if not isinstance(v, wandb.Table)
    }
    table_metrics = {k: v for k, v in all_metrics.items() if isinstance(v, wandb.Table)}

    if run is not None:
        try:
            run.log({**scalar_metrics, **table_metrics})
            for k, v in scalar_metrics.items():
                if not isinstance(v, float) or not math.isnan(v):
                    run.summary[k] = v
        except Exception as exc:
            logger.warning(f"Could not log scalar metrics to W&B: {exc}")

        # Predictions table
        try:
            display_cols = [
                c
                for c in [
                    "text",
                    "true_label",
                    "predicted_label",
                    "correct",
                    "is_fallback",
                    "format_compliant",
                    "student_confidence",
                    "latency_ms",
                ]
                if c in pred_df.columns
            ]
            run.log(
                {
                    f"{prefix}/predictions_table": wandb.Table(
                        dataframe=pred_df[display_cols]
                    )
                }
            )
        except Exception as exc:
            logger.warning(f"Could not log predictions table to W&B: {exc}")

        # Confusion matrix
        try:
            run.log(
                {
                    f"{prefix}/confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=trues,
                        preds=preds,
                        title=f"Confusion Matrix — {dataset_type}",
                    )
                }
            )
        except Exception as exc:
            logger.warning(f"Could not log confusion matrix to W&B: {exc}")

    # Log summary to console
    logger.info(
        f"[{dataset_type.upper()}] "
        f"agreement={scalar_metrics.get(f'{prefix}/agreement_rate', float('nan')):.3f} | "
        f"macro_f1={scalar_metrics.get(f'{prefix}/macro_f1', float('nan')):.3f} | "
        f"weighted_f1={scalar_metrics.get(f'{prefix}/weighted_f1', float('nan')):.3f} | "
        f"L1={scalar_metrics.get(f'{prefix}/accuracy_level1', float('nan')):.3f} | "
        f"L2={scalar_metrics.get(f'{prefix}/accuracy_level2', float('nan')):.3f} | "
        f"L3={scalar_metrics.get(f'{prefix}/accuracy_level3', float('nan')):.3f} | "
        f"format={scalar_metrics.get(f'{prefix}/format_compliance_rate', float('nan')):.3f} | "
        f"fallback={scalar_metrics.get(f'{prefix}/fallback_rate', float('nan')):.3f} | "
        f"latency={scalar_metrics.get(f'{prefix}/mean_latency_ms', float('nan')):.0f}ms"
    )
