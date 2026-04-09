# =============================================================================
# utils/training/wandb_plots.py
# =============================================================================

import logging
import math
import os
import re
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

import wandb as _wandb

wandb: Any = _wandb
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_to_pandas(dataset, columns: Optional[List[str]] = None) -> pd.DataFrame:
    if columns is None:
        columns = dataset.column_names
    available = [c for c in columns if c in dataset.column_names]
    return dataset.select_columns(available).to_pandas()


def _extract_label(generated_text: str) -> str:
    """
    Extract the Tag from the student's full output.
    Handles the distillation format: Reasoning: ... \\n Tag: ...
    Falls back to the first non-empty line if no Tag: prefix is found.
    """
    text = str(generated_text).strip()

    if "Tag:" in text:
        try:
            match = re.search(r"Tag:\s*(.*)", text, re.IGNORECASE)
            if match:
                return match.group(1).splitlines()[0].strip()
        except Exception:
            pass

    # Fallback: legacy format where the tag was the first line after "Output Tag:"
    if "Output Tag:" in text:
        text = text.split("Output Tag:")[-1].strip()
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text


def _parse_hierarchical(tag: str) -> tuple:
    """
    Split a full tag string into its three taxonomy levels.

    Example:
        '(1g Self service, 2g Mit YouSee), 3 Login issues'
        -> ('1g Self service', '2g Mit YouSee', '3 Login issues')

    Falls back to the full tag string at all three levels if parsing fails,
    so that hierarchical accuracy degrades gracefully for malformed outputs.
    """
    match = re.match(r"\(([^,]+),\s*([^)]+)\),\s*(.+)", tag.strip())
    if match:
        return match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
    return tag, tag, tag


# ---------------------------------------------------------------------------
# One-shot logging at run start
# ---------------------------------------------------------------------------


def log_dataset_overview(train_dataset, eval_dataset) -> None:
    """
    Log train/eval label distributions and dataset sizes to W&B at run start.
    Called once from train.py before training begins.
    """
    if wandb.run is None:
        logger.warning("W&B not initialised — skipping dataset overview.")
        return

    train_df = _safe_to_pandas(train_dataset, ["text", "label"])
    eval_df = _safe_to_pandas(eval_dataset, ["text", "label"])

    def _label_counts(df: pd.DataFrame) -> pd.DataFrame:
        # Extract just the tag so the distribution is not distorted by unique reasoning
        df = df.copy()
        df["clean_tag"] = df["label"].apply(_extract_label)
        counts = df["clean_tag"].value_counts(dropna=False).reset_index()
        counts.columns = ["label", "count"]
        return counts

    wandb.run.summary["dataset/train_size"] = len(train_df)
    wandb.run.summary["dataset/eval_size"] = len(eval_df)

    wandb.log(
        {
            "dataset/train_label_distribution": wandb.Table(
                dataframe=_label_counts(train_df)
            ),
            "dataset/eval_label_distribution": wandb.Table(
                dataframe=_label_counts(eval_df)
            ),
        }
    )

    logger.info(
        f"Logged label distributions — "
        f"train: {len(train_df)} rows, eval: {len(eval_df)} rows."
    )


def log_lora_efficiency(model, cfg: dict) -> None:
    """
    Log the LoRA adapter size (trainable vs frozen parameters) to W&B.
    Called once from train.py after the model is loaded.
    """
    if wandb.run is None:
        logger.warning("W&B not initialised — skipping LoRA efficiency log.")
        return

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen
    pct = round(100.0 * trainable / total, 4) if total else 0.0

    wandb.run.summary["lora/trainable_params"] = trainable
    wandb.run.summary["lora/total_params"] = total
    wandb.run.summary["lora/trainable_pct"] = pct

    breakdown_table = wandb.Table(
        columns=["param_group", "count"],
        data=[["trainable (LoRA)", trainable], ["frozen (base)", frozen]],
    )
    wandb.log(
        {
            "lora/param_breakdown": wandb.plot.bar(
                breakdown_table,
                label="param_group",
                value="count",
                title="Trainable vs Frozen Parameters",
            )
        }
    )

    logger.info(f"LoRA: {trainable:,} trainable / {total:,} total ({pct:.2f}%)")


# ---------------------------------------------------------------------------
# PerplexityAndLossCallback
# ---------------------------------------------------------------------------


class PerplexityAndLossCallback(TrainerCallback):
    """
    Fires on every eval step. Reads eval_loss from the Trainer logs and
    derives perplexity, then logs it to W&B.

    Why no explicit step= in wandb.log():
    The HuggingFace Trainer controls the W&B step counter internally and has
    already advanced it past global_step by the time on_log fires. Passing an
    explicit step causes a "steps must be monotonically increasing" warning and
    the log entry is silently dropped.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "eval_loss" not in logs:
            return

        eval_loss = logs["eval_loss"]
        perplexity = min(math.exp(eval_loss), 1e4)

        # Log both together — do NOT call wandb.log() separately
        # as it creates a new step and causes the Trainer's eval/loss to be dropped
        wandb.log(
            {
                "eval/perplexity": perplexity,
                "eval/loss": eval_loss,
            }
        )

        logger.info(
            f"Step {state.global_step} | "
            f"eval_loss={eval_loss:.4f} | perplexity={perplexity:.4f}"
        )


# ---------------------------------------------------------------------------
# MidTrainingEvalCallback
# ---------------------------------------------------------------------------


class MidTrainingEvalCallback(TrainerCallback):
    """
    A100-only callback. On every eval step, runs generation on a small sample
    of the eval set and logs exact match and macro F1 to W&B.

    This is lightweight (sample_size=50 by default) and gives a mid-training
    signal of classification quality without waiting until the end of training.
    Only enabled when GPU_TIER == 'a100' in train.py.
    """

    def __init__(self, model, tokenizer, eval_dataset, sample_size: int = 50) -> None:
        self.model = model
        self.tokenizer = tokenizer
        total = len(eval_dataset)
        sample_size = min(sample_size, total)
        indices = list(range(0, total, max(1, total // sample_size)))[:sample_size]
        self.sample = eval_dataset.select(indices)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if wandb.run is None:
            return

        step = state.global_step
        device = next(self.model.parameters()).device
        preds, golds = [], []

        self.model.eval()
        for row in self.sample:
            gold_full = str(row["label"]).strip()
            gold_tag = _extract_label(gold_full)

            formatted_text = row["formatted_text"]
            inference_prompt = formatted_text.rsplit(gold_full, 1)[0].rstrip()

            inputs = self.tokenizer(
                inference_prompt, return_tensors="pt", truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=250,
                    do_sample=False,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
            raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            preds.append(_extract_label(raw))
            golds.append(gold_tag)

        exact_match = sum(p == g for p, g in zip(preds, golds)) / len(golds)
        macro_f1 = f1_score(golds, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(golds, preds, average="weighted", zero_division=0)

        # No explicit step= for the same reason as PerplexityAndLossCallback
        wandb.log(
            {
                "eval/mid_exact_match": exact_match,
                "eval/mid_macro_f1": macro_f1,
                "eval/mid_weighted_f1": weighted_f1,
            }
        )

        logger.info(
            f"Step {step} mid-eval | "
            f"exact_match={exact_match:.3f} | macro_f1={macro_f1:.3f} | "
            f"weighted_f1={weighted_f1:.3f}"
        )


# ---------------------------------------------------------------------------
# save_loss_plot
# ---------------------------------------------------------------------------


def save_loss_plot(log_history: list, output_dir: str, run: Any = None) -> None:
    """
    Generate and save a train vs validation loss curve at the end of training.
    Perplexity is overlaid on a secondary y-axis.
    Called from train.py after trainer.train() completes.
    """
    logger.info("Generating combined Train vs Validation Loss plot...")

    train_epochs, train_losses = [], []
    eval_epochs, eval_losses = [], []

    for log in log_history:
        if "loss" in log and "epoch" in log:
            train_epochs.append(log["epoch"])
            train_losses.append(log["loss"])
        if "eval_loss" in log and "epoch" in log:
            eval_epochs.append(log["epoch"])
            eval_losses.append(log["eval_loss"])

    if not train_losses or not eval_losses:
        logger.warning("Not enough data to plot — skipping.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(
        train_epochs,
        train_losses,
        label="Train Loss",
        color="steelblue",
        alpha=0.8,
        linewidth=2,
    )
    ax1.plot(
        eval_epochs,
        eval_losses,
        label="Validation Loss",
        color="darkorange",
        linewidth=2.5,
        marker="o",
        markersize=6,
    )

    ax2 = ax1.twinx()
    eval_perplexities = [min(math.exp(loss), 1e4) for loss in eval_losses]
    ax2.plot(
        eval_epochs,
        eval_perplexities,
        label="Val Perplexity",
        color="mediumseagreen",
        linewidth=1.5,
        linestyle="--",
        alpha=0.7,
    )
    ax2.set_ylabel("Perplexity", fontsize=11, color="mediumseagreen")
    ax2.tick_params(axis="y", labelcolor="mediumseagreen")

    ax1.set_title("Training vs. Validation Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.6)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11)

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "loss_curve.png")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved loss plot -> {save_path}")

    active_run = run if run is not None else wandb.run
    if active_run is not None:
        try:
            active_run.log({"charts/final_loss_curve": wandb.Image(save_path)})
            logger.info("Uploaded loss curve to W&B.")
        except Exception as exc:
            logger.warning(f"Could not upload loss curve to W&B: {exc}")


# ---------------------------------------------------------------------------
# regenerate_loss_plot_from_wandb
# ---------------------------------------------------------------------------


def regenerate_loss_plot_from_wandb(run: Any, output_dir: str) -> None:
    """
    Fetch the training history from W&B and regenerate loss_curve.png.

    Used when training was interrupted before save_loss_plot() could run
    (e.g. the process was killed during FinalEvalWandbCallback). Pulls the
    logged train/loss and eval/loss series from the W&B API and calls
    save_loss_plot() with the reconstructed log_history.

    The HuggingFace Trainer logs to W&B with keys:
        train/loss, train/epoch   — one row per logging_steps interval
        eval/loss,  eval/epoch    — one row per eval_steps interval
    """
    logger.info("Fetching training history from W&B to regenerate loss curve...")
    try:
        api = wandb.Api()
        api_run = api.run(f"{run.entity}/{run.project}/{run.id}")
        history = api_run.history(samples=10_000, pandas=True)
    except Exception as exc:
        logger.warning(f"Could not fetch W&B history: {exc}")
        return

    log_history = []

    # Forward-fill train/epoch so that eval rows (which have NaN for train/epoch)
    # inherit the epoch value from the most recent training step.
    history = history.sort_values("_step").reset_index(drop=True)
    if "train/epoch" in history.columns:
        history["_epoch"] = history["train/epoch"].ffill()
    else:
        history["_epoch"] = history.get("epoch", 0)

    # --- Training loss rows ---
    if "train/loss" in history.columns:
        train_rows = history[["train/loss", "_epoch"]].dropna()
        for _, row in train_rows.iterrows():
            log_history.append({"loss": row["train/loss"], "epoch": row["_epoch"]})

    # --- Eval loss rows ---
    if "eval/loss" in history.columns:
        eval_rows = history[["eval/loss", "_epoch"]].dropna()
        for _, row in eval_rows.iterrows():
            log_history.append({"eval_loss": row["eval/loss"], "epoch": row["_epoch"]})

    if not log_history:
        logger.warning(
            "W&B history did not contain expected train/loss or eval/loss columns — "
            f"found: {list(history.columns)}. Skipping loss plot."
        )
        return

    logger.info(f"Reconstructed {len(log_history)} log entries from W&B history.")
    save_loss_plot(log_history, output_dir, run)


# ---------------------------------------------------------------------------
# FinalEvalWandbCallback
# ---------------------------------------------------------------------------


class FinalEvalWandbCallback(TrainerCallback):
    """
    Fires once after training ends via on_train_end.

    At this point load_best_model_at_end=True has already restored the
    best checkpoint into memory, so run_final_evaluation sees the best
    weights, not the final-step weights.

    This is the bridge between the Trainer lifecycle and the post-training
    classification evaluation. All evaluation logic lives in run_final_evaluation.
    """

    def __init__(self, model, tokenizer, eval_dataset, output_dir, run):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.run = run

    def on_train_end(self, args, state, control, **kwargs):
        import torch

        torch.cuda.empty_cache()
        run_final_evaluation(
            model=self.model,
            tokenizer=self.tokenizer,
            eval_dataset=self.eval_dataset,
            output_dir=self.output_dir,
            run=self.run,
        )


# ---------------------------------------------------------------------------
# run_final_evaluation
# ---------------------------------------------------------------------------


def run_final_evaluation(
    model,
    tokenizer,
    eval_dataset,
    output_dir: str,
    run: Any = None,
    batch_size: int = 1,
) -> None:
    """
    Post-training evaluation on the full silver eval set.

    Runs once immediately after training ends (triggered by FinalEvalWandbCallback),
    while the best checkpoint is still loaded in memory. This measures distillation
    fidelity — how well the student reproduces the teacher's Tag assignments on
    held-out silver data.

    Note: true labels here are silver (machine-generated, S* >= 0.75), not gold.
    Results measure student-teacher agreement, not ground-truth accuracy.
    Ground-truth accuracy is measured separately in scripts/evaluate_gold.py.

    Metrics logged to W&B:
        eval/final_accuracy          — exact tag match rate vs silver labels
        eval/final_macro_f1          — macro-averaged F1 across all tag classes (all classes equal weight)
        eval/final_weighted_f1       — support-weighted F1 (weight by class frequency; operationally honest
          for high-volume triage)
        eval/final_macro_precision   — macro-averaged precision
        eval/final_macro_recall      — macro-averaged recall
        eval/accuracy_level1         — was the Level 1 category correct?
        eval/accuracy_level2         — was the Level 2 sub-category correct?
        eval/accuracy_level3         — was the Level 3 scenario correct?
        eval/format_compliance_rate  — fraction of outputs with Reasoning: + Tag:
        eval/fallback_rate           — fraction of predictions using Misc fallback
        eval/predictions_table       — full per-ticket predictions table
        eval/confusion_matrix        — confusion matrix over predicted vs true tags

    Saved to disk:
        outputs/checkpoints/predictions_final.csv
        Columns: text | predicted_label | true_label | correct |
                 full_output | format_compliant | is_fallback
    """
    logger.info("Running post-training evaluation on silver eval set...")

    try:
        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(model)
        logger.info("Switched to unsloth inference mode.")
    except Exception as exc:
        logger.warning(
            f"Could not switch to unsloth inference mode ({exc}). Using model.eval()."
        )
        model.eval()

    device = next(model.parameters()).device
    rows: List[dict] = []

    # Collect all samples
    all_texts = [str(row["text"]) for row in eval_dataset]
    all_gold_fulls = [str(row["label"]).strip() for row in eval_dataset]
    all_gold_tags = [_extract_label(g) for g in all_gold_fulls]
    all_prompts = [
        row["formatted_text"].rsplit(gold_full, 1)[0].rstrip()
        for row, gold_full in zip(eval_dataset, all_gold_fulls)
    ]

    # Set pad token for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    for i in tqdm(range(0, len(all_prompts), batch_size), desc="Evaluating"):
        batch_prompts = all_prompts[i : i + batch_size]
        batch_texts = all_texts[i : i + batch_size]
        batch_gold_tags = all_gold_tags[i : i + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=250,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        for j, output in enumerate(outputs):
            generated_sequence = output[input_len:]
            raw_output = tokenizer.decode(generated_sequence, skip_special_tokens=True)
            pred_tag = _extract_label(raw_output)
            gold_tag = batch_gold_tags[j]

            rows.append(
                {
                    "text": batch_texts[j],
                    "predicted_label": pred_tag,
                    "true_label": gold_tag,
                    "correct": pred_tag == gold_tag,
                    "full_output": raw_output,
                    "format_compliant": (
                        "Reasoning:" in raw_output and "Tag:" in raw_output
                    ),
                    "is_fallback": "1 Misc incidents" in pred_tag,
                }
            )

    pred_df = pd.DataFrame(rows)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "predictions_final.csv")
    pred_df.to_csv(csv_path, index=False)
    logger.info(f"Saved predictions -> {csv_path}")

    # --- Classification metrics ---
    true_labels = pred_df["true_label"].tolist()
    pred_labels = pred_df["predicted_label"].tolist()

    accuracy = pred_df["correct"].mean()
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(
        true_labels, pred_labels, average="weighted", zero_division=0
    )
    macro_precision = precision_score(
        true_labels, pred_labels, average="macro", zero_division=0
    )
    macro_recall = recall_score(
        true_labels, pred_labels, average="macro", zero_division=0
    )

    # --- Hierarchical accuracy ---
    pred_levels = pd.DataFrame(
        pred_df["predicted_label"].apply(_parse_hierarchical).tolist(),
        columns=["pred_l1", "pred_l2", "pred_l3"],
    )
    true_levels = pd.DataFrame(
        pred_df["true_label"].apply(_parse_hierarchical).tolist(),
        columns=["true_l1", "true_l2", "true_l3"],
    )
    acc_l1 = (pred_levels["pred_l1"] == true_levels["true_l1"]).mean()
    acc_l2 = (pred_levels["pred_l2"] == true_levels["true_l2"]).mean()
    acc_l3 = (pred_levels["pred_l3"] == true_levels["true_l3"]).mean()

    # --- Format and fallback diagnostics ---
    format_compliance = pred_df["format_compliant"].mean()
    fallback_rate = pred_df["is_fallback"].mean()

    logger.info(
        f"Final eval | accuracy={accuracy:.3f} | macro_f1={macro_f1:.3f} | "
        f"weighted_f1={weighted_f1:.3f} | "
        f"precision={macro_precision:.3f} | recall={macro_recall:.3f} | "
        f"L1={acc_l1:.3f} | L2={acc_l2:.3f} | L3={acc_l3:.3f} | "
        f"format_compliance={format_compliance:.3f} | "
        f"fallback_rate={fallback_rate:.3f}"
    )

    active_run = run if run is not None else wandb.run
    if active_run is None:
        logger.warning("No active W&B run — CSV saved to disk, skipping W&B logging.")
        return

    try:
        active_run.log(
            {
                "eval/predictions_table": wandb.Table(dataframe=pred_df),
                "eval/final_accuracy": accuracy,
                "eval/final_macro_f1": macro_f1,
                "eval/final_weighted_f1": weighted_f1,
                "eval/final_macro_precision": macro_precision,
                "eval/final_macro_recall": macro_recall,
                "eval/accuracy_level1": acc_l1,
                "eval/accuracy_level2": acc_l2,
                "eval/accuracy_level3": acc_l3,
                "eval/format_compliance_rate": format_compliance,
                "eval/fallback_rate": fallback_rate,
            }
        )
        active_run.summary["eval/final_accuracy"] = accuracy
        active_run.summary["eval/final_macro_f1"] = macro_f1
        active_run.summary["eval/final_weighted_f1"] = weighted_f1
        active_run.summary["eval/final_macro_precision"] = macro_precision
        active_run.summary["eval/final_macro_recall"] = macro_recall
        active_run.summary["eval/accuracy_level1"] = acc_l1
        active_run.summary["eval/accuracy_level2"] = acc_l2
        active_run.summary["eval/accuracy_level3"] = acc_l3
        active_run.summary["eval/format_compliance_rate"] = format_compliance
        active_run.summary["eval/fallback_rate"] = fallback_rate
    except Exception as exc:
        logger.warning(f"Could not log metrics to W&B: {exc}")

    try:
        active_run.log(
            {
                "eval/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=true_labels,
                    preds=pred_labels,
                )
            }
        )
    except Exception as exc:
        logger.warning(f"Could not log confusion matrix to W&B: {exc}")

    logger.info("Post-training evaluation complete.")
