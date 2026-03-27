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
import torch.nn.functional as F
import wandb as _wandb
from sklearn.metrics import f1_score, precision_score, recall_score
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
    Smarter extraction to pull the exact tag out of the new Distillation format
    (Reasoning -> Tag -> Confidence).
    """
    text = str(generated_text).strip()

    # Look for the 'Tag:' prefix and extract everything on that line
    if "Tag:" in text:
        try:
            match = re.search(r"Tag:\s*(.*)", text, re.IGNORECASE)
            if match:
                return match.group(1).splitlines()[0].strip()
        except Exception:
            pass

    # Fallback to legacy behavior if 'Tag:' is missing
    if "Output Tag:" in text:
        text = text.split("Output Tag:")[-1].strip()
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text


# ---------------------------------------------------------------------------
# One-shot logging at run start
# ---------------------------------------------------------------------------


def log_dataset_overview(train_dataset, eval_dataset) -> None:
    if wandb.run is None:
        logger.warning("W&B not initialised — skipping dataset overview.")
        return

    train_df = _safe_to_pandas(train_dataset, ["text", "label"])
    eval_df = _safe_to_pandas(eval_dataset, ["text", "label"])

    def _label_counts(df: pd.DataFrame) -> pd.DataFrame:
        # Extract just the tag so the distribution plot isn't distorted by unique reasoning
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
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        if logs is None or "eval_loss" not in logs:
            return

        eval_loss = logs["eval_loss"]
        perplexity = min(math.exp(eval_loss), 1e4)

        # Do NOT pass step= here — the HuggingFace Trainer controls the W&B
        # step counter internally and has already advanced it past global_step
        # by the time on_log fires. Passing an explicit step causes the
        # "steps must be monotonically increasing" warning and the log is dropped.
        wandb.log({"eval/perplexity": perplexity})

        logger.info(
            f"Step {state.global_step} | "
            f"eval_loss={eval_loss:.4f} | perplexity={perplexity:.4f}"
        )


# ---------------------------------------------------------------------------
# MidTrainingEvalCallback
# ---------------------------------------------------------------------------


class MidTrainingEvalCallback(TrainerCallback):
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
                    max_new_tokens=250,  # Increased to handle the reasoning chain
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

        # Same reason as PerplexityAndLossCallback — no explicit step=
        wandb.log({"eval/mid_exact_match": exact_match, "eval/mid_macro_f1": macro_f1})

        logger.info(
            f"Step {step} mid-eval | "
            f"exact_match={exact_match:.3f} | macro_f1={macro_f1:.3f}"
        )


# ---------------------------------------------------------------------------
# save_loss_plot
# ---------------------------------------------------------------------------


def save_loss_plot(log_history: list, output_dir: str, run: Any = None) -> None:
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
# run_final_evaluation
# ---------------------------------------------------------------------------


def run_final_evaluation(
    model,
    tokenizer,
    eval_dataset,
    output_dir: str,
    run: Any = None,
) -> None:
    logger.info("Running post-training evaluation on full eval set with Logprobs...")

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

    for row in eval_dataset:
        gold_full = str(row["label"]).strip()
        gold_tag = _extract_label(gold_full)

        # 1. Grab the Teacher's S* score from the CSV row
        teacher_s_star = float(row.get("s_star", 0.0))

        formatted_text = row["formatted_text"]
        inference_prompt = formatted_text.rsplit(gold_full, 1)[0].rstrip()

        inputs = tokenizer(inference_prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=250,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # 2. TURN ON THE BRAIN SCANNER
                output_scores=True,
                return_dict_in_generate=True,
            )

        # 3. EXTRACT TOKENS AND CALCULATE MATH CONFIDENCE
        generated_sequence = outputs.sequences[0][inputs["input_ids"].shape[1] :]
        raw_output = tokenizer.decode(generated_sequence, skip_special_tokens=True)
        pred_tag = _extract_label(raw_output)

        # Calculate Logprobs for the generated tokens
        log_probs = []
        for step, token_id in enumerate(generated_sequence):
            # Convert raw logits to log-probabilities using Softmax
            step_log_probs = F.log_softmax(outputs.scores[step][0], dim=-1)
            # Get the exact log-probability of the token the model actually chose
            token_log_prob = step_log_probs[token_id].item()
            log_probs.append(token_log_prob)

        # Average the logprobs over the generated response
        mean_logprob = sum(log_probs) / len(log_probs) if log_probs else 0.0
        # Convert mathematical logprob back to a human-readable percentage (0.0 to 1.0)
        student_confidence = math.exp(mean_logprob)

        rows.append(
            {
                "text": row["text"],
                "predicted_label": pred_tag,
                "true_label": gold_tag,
                "correct": pred_tag == gold_tag,
                "student_confidence": round(student_confidence, 4),  # New!
                "teacher_s_star": round(teacher_s_star, 4),  # New!
                "mean_logprob": round(mean_logprob, 4),  # New!
                "full_output": raw_output,
            }
        )

    pred_df = pd.DataFrame(rows)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "predictions_final.csv")
    pred_df.to_csv(csv_path, index=False)
    logger.info(f"Saved predictions -> {csv_path}")

    # ... [Keep the rest of your logging logic (Accuracy, F1, etc) exactly the same] ...

    true_labels = pred_df["true_label"].tolist()
    pred_labels = pred_df["predicted_label"].tolist()

    accuracy = pred_df["correct"].mean()
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    macro_precision = precision_score(
        true_labels, pred_labels, average="macro", zero_division=0
    )
    macro_recall = recall_score(
        true_labels, pred_labels, average="macro", zero_division=0
    )

    logger.info(
        f"Final eval | accuracy={accuracy:.3f} | macro_f1={macro_f1:.3f} | "
        f"precision={macro_precision:.3f} | recall={macro_recall:.3f}"
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
                "eval/final_macro_precision": macro_precision,
                "eval/final_macro_recall": macro_recall,
            }
        )
        active_run.summary["eval/final_accuracy"] = accuracy
        active_run.summary["eval/final_macro_f1"] = macro_f1
        active_run.summary["eval/final_macro_precision"] = macro_precision
        active_run.summary["eval/final_macro_recall"] = macro_recall
    except Exception as exc:
        logger.warning(f"Could not log prediction table to W&B: {exc}")

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
        logger.warning(f"Could not log confusion matrix: {exc}")

    logger.info("Post-training evaluation complete.")
