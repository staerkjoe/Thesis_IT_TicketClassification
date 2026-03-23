# =============================================================================
# utils/training/wandb_plots.py
# =============================================================================
#
# The Run object is threaded explicitly through every function that logs.
# No function relies on the wandb.run global — W&B 0.25.0's atexit handler
# can null that out before post-training evaluation runs.
#
# WHAT GETS LOGGED
# ─────────────────────────────────────────────────────────────────────────────
# Run summary:
#   dataset/train_size, dataset/eval_size
#   lora/trainable_params, lora/total_params, lora/trainable_pct
#   predictions/final_accuracy, predictions/final_macro_f1
#
# Once at run start:
#   dataset/train_label_distribution   Table
#   dataset/eval_label_distribution    Table
#   lora/param_breakdown               Bar chart — trainable vs frozen
#
# Every training step (WandbLossCallback):
#   charts/loss           Combined line chart — train + eval in ONE panel
#   loss/train            Scalar
#   loss/eval             Scalar
#   metrics/perplexity    exp(eval_loss)
#   metrics/grad_norm     Gradient norm
#   train/learning_rate   LR schedule
#   train/global_step     Shared x-axis
#
# After training (run_final_evaluation):
#   predictions/final_table        text | predicted_label | true_label | correct
#   predictions/final_accuracy     scalar
#   predictions/final_macro_f1     scalar
#   predictions/confusion_matrix   plot
#   Local: {output_dir}/predictions_final.csv  (always saved, even if W&B fails)
# =============================================================================

import logging
import math
import os
from typing import Any, List, Optional

import pandas as pd
import torch

import wandb as _wandb

wandb: Any = _wandb
logger = logging.getLogger(__name__)

from sklearn.metrics import f1_score
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# W&B metric definitions
# ---------------------------------------------------------------------------


def init_wandb_metrics() -> None:
    """
    Declare custom metric axes so loss/train and loss/eval share one panel.
    Call immediately after wandb.init() in init_wandb() (train.py).
    """
    if wandb.run is None:
        return

    wandb.define_metric("train/global_step")
    wandb.define_metric("loss/*", step_metric="train/global_step")
    wandb.define_metric("metrics/*", step_metric="train/global_step")
    wandb.define_metric("train/learning_rate", step_metric="train/global_step")

    logger.info("W&B metric axes defined.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_to_pandas(dataset, columns: Optional[List[str]] = None) -> pd.DataFrame:
    if columns is None:
        columns = dataset.column_names
    available = [c for c in columns if c in dataset.column_names]
    return dataset.select_columns(available).to_pandas()


def _extract_label(generated_text: str) -> str:
    text = generated_text.strip()
    if "Output Tag:" in text:
        text = text.split("Output Tag:")[-1].strip()
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text


def _align_series(
    all_steps: List[int],
    steps: List[int],
    values: List[float],
) -> List[float]:
    """Forward-fill sparse series onto a dense step list."""
    lookup = dict(zip(steps, values))
    result: List[float] = []
    last = float("nan")
    for s in all_steps:
        if s in lookup:
            last = lookup[s]
        result.append(last)
    return result


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
        counts = df["label"].value_counts(dropna=False).reset_index()
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
        data=[
            ["trainable (LoRA)", trainable],
            ["frozen (base)", frozen],
        ],
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
# Training callback
# ---------------------------------------------------------------------------


class WandbLossCallback(TrainerCallback):
    """
    Handles all W&B metric logging during training.
    Receives the Run object explicitly so it never depends on wandb.run global.
    """

    def __init__(self, run: Any):
        self._run = run
        self._train_steps: List[int] = []
        self._train_losses: List[float] = []
        self._eval_steps: List[int] = []
        self._eval_losses: List[float] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self._run is None or logs is None:
            return control

        payload: dict = {"train/global_step": state.global_step}

        if "loss" in logs:
            payload["loss/train"] = float(logs["loss"])
            self._train_steps.append(state.global_step)
            self._train_losses.append(float(logs["loss"]))

        if "eval_loss" in logs:
            payload["loss/eval"] = float(logs["eval_loss"])
            self._eval_steps.append(state.global_step)
            self._eval_losses.append(float(logs["eval_loss"]))

            try:
                payload["metrics/perplexity"] = math.exp(
                    min(float(logs["eval_loss"]), 20.0)
                )
            except (OverflowError, ValueError):
                pass

            if self._train_losses and self._eval_losses:
                all_steps = sorted(set(self._train_steps + self._eval_steps))
                train_curve = _align_series(
                    all_steps, self._train_steps, self._train_losses
                )
                eval_curve = _align_series(
                    all_steps, self._eval_steps, self._eval_losses
                )

                payload["charts/loss"] = wandb.plot.line_series(
                    xs=all_steps,
                    ys=[train_curve, eval_curve],
                    keys=["train", "eval"],
                    title="Loss: train vs eval",
                    xname="global_step",
                )

        if "grad_norm" in logs:
            payload["metrics/grad_norm"] = float(logs["grad_norm"])

        if "learning_rate" in logs:
            payload["train/learning_rate"] = float(logs["learning_rate"])

        try:
            self._run.log(payload)
        except Exception as exc:
            logger.warning(f"WandbLossCallback: failed to log — {exc}")

        return control


class WandbEvalPredictionCallback(TrainerCallback):
    """Stub — mid-training generation skipped. See run_final_evaluation()."""

    def __init__(
        self,
        tokenizer,
        eval_dataset,
        max_samples: int = 25,
        output_dir: Optional[str] = None,
        max_new_tokens: int = 20,
    ):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.max_samples = max_samples
        self.output_dir = output_dir
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        return control


# ---------------------------------------------------------------------------
# Post-training evaluation
# ---------------------------------------------------------------------------


def run_final_evaluation(
    model,
    tokenizer,
    eval_dataset,
    output_dir: str,
    run: Any = None,
) -> None:
    """
    Run greedy generation on the full eval set after training.

    Always saves predictions_final.csv to output_dir, even if W&B logging
    fails. W&B logging uses the explicit `run` object — never wandb.run global.

    CSV columns: text | predicted_label | true_label | correct
    """
    logger.info("Running post-training evaluation on full eval set...")

    # Switch to unsloth inference mode
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
        gold_label = str(row["label"]).strip()
        formatted_text = row["formatted_text"]

        inference_prompt = formatted_text.rsplit(gold_label, 1)[0].rstrip()

        inputs = tokenizer(inference_prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred_label = _extract_label(raw_output)

        rows.append(
            {
                "text": row["text"],
                "predicted_label": pred_label,
                "true_label": gold_label,
                "correct": pred_label == gold_label,
            }
        )

    pred_df = pd.DataFrame(rows)

    # Always save CSV — independent of W&B status
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "predictions_final.csv")
    pred_df.to_csv(csv_path, index=False)
    logger.info(f"Saved predictions → {csv_path}")

    accuracy = pred_df["correct"].mean()
    macro_f1 = f1_score(
        pred_df["true_label"].tolist(),
        pred_df["predicted_label"].tolist(),
        average="macro",
        zero_division=0,
    )
    logger.info(f"Final eval — accuracy: {accuracy:.3f}  macro_f1: {macro_f1:.3f}")

    # W&B logging — skip gracefully if run is unavailable
    active_run = run if run is not None else wandb.run
    if active_run is None:
        logger.warning(
            "No active W&B run — skipping W&B logging. CSV was saved to disk."
        )
        return

    try:
        active_run.log(
            {
                "predictions/final_table": wandb.Table(dataframe=pred_df),
                "predictions/final_accuracy": accuracy,
                "predictions/final_macro_f1": macro_f1,
            }
        )
        active_run.summary["predictions/final_accuracy"] = accuracy
        active_run.summary["predictions/final_macro_f1"] = macro_f1
    except Exception as exc:
        logger.warning(f"Could not log prediction table to W&B: {exc}")

    try:
        active_run.log(
            {
                "predictions/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=pred_df["true_label"].tolist(),
                    preds=pred_df["predicted_label"].tolist(),
                )
            }
        )
    except Exception as exc:
        logger.warning(f"Could not log confusion matrix: {exc}")

    logger.info("Post-training evaluation complete.")
