# =============================================================================
# utils/training/wandb_plots.py
# =============================================================================
#
# Logging strategy summary:
#
#   loss/train          — logged by WandbLossCallback every logging_steps
#   loss/eval           — re-logged by WandbLossCallback when Trainer emits eval_loss
#   ↑ Both share the "loss/" prefix → W&B puts them in ONE panel automatically
#
#   lora/param_breakdown       — Table (2 rows: trainable vs frozen) → bar chart
#   lora/trainable_pct         — run summary scalar
#
#   dataset/train_label_distribution  — Table logged once at run start
#   dataset/eval_label_distribution   — Table logged once at run start
#
#   predictions/final_table    — text | true_label | predicted_label | correct
#   predictions/final_accuracy — scalar (also in run summary)
#   predictions/final_macro_f1 — scalar (also in run summary)
#   predictions/confusion_matrix
#   Local file: {output_dir}/predictions_final.csv
#
# WHY generation lives outside the Trainer
#   The HF SFT Trainer optimises next-token-prediction loss — it has no concept
#   of a "predicted label". Calling model.generate() inside on_evaluate() is
#   unreliable: gradient checkpointing is still active and the model is in
#   training mode. run_final_evaluation() is called from train.py AFTER
#   trainer.train() returns; we then switch to unsloth inference mode for
#   clean, reliable generation on the full eval set.
#
# NOTE — no manual step= anywhere
#   The Trainer owns the W&B global step. Passing a stale step causes the
#   "Steps must be monotonically increasing" warning.
# =============================================================================

import logging
import os
from typing import Any, List, Optional

import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import TrainerCallback

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
    Pull the predicted label from raw model output.
    Takes the first non-empty line after 'Output Tag:' if present,
    otherwise the first non-empty line of the whole output.
    """
    text = generated_text.strip()
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
    """Log label distribution tables once at run start."""
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
        f"Logged label distributions — train: {len(train_df)} rows, "
        f"eval: {len(eval_df)} rows."
    )


def log_lora_efficiency(model, cfg: dict) -> None:
    """
    Log trainable vs frozen parameter counts.
    Scalars go to run summary; two-row table renders as a W&B bar chart.
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

    breakdown = pd.DataFrame(
        [
            {"param_group": "trainable (LoRA)", "count": trainable},
            {"param_group": "frozen (base)", "count": frozen},
        ]
    )
    wandb.log({"lora/param_breakdown": wandb.Table(dataframe=breakdown)})

    logger.info(f"LoRA: {trainable:,} trainable / {total:,} total ({pct:.2f}%)")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class WandbLossCallback(TrainerCallback):
    """
    Re-logs train and eval loss under a shared 'loss/' prefix so W&B
    places both curves in the same panel automatically.

        loss/train   ← emitted every logging_steps
        loss/eval    ← emitted whenever the Trainer runs eval

    The Trainer already logs eval_loss as 'eval/loss' in its own panel.
    We intercept it in on_log and also emit 'loss/eval' to get the combined view.
    No manual step= — the Trainer owns the W&B step axis.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if wandb.run is None or logs is None:
            return control

        payload = {}

        if "loss" in logs:
            payload["loss/train"] = logs["loss"]

        if "eval_loss" in logs:
            payload["loss/eval"] = logs["eval_loss"]

        if "learning_rate" in logs:
            payload["train/learning_rate"] = logs["learning_rate"]

        if payload:
            wandb.log(payload)  # no step= intentional

        return control


class WandbEvalPredictionCallback(TrainerCallback):
    """
    Stub — kept so existing train.py imports don't break.

    Mid-training generation is skipped because gradient checkpointing is
    active during training. All generation-based evaluation happens in
    run_final_evaluation(), called from train.py after trainer.train().
    """

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
        return control  # intentionally no-op


# ---------------------------------------------------------------------------
# Post-training evaluation
# ---------------------------------------------------------------------------


def run_final_evaluation(model, tokenizer, eval_dataset, output_dir: str) -> None:
    """
    Run generation on the full eval set after training is complete.

    Called from train.py immediately after trainer.train() returns and
    before wandb.finish(). Switches the model to unsloth inference mode
    for clean, reliable generation.

    Saves {output_dir}/predictions_final.csv and logs to W&B:
        predictions/final_table        text | true_label | predicted_label | correct
        predictions/final_accuracy     scalar
        predictions/final_macro_f1     scalar  ← key metric for imbalanced classes
        predictions/confusion_matrix   plot
    """
    if wandb.run is None:
        logger.warning("W&B not initialised — skipping final evaluation.")
        return

    logger.info("Running post-training evaluation on full eval set...")

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
    rows = []

    for row in eval_dataset:
        gold_label = str(row["label"]).strip()
        formatted_text = row["formatted_text"]

        # Strip the gold answer → pure inference prompt
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

        # Decode only the newly generated tokens, not the echoed prompt
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred_label = _extract_label(raw_output)

        rows.append(
            {
                "text": row["text"],
                "true_label": gold_label,
                "predicted_label": pred_label,
                "correct": pred_label == gold_label,
            }
        )

    pred_df = pd.DataFrame(rows)

    # Save CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "predictions_final.csv")
    pred_df.to_csv(csv_path, index=False)
    logger.info(f"Saved predictions → {csv_path}")

    # Metrics
    accuracy = pred_df["correct"].mean()
    macro_f1 = f1_score(
        pred_df["true_label"].tolist(),
        pred_df["predicted_label"].tolist(),
        average="macro",
        zero_division=0,
    )
    logger.info(f"Final eval — accuracy: {accuracy:.3f}  macro_f1: {macro_f1:.3f}")

    # Log to W&B
    wandb.log(
        {
            "predictions/final_table": wandb.Table(dataframe=pred_df),
            "predictions/final_accuracy": accuracy,
            "predictions/final_macro_f1": macro_f1,
        }
    )
    wandb.run.summary["predictions/final_accuracy"] = accuracy
    wandb.run.summary["predictions/final_macro_f1"] = macro_f1

    try:
        wandb.log(
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
