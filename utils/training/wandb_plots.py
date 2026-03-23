# =============================================================================
# utils/training/wandb_plots.py
# =============================================================================

import logging
import os
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import f1_score

import wandb as _wandb

wandb: Any = _wandb
logger = logging.getLogger(__name__)
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


def save_loss_plot(log_history: list, output_dir: str, run: Any = None) -> None:
    """
    Parses the Hugging Face Trainer's internal log history to plot Train vs Val loss
    and saves it as a PNG file, completely bypassing API timeouts.
    """
    logger.info("Generating Training vs Validation Loss plot...")

    train_epochs, train_losses = [], []
    eval_epochs, eval_losses = [], []

    # Parse the Trainer's memory
    for log in log_history:
        if "loss" in log and "epoch" in log:
            train_epochs.append(log["epoch"])
            train_losses.append(log["loss"])
        if "eval_loss" in log and "epoch" in log:
            eval_epochs.append(log["epoch"])
            eval_losses.append(log["eval_loss"])

    if not train_losses or not eval_losses:
        logger.warning("Not enough data to plot Train vs Val loss. Skipping plot.")
        return

    # Build the plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        train_epochs,
        train_losses,
        label="Train Loss",
        color="steelblue",
        alpha=0.8,
        linewidth=2,
    )
    plt.plot(
        eval_epochs,
        eval_losses,
        label="Validation Loss",
        color="darkorange",
        linewidth=2.5,
        marker="o",
        markersize=6,
    )

    plt.title("Training vs. Validation Loss over Time", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Save to your output directory
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    logger.info(f"Saved loss plot to {save_path}")

    # Optional: Send the finished image back to W&B as an artifact
    active_run = run if run is not None else wandb.run
    if active_run is not None:
        try:
            active_run.log({"charts/final_loss_curve": wandb.Image(save_path)})
        except Exception as exc:
            logger.warning(f"Could not log image to W&B: {exc}")


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
