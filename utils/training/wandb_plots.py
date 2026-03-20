# =============================================================================
# utils/wandb_plots.py
# =============================================================================
import logging
from collections import Counter
from typing import List, Optional

import pandas as pd
import torch
import wandb
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


def _safe_to_pandas(dataset, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convert a Hugging Face dataset split to pandas safely.
    """
    if columns is None:
        columns = dataset.column_names
    available = [c for c in columns if c in dataset.column_names]
    return dataset.select_columns(available).to_pandas()


def log_dataset_overview(train_dataset, eval_dataset) -> None:
    """
    Log basic dataset stats to W&B:
    - label distribution (train/eval)
    - text length distribution (train/eval)
    """
    if wandb.run is None:
        logger.warning("W&B is not initialized. Skipping dataset overview logging.")
        return

    train_df = _safe_to_pandas(train_dataset, ["text", "label"])
    eval_df = _safe_to_pandas(eval_dataset, ["text", "label"])

    # -------------------------
    # Label distribution tables
    # -------------------------
    train_label_counts = (
        train_df["label"].value_counts(dropna=False).reset_index()
        .rename(columns={"index": "label", "label": "count"})
    )
    train_label_counts.columns = ["label", "count"]

    eval_label_counts = (
        eval_df["label"].value_counts(dropna=False).reset_index()
        .rename(columns={"index": "label", "label": "count"})
    )
    eval_label_counts.columns = ["label", "count"]

    wandb.log(
        {
            "dataset/train_label_distribution": wandb.Table(dataframe=train_label_counts),
            "dataset/eval_label_distribution": wandb.Table(dataframe=eval_label_counts),
        }
    )

    # -------------------------
    # Text length distributions
    # -------------------------
    train_lengths = train_df["text"].fillna("").astype(str).apply(len).tolist()
    eval_lengths = eval_df["text"].fillna("").astype(str).apply(len).tolist()

    wandb.log(
        {
            "dataset/train_text_length_hist": wandb.Histogram(train_lengths),
            "dataset/eval_text_length_hist": wandb.Histogram(eval_lengths),
            "dataset/train_size": len(train_df),
            "dataset/eval_size": len(eval_df),
        }
    )

    # -------------------------
    # Raw example tables
    # -------------------------
    wandb.log(
        {
            "dataset/train_samples": wandb.Table(dataframe=train_df.head(50)),
            "dataset/eval_samples": wandb.Table(dataframe=eval_df.head(50)),
        }
    )

    logger.info("Logged dataset overview to W&B.")


def extract_label_from_generation(generated_text: str) -> str:
    """
    Best-effort extraction of a predicted label from generated text.
    Adjust this if your model output format changes.
    """
    if not generated_text:
        return ""

    text = generated_text.strip()

    # If model echoes "Output Tag:" then keep only what comes after it
    if "Output Tag:" in text:
        text = text.split("Output Tag:")[-1].strip()

    # Take the first non-empty line as label prediction
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line

    return text


def generate_predictions_table(
    model,
    tokenizer,
    eval_dataset,
    max_samples: int = 50,
    max_new_tokens: int = 20,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run generation on a sample of eval data and return a dataframe with:
    text, true_label, predicted_label, correct
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    rows = []

    sample_count = min(max_samples, len(eval_dataset))
    sample_ds = eval_dataset.select(range(sample_count))

    for row in sample_ds:
        prompt = row["formatted_text"]

        # Remove the gold answer from the end to turn it into an inference prompt
        # Assumes formatted_text ends with the assistant answer.
        gold_label = str(row["label"]).strip()
        inference_prompt = prompt.rsplit(gold_label, 1)[0].rstrip()

        inputs = tokenizer(inference_prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_label = extract_label_from_generation(decoded)

        rows.append(
            {
                "text": row["text"],
                "true_label": gold_label,
                "predicted_label": pred_label,
                "correct": pred_label == gold_label,
            }
        )

    return pd.DataFrame(rows)


def log_prediction_results(
    model,
    tokenizer,
    eval_dataset,
    max_samples: int = 50,
) -> None:
    """
    Log:
    - prediction table
    - simple accuracy on sampled examples
    - confusion matrix
    """
    if wandb.run is None:
        logger.warning("W&B is not initialized. Skipping prediction logging.")
        return

    device = next(model.parameters()).device
    pred_df = generate_predictions_table(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        max_samples=max_samples,
        device=device,
    )

    if pred_df.empty:
        logger.warning("Prediction dataframe is empty. Skipping logging.")
        return

    accuracy = pred_df["correct"].mean()

    wandb.log(
        {
            "predictions/sample_table": wandb.Table(dataframe=pred_df),
            "predictions/sample_accuracy": accuracy,
        }
    )

    # W&B confusion matrix
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
    except Exception as e:
        logger.warning(f"Could not log confusion matrix: {e}")

    logger.info("Logged prediction table and confusion matrix to W&B.")


class WandbEvalPredictionCallback(TrainerCallback):
    """
    Logs prediction samples + confusion matrix at evaluation points.

    Note:
    This can be expensive because it runs generation during evaluation.
    Use a small max_samples value.
    """

    def __init__(self, tokenizer, eval_dataset, max_samples: int = 25):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.max_samples = max_samples

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control

        try:
            log_prediction_results(
                model=model,
                tokenizer=self.tokenizer,
                eval_dataset=self.eval_dataset,
                max_samples=self.max_samples,
            )
        except Exception as e:
            logger.warning(f"Failed to log eval predictions to W&B: {e}")

        return control