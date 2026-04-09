#!/usr/bin/env python
# =============================================================================
# scripts/eval_final.py — Post-training evaluation with batched inference
#
# Loads the best checkpoint, runs generation in batches on the full eval set,
# and logs all metrics + confusion matrix to the existing W&B run.
#
# Run:
#   TORCHDYNAMO_DISABLE=1 poetry run python scripts/eval_final.py \
#       --config config/model_config_mistral.yaml \
#       --checkpoint outputs/checkpoints/checkpoint-2000 \
#       --wandb_run_id e5mnr5ty \
#       --batch_size 16
# =============================================================================

import argparse
import logging
import os
import pathlib
import sys

import yaml
from datasets import load_dataset
from dotenv import load_dotenv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import wandb  # noqa: E402
from utils.training.llm_handler import format_prompt, load_labels_config  # noqa: E402
from utils.training.wandb_plots import run_final_evaluation  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(base_path: str, override_path: str) -> dict:
    import copy

    def deep_merge(base, override):
        result = copy.deepcopy(base)
        for key, val in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(val, dict)
            ):
                result[key] = deep_merge(result[key], val)
            else:
                result[key] = val
        return result

    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    with open(override_path) as f:
        cfg = deep_merge(cfg, yaml.safe_load(f))
    return cfg


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/model_config_mistral.yaml")
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/checkpoint-2000",
        help="Path to best checkpoint (LoRA adapters)",
    )
    parser.add_argument(
        "--wandb_run_id", required=True, help="W&B run ID to resume (e.g. e5mnr5ty)"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    cfg = load_config("config/model_config_base.yaml", args.config)
    m = cfg["model"]
    data_cfg = cfg["data"]

    # --- Resume the existing W&B run ---
    run = wandb.init(
        project=cfg["wandb"]["project"],
        id=args.wandb_run_id,
        resume="must",
    )
    logger.info(f"Resumed W&B run: {args.wandb_run_id}")

    # --- Load model from best checkpoint ---
    from unsloth import FastLanguageModel

    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.checkpoint,
        max_seq_length=m["max_seq_length"],
        load_in_4bit=m["load_in_4bit"],
        token=os.environ.get("HF_TOKEN"),
    )
    FastLanguageModel.for_inference(model)
    logger.info("Model loaded and switched to inference mode.")

    # --- Load eval dataset ---
    labels_cfg = load_labels_config(data_cfg["labels_config"])
    dataset = load_dataset(
        "csv",
        data_files={"eval": data_cfg["eval_file"]},
        sep=data_cfg.get("sep", ","),
        encoding="latin-1",
    )
    eval_dataset = dataset["eval"].map(
        lambda row: {
            "formatted_text": format_prompt(
                row["text"], row["label"], tokenizer, labels_cfg
            )
        },
        desc="Formatting eval prompts",
    )
    logger.info(f"Eval dataset: {len(eval_dataset):,} samples")

    # --- Run evaluation ---
    run_final_evaluation(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        output_dir=cfg["training"]["output_dir"],
        run=run,
        batch_size=args.batch_size,
    )

    run.finish()
    logger.info("Done.")


if __name__ == "__main__":
    main()
