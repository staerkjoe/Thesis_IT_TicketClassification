#!/usr/bin/env python
# =============================================================================
# scripts/train.py
# poetry run python scripts/train.py --config config/model_config.yaml
#
# W&B lifecycle
# ─────────────
#   wandb.init()              called in init_wandb()
#   trainer.remove_callback() removes WandbCallback AFTER SFTTrainer is built
#                             so the Trainer never touches the W&B run
#   run_final_evaluation()    logs predictions while run is still open
#   run.finish()              __main__ try/finally, always last
#
# WHY remove_callback() instead of report_to=[]
#   report_to=[] stops the Trainer from adding WandbCallback at init time,
#   but SFTTrainer (trl) and the HF Trainer still detect an active wandb run
#   in the environment and can re-attach or trigger shutdown hooks.
#   Calling trainer.remove_callback(WandbCallback) after construction is the
#   only reliable way to guarantee the Trainer never calls wandb.finish().
#   We keep report_to=[] as well for belt-and-suspenders.
# =============================================================================

import argparse
import logging
import os
import pathlib
import sys
import warnings
from typing import Any

import unsloth  # noqa: F401 — must be first
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from transformers.integrations import WandbCallback
from trl import SFTConfig, SFTTrainer

import wandb as _wandb

wandb: Any = _wandb
logger = logging.getLogger(__name__)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from utils.training.llm_handler import (  # noqa: E402
    format_prompt,
    load_labels_config,
    load_model_for_training,
    push_model_to_hub,
)
from utils.training.wandb_plots import (  # noqa: E402
    WandbEvalPredictionCallback,
    WandbLossCallback,
    init_wandb_metrics,
    log_dataset_overview,
    log_lora_efficiency,
    run_final_evaluation,
)

# Suppress the transformers attention mask deprecation warning.
# It uses %-style logging with a FutureWarning class as a positional arg,
# which causes a TypeError in Python's logging formatter — not our code.
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(cfg: dict, tokenizer):
    data_cfg = cfg["data"]
    labels_cfg = load_labels_config(data_cfg["labels_config"])

    dataset = load_dataset(
        "csv",
        data_files={"train": data_cfg["train_file"], "eval": data_cfg["eval_file"]},
    )

    for split in ("train", "eval"):
        for col in ("text", "label"):
            if col not in dataset[split].column_names:
                raise ValueError(f"Column '{col}' missing from {split} CSV.")

    dataset = dataset.map(
        lambda row: {
            "formatted_text": format_prompt(
                row["text"], row["label"], tokenizer, labels_cfg
            )
        },
        desc="Formatting prompts",
    )
    logger.info(
        f"Train: {len(dataset['train']):,} rows | Eval: {len(dataset['eval']):,} rows"
    )
    return dataset


def init_wandb(cfg: dict) -> wandb.sdk.wandb_run.Run:
    """
    Initialise W&B, define metric axes, and return the Run object.
    The Run object is passed explicitly through the call stack so we never
    rely on the wandb.run global, which W&B's atexit handler can null out.
    """
    wb = cfg["wandb"]
    os.environ["WANDB_PROJECT"] = wb["project"]
    run = wandb.init(
        project=wb["project"],
        name=wb.get("run_name") or None,
        tags=wb.get("tags", []),
        config=cfg,
    )
    init_wandb_metrics()
    return run


def train(cfg: dict, run: wandb.sdk.wandb_run.Run) -> None:
    """
    Main training function. Does NOT call wandb.finish() or run.finish().
    That is __main__'s responsibility so the run stays open for evaluation.
    """
    model, tokenizer = load_model_for_training(cfg)
    dataset = load_data(cfg, tokenizer)

    log_lora_efficiency(model, cfg)
    log_dataset_overview(dataset["train"], dataset["eval"])

    t = cfg["training"]
    wb_cfg = cfg.get("wandb", {})

    steps_per_epoch = len(dataset["train"]) // (
        t["per_device_train_batch_size"] * t["gradient_accumulation_steps"]
    )
    total_steps = max(1, steps_per_epoch * t["num_train_epochs"])
    warmup_steps = max(1, int(total_steps * t["warmup_ratio"]))
    logger.info(f"Total steps: {total_steps} | Warmup steps: {warmup_steps}")

    training_args = SFTConfig(
        output_dir=t["output_dir"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_steps=warmup_steps,
        num_train_epochs=t["num_train_epochs"],
        bf16=t["bf16"],
        fp16=t["fp16"],
        logging_steps=t["logging_steps"],
        eval_strategy=t["eval_strategy"],
        eval_steps=t["eval_steps"],
        save_strategy=t["save_strategy"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        report_to=[],  # belt-and-suspenders: tells Trainer not to add WandbCallback
        seed=t["seed"],
        dataloader_num_workers=t["dataloader_num_workers"],
        dataset_text_field="formatted_text",
        push_to_hub=False,
    )

    callbacks = [WandbLossCallback(run=run)]
    if wb_cfg.get("log_prediction_samples", False):
        callbacks.append(
            WandbEvalPredictionCallback(
                tokenizer=tokenizer,
                eval_dataset=dataset["eval"],
                max_samples=wb_cfg.get("prediction_sample_size", 25),
                output_dir=t["output_dir"],
            )
        )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        args=training_args,
        callbacks=callbacks,
        dataset_kwargs={"append_concat_token": False},
    )

    # Belt-and-suspenders: explicitly remove WandbCallback even if the Trainer
    # somehow added it despite report_to=[]. This is the key fix — without this,
    # SFTTrainer can still detect the active wandb run and attach its own callback
    # which calls wandb.finish() when training ends.
    trainer.remove_callback(WandbCallback)
    logger.info("WandbCallback removed from Trainer — W&B lifecycle is fully manual.")

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")

    # Run is still open here because WandbCallback was removed.
    # Pass run explicitly — do not rely on wandb.run global.
    run_final_evaluation(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=dataset["eval"],
        output_dir=t["output_dir"],
        run=run,
    )

    if t.get("push_to_hub", False):
        push_model_to_hub(model, tokenizer, cfg)


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/model_config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    run = init_wandb(cfg)
    try:
        train(cfg, run=run)
    finally:
        # Only place run.finish() is called. The try/finally guarantees it
        # runs even if train() raises an exception mid-way.
        run.finish()
