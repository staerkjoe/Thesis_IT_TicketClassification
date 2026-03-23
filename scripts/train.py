#!/usr/bin/env python
# =============================================================================
# scripts/train.py
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
from transformers import TrainerCallback
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
    log_dataset_overview,
    log_lora_efficiency,
    run_final_evaluation,
    save_loss_plot,
)

# Suppress standard warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# FIX: Suppress the buggy transformers AttentionMask deprecation warning
# that causes the TypeError crash in the logging module
logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)


class FinalEvalWandbCallback(TrainerCallback):
    """
    A custom callback that hooks into the end of training to run inference.
    We will insert this at the FRONT of the callback list so it runs before
    W&B's native callback closes the run.
    """

    def __init__(self, model, tokenizer, eval_dataset, output_dir, run):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.run = run

    def on_train_end(self, args, state, control, **kwargs):
        run_final_evaluation(
            model=self.model,
            tokenizer=self.tokenizer,
            eval_dataset=self.eval_dataset,
            output_dir=self.output_dir,
            run=self.run,
        )


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
    wb = cfg["wandb"]
    os.environ["WANDB_PROJECT"] = wb["project"]
    run = wandb.init(
        project=wb["project"],
        name=wb.get("run_name") or None,
        tags=wb.get("tags", []),
        config=cfg,
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/loss", step_metric="epoch")
    wandb.define_metric("eval/loss", step_metric="epoch")
    return run


def train(cfg: dict, run: wandb.sdk.wandb_run.Run) -> None:
    model, tokenizer = load_model_for_training(cfg)
    dataset = load_data(cfg, tokenizer)

    log_lora_efficiency(model, cfg)
    log_dataset_overview(dataset["train"], dataset["eval"])

    t = cfg["training"]

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
        report_to="wandb",  # Enable out-of-the-box W&B!
        seed=t["seed"],
        dataloader_num_workers=t["dataloader_num_workers"],
        dataset_text_field="formatted_text",
        push_to_hub=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        args=training_args,
        dataset_kwargs={"append_concat_token": False},
    )

    # Insert our custom evaluation callback at the FRONT of the callback list (index 0)
    # This guarantees it executes BEFORE WandbCallback finishes the run.
    eval_cb = FinalEvalWandbCallback(
        model, tokenizer, dataset["eval"], t["output_dir"], run
    )
    trainer.callback_handler.callbacks.insert(0, eval_cb)

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")

    save_loss_plot(trainer.state.log_history, t["output_dir"], run)

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
        run.finish()
