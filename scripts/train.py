#!/usr/bin/env python
# =============================================================================
# scripts/train.py - Stage II Distillation Version
# =============================================================================

import argparse
import copy
import logging
import os
import pathlib
import sys
import warnings
from typing import Any

import torch
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
    MidTrainingEvalCallback,
    PerplexityAndLossCallback,
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
logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)

# =============================================================================
# THE ONLY TWO LINES YOU EVER CHANGE
# =============================================================================
PIPELINE_TEST = True
GPU_TIER = "t4"  # change to "a100" when switching VM

# =============================================================================
# Per-GPU VRAM budgets
# =============================================================================
GPU_OVERRIDES: dict[str, dict[str, Any]] = {
    "t4": {
        "max_seq_length": 512,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "run_mid_eval": False,
        "mid_eval_sample_size": 0,
    },
    "a100": {
        "max_seq_length": None,  # use model_config.yaml value
        "per_device_train_batch_size": None,
        "per_device_eval_batch_size": None,
        "gradient_accumulation_steps": None,
        "run_mid_eval": True,
        "mid_eval_sample_size": 50,
    },
}


def free_vram(label: str = "") -> None:
    torch.cuda.empty_cache()
    if label:
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(
            f"[VRAM {label}] allocated={allocated:.2f} GB | reserved={reserved:.2f} GB"
        )


class FinalEvalWandbCallback(TrainerCallback):
    def __init__(self, model, tokenizer, eval_dataset, output_dir, run):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.run = run

    def on_train_end(self, args, state, control, **kwargs):
        free_vram("before final eval")
        run_final_evaluation(
            model=self.model,
            tokenizer=self.tokenizer,
            eval_dataset=self.eval_dataset,
            output_dir=self.output_dir,
            run=self.run,
        )


def load_config(override_path: str) -> dict:
    """Deep-merge base config with model-specific override."""
    base_path = os.path.join(os.path.dirname(override_path), "model_config_base.yaml")

    with open(base_path) as f:
        base = yaml.safe_load(f)
    with open(override_path) as f:
        override = yaml.safe_load(f)

    return _deep_merge(base, override)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on conflict."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_data(cfg: dict, tokenizer, max_seq_length: int):
    data_cfg = cfg["data"]
    # Dynamic Loading: This loads your 'prompt_template_student.yaml'
    labels_cfg = load_labels_config(data_cfg["labels_config"])

    dataset = load_dataset(
        "csv",
        data_files={"train": data_cfg["train_file"], "eval": data_cfg["eval_file"]},
        sep=";",
    )

    # CHECK: Ensure 'text' (description) and 'label' (Reasoning+Tag) exist
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
        desc="Formatting Distillation Prompts",
    )
    logger.info(
        f"Train: {len(dataset['train']):,} | "
        f"Eval: {len(dataset['eval']):,} | "
        f"max_seq_length={max_seq_length}"
    )
    return dataset


def init_wandb(cfg: dict, pipeline_test: bool, gpu_tier: str) -> Any:
    wb = cfg["wandb"]
    os.environ["WANDB_PROJECT"] = wb["project"]
    os.environ["WANDB_HTTP_TIMEOUT"] = "60"

    base_name = wb.get("run_name") or "run"
    run_name = (
        f"DISTILL-{gpu_tier.upper()}-{base_name}"
        if not pipeline_test
        else f"TEST-{gpu_tier.upper()}-{base_name}"
    )

    tags = list(wb.get("tags", []))
    if pipeline_test:
        tags += ["pipeline-test", gpu_tier]

    run = wandb.init(
        project=wb["project"],
        name=run_name,
        tags=tags,
        config={**cfg, "pipeline_test": pipeline_test, "gpu_tier": gpu_tier},
    )
    wandb.define_metric("eval/*", step_metric="epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("charts/*", step_metric="epoch")
    return run


def compute_step_schedule(
    train_size, batch_size, grad_accum, num_epochs, pipeline_test, cfg_training
):
    steps_per_epoch = max(1, train_size // (batch_size * grad_accum))
    if pipeline_test:
        return {
            "logging_steps": max(1, steps_per_epoch // 4),
            "eval_steps": max(1, steps_per_epoch),
            "save_steps": max(1, steps_per_epoch),
            "save_total_limit": 2,
        }
    return {
        "logging_steps": cfg_training["logging_steps"],
        "eval_steps": cfg_training["eval_steps"],
        "save_steps": cfg_training["save_steps"],
        "save_total_limit": cfg_training["save_total_limit"],
    }


def train(cfg: dict, run: Any) -> None:
    overrides = GPU_OVERRIDES[GPU_TIER]
    if overrides["max_seq_length"] is not None:
        cfg["model"]["max_seq_length"] = overrides["max_seq_length"]

    model, tokenizer = load_model_for_training(cfg)
    dataset = load_data(cfg, tokenizer, cfg["model"]["max_seq_length"])

    log_lora_efficiency(model, cfg)
    log_dataset_overview(dataset["train"], dataset["eval"])

    t = cfg["training"]
    # FIXED: Removed unused eval_size variable
    train_size = len(dataset["train"])

    batch_size = (
        t["per_device_train_batch_size"]
        if overrides["per_device_train_batch_size"] is None
        else overrides["per_device_train_batch_size"]
    )
    eval_batch_size = (
        t["per_device_eval_batch_size"]
        if overrides["per_device_eval_batch_size"] is None
        else overrides["per_device_eval_batch_size"]
    )
    grad_accum = (
        t["gradient_accumulation_steps"]
        if overrides["gradient_accumulation_steps"] is None
        else overrides["gradient_accumulation_steps"]
    )

    num_epochs = 2 if PIPELINE_TEST else t["num_train_epochs"]

    steps_per_epoch = max(1, train_size // (batch_size * grad_accum))
    total_steps = max(1, steps_per_epoch * num_epochs)
    warmup_steps = max(1, int(total_steps * t["warmup_ratio"]))

    step_schedule = compute_step_schedule(
        train_size, batch_size, grad_accum, num_epochs, PIPELINE_TEST, t
    )

    training_args = SFTConfig(
        output_dir=t["output_dir"],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_steps=warmup_steps,
        num_train_epochs=num_epochs,
        bf16=t["bf16"],
        fp16=t["fp16"],
        logging_steps=step_schedule["logging_steps"],
        eval_strategy="steps",
        eval_steps=step_schedule["eval_steps"],
        save_strategy="steps",
        save_steps=step_schedule["save_steps"],
        save_total_limit=step_schedule["save_total_limit"],
        report_to="wandb",
        seed=t["seed"],
        dataloader_num_workers=t["dataloader_num_workers"],
        dataset_text_field="formatted_text",
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        args=training_args,
        dataset_kwargs={"append_concat_token": False},
    )

    trainer.callback_handler.callbacks.insert(
        0,
        FinalEvalWandbCallback(model, tokenizer, dataset["eval"], t["output_dir"], run),
    )
    trainer.add_callback(PerplexityAndLossCallback())

    if overrides["run_mid_eval"]:
        mid_eval_cb = MidTrainingEvalCallback(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=dataset["eval"],
            sample_size=overrides["mid_eval_sample_size"],
        )
        trainer.callback_handler.callbacks.insert(0, mid_eval_cb)

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
    run = init_wandb(cfg, pipeline_test=PIPELINE_TEST, gpu_tier=GPU_TIER)
    try:
        train(cfg, run=run)
    finally:
        run.finish()
