#!/usr/bin/env python
# =============================================================================
# scripts/train.py — Stage II Distillation
# =============================================================================
#
# CONFIG HIERARCHY (each layer wins over the previous):
#   1. config/model_config_base.yaml      — shared A100 production defaults
#   2. config/model_config.yaml           — model-specific overrides (Llama/Mistral)
#   3. config/pipeline_test_config.yaml   — test overrides (only when PIPELINE_TEST=True)
#
# THE ONLY TWO LINES YOU EVER CHANGE BEFORE A RUN:
#   PIPELINE_TEST = True/False
#   GPU_TIER      = "t4" / "a100"
#
# GPU_TIER is kept here (not in YAML) because it is a runtime label used only
# for W&B run naming — it is not a training hyperparameter.
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
import unsloth  # noqa: F401 — must be imported before transformers
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from trl import SFTConfig, SFTTrainer

import wandb as _wandb

wandb: Any = _wandb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from utils.training.llm_handler import (  # noqa: E402
    format_prompt,
    load_labels_config,
    load_model_for_training,
    push_model_to_hub,
)
from utils.training.wandb_plots import (  # noqa: E402
    FinalEvalWandbCallback,
    MidTrainingEvalCallback,
    PerplexityAndLossCallback,
    log_dataset_overview,
    log_lora_efficiency,
    save_loss_plot,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress the noisy transformers AttentionMask deprecation warning
logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)

# =============================================================================
# THE ONLY TWO LINES YOU EVER CHANGE
# =============================================================================
PIPELINE_TEST = True
GPU_TIER = "t4"  # "t4" for pipeline test | "a100" for real run


# =============================================================================
# Config loading
# =============================================================================


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on conflict."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(override_path: str, pipeline_test: bool = False) -> dict:
    """
    Merge configs in three layers (each wins over the previous):
      1. model_config_base.yaml      — shared A100 production defaults
      2. model_config.yaml           — model-specific overrides (Llama / Mistral)
      3. pipeline_test_config.yaml   — test overrides (only when pipeline_test=True)
    """
    config_dir = os.path.dirname(override_path)
    base_path = os.path.join(config_dir, "model_config_base.yaml")
    test_path = os.path.join(config_dir, "pipeline_test_config.yaml")

    with open(base_path) as f:
        cfg = yaml.safe_load(f)

    with open(override_path) as f:
        cfg = _deep_merge(cfg, yaml.safe_load(f))

    if pipeline_test:
        with open(test_path) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f))
        logger.info("Pipeline test mode — applied pipeline_test_config.yaml overrides.")

    return cfg


# =============================================================================
# W&B initialisation
# =============================================================================


def init_wandb(cfg: dict, pipeline_test: bool, gpu_tier: str) -> Any:
    wb = cfg["wandb"]
    os.environ["WANDB_PROJECT"] = wb["project"]
    os.environ["WANDB_HTTP_TIMEOUT"] = "60"

    base_name = wb.get("run_name") or "run"
    run_name = (
        f"TEST-{gpu_tier.upper()}-{base_name}"
        if pipeline_test
        else f"DISTILL-{gpu_tier.upper()}-{base_name}"
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


# =============================================================================
# Data loading
# =============================================================================


def load_data(cfg: dict, tokenizer):
    data_cfg = cfg["data"]
    labels_cfg = load_labels_config(data_cfg["labels_config"])

    dataset = load_dataset(
        "csv",
        data_files={
            "train": data_cfg["train_file"],
            "eval": data_cfg["eval_file"],
        },
        sep=";",
        encoding="latin-1",
    )

    for split in ("train", "eval"):
        for col in ("text", "label"):
            if col not in dataset[split].column_names:
                raise ValueError(
                    f"Column '{col}' missing from {split} CSV. "
                    f"Found columns: {dataset[split].column_names}"
                )

    dataset = dataset.map(
        lambda row: {
            "formatted_text": format_prompt(
                row["text"], row["label"], tokenizer, labels_cfg
            )
        },
        desc="Formatting distillation prompts",
    )

    logger.info(
        f"Dataset loaded — train: {len(dataset['train']):,} | "
        f"eval: {len(dataset['eval']):,} | "
        f"max_seq_length: {cfg['model']['max_seq_length']}"
    )
    return dataset


# =============================================================================
# VRAM utility
# =============================================================================


def free_vram(label: str = "") -> None:
    torch.cuda.empty_cache()
    if label:
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(
            f"[VRAM {label}] allocated={allocated:.2f} GB | reserved={reserved:.2f} GB"
        )


# =============================================================================
# Main training function
# =============================================================================


def train(cfg: dict, run: Any) -> None:
    model, tokenizer = load_model_for_training(cfg)
    dataset = load_data(cfg, tokenizer)

    log_lora_efficiency(model, cfg)
    log_dataset_overview(dataset["train"], dataset["eval"])

    t = cfg["training"]
    train_size = len(dataset["train"])

    batch_size = t["per_device_train_batch_size"]
    eval_batch_size = t["per_device_eval_batch_size"]
    grad_accum = t["gradient_accumulation_steps"]
    num_epochs = t["num_train_epochs"]

    # Warmup is computed dynamically from the actual dataset size and schedule
    steps_per_epoch = max(1, train_size // (batch_size * grad_accum))
    total_steps = max(1, steps_per_epoch * num_epochs)
    warmup_steps = max(1, int(total_steps * t["warmup_ratio"]))

    logger.info(
        f"Training schedule — epochs: {num_epochs} | "
        f"steps/epoch: {steps_per_epoch} | total_steps: {total_steps} | "
        f"warmup_steps: {warmup_steps} | effective_batch: {batch_size * grad_accum}"
    )

    training_args = SFTConfig(
        output_dir=t["output_dir"],
        # Batch and gradient settings
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        # Learning rate schedule
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_steps=warmup_steps,
        num_train_epochs=num_epochs,
        # Precision — A100 uses bf16, T4 uses fp16 (set in YAML)
        bf16=t["bf16"],
        fp16=t["fp16"],
        # Sequence length — enforced at trainer level, not just model level
        max_seq_length=cfg["model"]["max_seq_length"],
        # Logging and evaluation cadence (all driven from YAML)
        logging_steps=t["logging_steps"],
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        save_strategy="steps",
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        # Checkpoint selection — always use lowest eval_loss
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Infrastructure
        report_to="wandb",
        seed=t["seed"],
        dataloader_num_workers=t["dataloader_num_workers"],
        dataset_text_field="formatted_text",
        # Hub push is handled manually at the end via push_model_to_hub()
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

    # FinalEvalWandbCallback fires once after training ends on the best checkpoint.
    # It runs generation on the full eval set and logs classification metrics.
    # Inserted at position 0 so it fires before the default end-of-training callbacks.
    trainer.callback_handler.callbacks.insert(
        0,
        FinalEvalWandbCallback(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=dataset["eval"],
            output_dir=t["output_dir"],
            run=run,
        ),
    )

    # PerplexityAndLossCallback derives perplexity from eval_loss at every eval step.
    trainer.add_callback(PerplexityAndLossCallback())

    # MidTrainingEvalCallback runs generation on a sample during training.
    # Only enabled on real A100 runs — too slow for pipeline tests and T4.
    if not PIPELINE_TEST:
        mid_eval_cb = MidTrainingEvalCallback(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=dataset["eval"],
            sample_size=50,
        )
        trainer.callback_handler.callbacks.insert(0, mid_eval_cb)

    logger.info("Starting training...")
    free_vram("before training")
    trainer.train()
    logger.info("Training complete.")

    save_loss_plot(trainer.state.log_history, t["output_dir"], run)

    if t.get("push_to_hub", False):
        push_model_to_hub(model, tokenizer, cfg)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Stage II distillation training")
    parser.add_argument(
        "--config",
        default="config/model_config.yaml",
        help="Path to model-specific config (merged on top of model_config_base.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, pipeline_test=PIPELINE_TEST)
    run = init_wandb(cfg, pipeline_test=PIPELINE_TEST, gpu_tier=GPU_TIER)

    try:
        train(cfg, run=run)
    finally:
        run.finish()
