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

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)


# =============================================================================
# THE ONLY TWO LINES YOU EVER CHANGE
# =============================================================================
#
#   PIPELINE_TEST = True   -> 2 epochs, step counts derived from dataset size
#   PIPELINE_TEST = False  -> epochs + step counts from model_config.yaml
#
#   GPU_TIER = "t4"        -> 16GB overrides: seq_len 512, batch 1, no mid-eval
#   GPU_TIER = "a100"      -> 64GB: no overrides, full config, mid-eval enabled
#
# =============================================================================
PIPELINE_TEST = True
GPU_TIER = "t4"  # change to "a100" when switching VM


# =============================================================================
# Per-GPU VRAM budgets
# =============================================================================
#
# T4 (16 GB total):
#   Llama-3.1-8B weights in 4-bit:       ~5.5 GB
#   QLoRA adapter + optimizer states:     ~2.5 GB
#   Activations at seq_len=2048:          ~6.0 GB  <- kills it
#   Activations at seq_len=512:           ~0.4 GB  <- safe
#   Inference pass on top of training:    ~1.5 GB  <- second killer -> disabled
#
# A100 (64 GB total):
#   Everything above at full seq_len:    ~22 GB
#   Headroom for inference callbacks:    ~42 GB remaining — no problem
#
GPU_OVERRIDES: dict[str, dict[str, Any]] = {
    "t4": {
        "max_seq_length": 512,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,  # keeps effective batch = 8 (same as 2*4)
        "run_mid_eval": False,  # inference on top of training = OOM on T4
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
    """
    Explicit VRAM cache flush before any inference callback.
    On T4: difference between OOM and not.
    On A100: ~50ms overhead, good hygiene.
    """
    torch.cuda.empty_cache()
    if label:
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(
            f"[VRAM {label}] allocated={allocated:.2f} GB | reserved={reserved:.2f} GB"
        )


class FinalEvalWandbCallback(TrainerCallback):
    """
    Runs full inference evaluation at on_train_end.
    Inserted at index 0 so it fires before WandbCallback closes the run.
    Flushes VRAM before generating to avoid OOM on smaller GPUs.
    """

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


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(cfg: dict, tokenizer, max_seq_length: int):
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
        f"Train: {len(dataset['train']):,} | "
        f"Eval: {len(dataset['eval']):,} | "
        f"max_seq_length={max_seq_length}"
    )
    return dataset


def init_wandb(cfg: dict, pipeline_test: bool, gpu_tier: str) -> Any:
    wb = cfg["wandb"]
    os.environ["WANDB_PROJECT"] = wb["project"]

    # Silence the GraphQL timeout warning by setting a longer timeout.
    # The default is 19s which is too short for slow corporate networks.
    os.environ["WANDB_HTTP_TIMEOUT"] = "60"

    base_name = wb.get("run_name") or "run"
    run_name = f"TEST-{gpu_tier.upper()}-{base_name}" if pipeline_test else base_name

    tags = list(wb.get("tags", []))
    if pipeline_test:
        tags += ["pipeline-test", gpu_tier]

    run = wandb.init(
        project=wb["project"],
        name=run_name,
        tags=tags,
        config={**cfg, "pipeline_test": pipeline_test, "gpu_tier": gpu_tier},
    )

    # Declare step axes BEFORE any logging happens.
    # All eval/* metrics share the trainer's native "step" counter so they
    # appear in the same W&B section and on the same x-axis as eval/loss.
    # charts/* is reserved for the final PNG image only.
    wandb.define_metric("eval/*", step_metric="epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("charts/*", step_metric="epoch")

    return run


def compute_step_schedule(
    train_size: int,
    batch_size: int,
    grad_accum: int,
    num_epochs: int,
    pipeline_test: bool,
    cfg_training: dict,
) -> dict:
    """
    Pipeline test  -> steps derived from actual dataset size so you always
                      get at least 3-4 logging events and 1 eval/save per epoch.
    Full run       -> taken directly from model_config.yaml.
    """
    steps_per_epoch = max(1, train_size // (batch_size * grad_accum))

    if pipeline_test:
        logging_steps = max(1, steps_per_epoch // 4)
        eval_steps = max(1, steps_per_epoch)
        save_steps = max(1, steps_per_epoch)
        save_total_limit = 2
        logger.info(
            f"[PIPELINE TEST] steps_per_epoch={steps_per_epoch} | "
            f"logging={logging_steps} | eval={eval_steps} | save={save_steps}"
        )
    else:
        logging_steps = cfg_training["logging_steps"]
        eval_steps = cfg_training["eval_steps"]
        save_steps = cfg_training["save_steps"]
        save_total_limit = cfg_training["save_total_limit"]

    return {
        "logging_steps": logging_steps,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
    }


def train(cfg: dict, run: Any) -> None:
    overrides = GPU_OVERRIDES[GPU_TIER]

    # Apply GPU-tier seq_len override before loading the model.
    # FastLanguageModel reads max_seq_length from cfg at load time.
    if overrides["max_seq_length"] is not None:
        orig = cfg["model"]["max_seq_length"]
        cfg["model"]["max_seq_length"] = overrides["max_seq_length"]
        logger.info(
            f"[{GPU_TIER.upper()} override] max_seq_length: {orig} -> "
            f"{overrides['max_seq_length']}"
        )

    model, tokenizer = load_model_for_training(cfg)
    dataset = load_data(cfg, tokenizer, cfg["model"]["max_seq_length"])

    log_lora_efficiency(model, cfg)
    log_dataset_overview(dataset["train"], dataset["eval"])

    t = cfg["training"]
    train_size = len(dataset["train"])
    eval_size = len(dataset["eval"])

    # Resolve batch sizes: GPU override takes priority over yaml
    batch_size = (
        overrides["per_device_train_batch_size"]
        if overrides["per_device_train_batch_size"] is not None
        else t["per_device_train_batch_size"]
    )
    eval_batch_size = (
        overrides["per_device_eval_batch_size"]
        if overrides["per_device_eval_batch_size"] is not None
        else t["per_device_eval_batch_size"]
    )
    grad_accum = (
        overrides["gradient_accumulation_steps"]
        if overrides["gradient_accumulation_steps"] is not None
        else t["gradient_accumulation_steps"]
    )

    # 2 epochs on pipeline test: exercises warmup -> decay -> eval -> save
    num_epochs = 10 if PIPELINE_TEST else t["num_train_epochs"]

    steps_per_epoch = max(1, train_size // (batch_size * grad_accum))
    total_steps = max(1, steps_per_epoch * num_epochs)
    warmup_steps = max(1, int(total_steps * t["warmup_ratio"]))

    step_schedule = compute_step_schedule(
        train_size=train_size,
        batch_size=batch_size,
        grad_accum=grad_accum,
        num_epochs=num_epochs,
        pipeline_test=PIPELINE_TEST,
        cfg_training=t,
    )

    logger.info(
        f"total_steps={total_steps} | warmup={warmup_steps} | epochs={num_epochs} | "
        f"batch={batch_size} | grad_accum={grad_accum} | "
        f"PIPELINE_TEST={PIPELINE_TEST} | GPU_TIER={GPU_TIER}"
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
        # No-op on 20 eval samples but critical for the real run
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

    # ------------------------------------------------------------------
    # Callbacks — inserted at index 0 to fire before WandbCallback
    # closes the run at on_train_end.
    # ------------------------------------------------------------------

    # 1. Final full inference eval — always enabled, VRAM flush built in
    final_eval_cb = FinalEvalWandbCallback(
        model, tokenizer, dataset["eval"], t["output_dir"], run
    )
    trainer.callback_handler.callbacks.insert(0, final_eval_cb)

    # 2. Perplexity + combined loss — always enabled, zero VRAM cost
    #    (reads scalars from trainer logs, no generation)
    trainer.add_callback(PerplexityAndLossCallback())

    # 3. Mid-training label eval — DISABLED on T4, ENABLED on A100
    #
    #    Why disabled on T4:
    #    Training fills ~15 GB. Running model.generate() on top triggers
    #    the exact OOM seen: "Tried to allocate 94 MiB, 79 MiB free".
    #    The final eval callback above still runs because free_vram() is
    #    called first after training completes.
    #
    #    Why safe on A100:
    #    64 GB gives ~40 GB headroom after training state. 50-ticket
    #    inference peaks at ~1.5 GB extra. Fully safe.
    if overrides["run_mid_eval"]:
        mid_sample = min(overrides["mid_eval_sample_size"], eval_size)
        mid_eval_cb = MidTrainingEvalCallback(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=dataset["eval"],
            sample_size=mid_sample,
        )
        trainer.callback_handler.callbacks.insert(0, mid_eval_cb)
        logger.info(f"MidTrainingEvalCallback enabled — sample_size={mid_sample}")
    else:
        logger.info(
            f"MidTrainingEvalCallback DISABLED on {GPU_TIER.upper()} "
            f"(prevents OOM). Will be active on A100."
        )

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
