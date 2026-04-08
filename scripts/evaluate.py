#!/usr/bin/env python
# =============================================================================
# scripts/evaluate.py — Post-training evaluation pipeline
# =============================================================================
#
# Evaluates a fine-tuned distillation model against one of three dataset types
# and logs all metrics to a dedicated W&B eval run.
#
# CONFIG HIERARCHY (same as train.py):
#   1. config/model_config_base.yaml  — shared defaults
#   2. config/model_config.yaml       — model-specific (Llama / Mistral)
#   3. config/pipeline_test_config.yaml — overrides when PIPELINE_TEST=True
#
# Example runs:
#
#   # Silver eval (held-out 20%)
#   poetry run python scripts/evaluate.py \
#     --config config/model_config_llama.yaml \
#     --dataset-type silver \
#     --data-file data/train/final_eval_data.csv
#
#   # Fresh unseen test set
#   poetry run python scripts/evaluate.py \
#     --config config/model_config_mistral.yaml \
#     --dataset-type fresh \
#     --data-file data/test/fresh_test_data.csv
#
#   # Gold standard (50 expert-labeled tickets)
#   poetry run python scripts/evaluate.py \
#     --config config/model_config_llama.yaml \
#     --dataset-type gold \
#     --data-file data/gold/gold_standard.csv \
#     --human-label-col human_label
#
# THE TWO LINES YOU CHANGE BEFORE A RUN (same convention as train.py):
#   PIPELINE_TEST = True/False
#   GPU_TIER      = "t4" / "a100"
# =============================================================================

import argparse
import copy
import logging
import os
import pathlib
import sys
import warnings
from typing import Any

import pandas as pd
import unsloth  # noqa: F401 — must be first
import yaml
from dotenv import load_dotenv

import wandb as _wandb

wandb: Any = _wandb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from utils.training.llm_handler import (  # noqa: E402
    load_labels_config,
    load_model_for_inference,
)
from utils.training.wandb_eval import (  # noqa: E402
    log_evaluation_to_wandb,
    run_inference_batch,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# =============================================================================
# THE TWO LINES YOU CHANGE BEFORE A RUN
# =============================================================================
PIPELINE_TEST = False
GPU_TIER = "a100"  # "t4" for test | "a100" for real run


# =============================================================================
# Config loading (identical three-layer merge as train.py)
# =============================================================================


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(override_path: str, pipeline_test: bool = False) -> dict:
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
# W&B initialisation for eval runs
# =============================================================================


def init_wandb_eval(
    cfg: dict,
    dataset_type: str,
    pipeline_test: bool,
    gpu_tier: str,
) -> Any:
    """
    Open a W&B run dedicated to evaluation.
    Separate from training runs so eval metrics are not mixed with train curves.
    """
    wb = cfg["wandb"]
    os.environ["WANDB_PROJECT"] = wb["project"]
    os.environ["WANDB_HTTP_TIMEOUT"] = "60"

    model_name = cfg["model"].get("hub_model_id", "unknown").split("/")[-1]
    prefix = "TEST-EVAL" if pipeline_test else "EVAL"
    run_name = f"{prefix}-{dataset_type.upper()}-{gpu_tier.upper()}-{model_name}"

    tags = list(wb.get("tags", []))
    tags += ["eval", dataset_type]
    if pipeline_test:
        tags += ["pipeline-test", gpu_tier]

    run = wandb.init(
        project=wb["project"],
        name=run_name,
        tags=tags,
        config={
            **cfg,
            "eval_dataset_type": dataset_type,
            "pipeline_test": pipeline_test,
            "gpu_tier": gpu_tier,
        },
    )
    return run


# =============================================================================
# Data loading
# =============================================================================


def load_eval_data(data_file: str, sep: str = ",") -> pd.DataFrame:
    """
    Load a CSV for evaluation.

    Expected columns (minimum):
        text    — the ticket description
        label   — the teacher's full output (Reasoning: ... Tag: ... Label_Confidence: ...)

    Optional columns:
        human_label — expert-assigned tag (gold standard only)
    """
    df = pd.read_csv(data_file, sep=sep, encoding="latin-1")
    missing = [c for c in ("text", "label") if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required columns {missing} not found. Available: {df.columns.tolist()}"
        )
    logger.info(f"Loaded {len(df):,} rows from {data_file}")
    return df


# =============================================================================
# Main evaluation function
# =============================================================================


def evaluate(cfg: dict, run: Any, args: argparse.Namespace) -> None:
    model, tokenizer = load_model_for_inference(cfg)
    labels_cfg = load_labels_config(cfg["data"]["labels_config"])

    sep = ","
    if PIPELINE_TEST:
        sep = cfg.get("data", {}).get("sep", ",")

    df = load_eval_data(args.data_file, sep=sep)

    # Pipeline test: evaluate on a small slice only
    if PIPELINE_TEST:
        df = df.head(20)
        logger.info("Pipeline test — evaluating on first 20 rows only.")

    pred_df = run_inference_batch(
        model=model,
        tokenizer=tokenizer,
        df=df,
        labels_cfg=labels_cfg,
        max_new_tokens=250,
    )

    output_dir = args.output_dir or os.path.join(cfg["training"]["output_dir"], "eval")

    log_evaluation_to_wandb(
        pred_df=pred_df,
        run=run,
        dataset_type=args.dataset_type,
        tokenizer=tokenizer,
        output_dir=output_dir,
    )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Post-training evaluation")
    parser.add_argument(
        "--config",
        default="config/model_config_llama.yaml",
        help="Model-specific config (merged on top of model_config_base.yaml)",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["silver", "fresh", "gold"],
        required=True,
        help=(
            "silver — held-out 20%% eval set (teacher-labeled)\n"
            "fresh  — unseen test set (teacher-labeled)\n"
            "gold   — 50-ticket expert-labeled gold standard"
        ),
    )
    parser.add_argument(
        "--data-file",
        required=True,
        help="Path to the evaluation CSV (must have 'text' and 'label' columns)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for prediction CSVs (defaults to training output_dir/eval)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, pipeline_test=PIPELINE_TEST)
    run = init_wandb_eval(
        cfg,
        dataset_type=args.dataset_type,
        pipeline_test=PIPELINE_TEST,
        gpu_tier=GPU_TIER,
    )

    try:
        evaluate(cfg, run=run, args=args)
    finally:
        run.finish()
