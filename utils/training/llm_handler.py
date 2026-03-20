# =============================================================================
# utils/models/llm_handler.py
# =============================================================================
import logging
import os
from typing import Tuple

import yaml
from unsloth import FastLanguageModel

logger = logging.getLogger(__name__)


def load_labels_config(path: str) -> dict:
    """Load labels/prompt config from a YAML file (e.g. config/config_labels.yaml)."""
    with open(path) as f:
        return yaml.safe_load(f)


def build_system_prompt(labels_cfg: dict) -> str:
    """
    Build the system prompt from config_labels.yaml.
    Fills the prompt_template with hierarchical_categories, scenarios, and
    few_shot_examples, then strips the trailing 'Input Description:' line so
    that part lives in the user turn instead.
    """
    template = labels_cfg["prompt_template"]

    cat_str = "\n".join(
        f"({row[0]} {row[1]})" for row in labels_cfg["hierarchical_categories"]
    )
    scen_str = "\n".join(labels_cfg["scenarios"])
    examples = labels_cfg["few_shot_examples"]

    # Fill everything except the description placeholder
    filled = template.format(
        hierarchical_categories=cat_str,
        scenarios=scen_str,
        examples_text=examples,
        description="{description}",  # keep as sentinel so .format() doesn't fail
    )

    # The "Input Description: ..." line belongs in the user turn, not the system prompt
    system_part = filled.rsplit("Input Description:", 1)[0].rstrip()
    return system_part


def format_prompt(text: str, label: str, tokenizer, labels_cfg: dict) -> str:
    """Format a single (text, label) pair into a full chat-template string."""
    messages = [
        {"role": "system", "content": build_system_prompt(labels_cfg)},
        {"role": "user", "content": f"Input Description: {text}\nOutput Tag:"},
        {"role": "assistant", "content": label},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def load_model_for_training(cfg: dict) -> Tuple:
    """Load quantized base model and attach QLoRA adapters."""
    m, lora_cfg = cfg["model"], cfg["lora"]

    logger.info(f"Loading base model: {m['base_model_id']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=m["base_model_id"],
        max_seq_length=m["max_seq_length"],
        load_in_4bit=m["load_in_4bit"],
        token=os.environ.get("HF_TOKEN"),
    )
    # ADD THIS — tell the tokenizer what its actual EOS token is
    tokenizer.eos_token = "<|eot_id|>"
    tokenizer.pad_token = tokenizer.eos_token  # also good practice

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        target_modules=lora_cfg["target_modules"],
        use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
        random_state=cfg["training"]["seed"],
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )

    return model, tokenizer


def load_model_for_inference(cfg: dict) -> Tuple:
    """Load fine-tuned model from Hub for fast inference."""
    m = cfg["model"]
    logger.info(f"Loading model from Hub: {m['hub_model_id']}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=m["hub_model_id"],
        max_seq_length=m["max_seq_length"],
        load_in_4bit=m["load_in_4bit"],
        token=os.environ.get("HF_TOKEN"),
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def push_model_to_hub(model, tokenizer, cfg: dict) -> None:
    """Push LoRA adapter and tokenizer to HuggingFace Hub."""
    hub_id = cfg["model"]["hub_model_id"]
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN is not set.")
    logger.info(f"Pushing to Hub: {hub_id}")
    model.push_to_hub(hub_id, token=token)
    tokenizer.push_to_hub(hub_id, token=token)
    logger.info("Upload complete.")
