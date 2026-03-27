import logging
import os
from typing import Tuple

import yaml
from unsloth import FastLanguageModel

logger = logging.getLogger(__name__)


def load_labels_config(path: str) -> dict:
    """Load labels/prompt config from YAML (e.g. config/prompt_template_student.yaml)."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_system_prompt(labels_cfg: dict) -> str:
    """
    Builds the pure zero-shot system prompt for distillation.
    Only fills taxonomy and removes the few-shot logic entirely.
    """
    template = labels_cfg["prompt_template"]

    # Format hierarchical_categories
    cat_str = "\n".join(
        f"({row[0]} {row[1]})" for row in labels_cfg["hierarchical_categories"]
    )

    # Format scenarios
    scen_str = "\n".join(labels_cfg["scenarios"])

    # Fill template: No examples_text anymore.
    # We keep {description} as a sentinel so the .format() works and
    # the rsplit logic below knows where to cut.
    filled = template.format(
        hierarchical_categories=cat_str,
        scenarios=scen_str,
        description="{description}",
    )

    # Move 'Input Description:' part to the User turn
    system_part = filled.rsplit("Input Description:", 1)[0].rstrip()
    return system_part


def format_prompt(text: str, label: str, tokenizer, labels_cfg: dict) -> str:
    system_content = build_system_prompt(labels_cfg)
    user_content = f"Input Description: {text}\nOutput Tag:"

    # Mistral chat template does not support system role — merge into user turn
    model_type = getattr(tokenizer, "name_or_path", "").lower()
    if "mistral" in model_type or "ministral" in model_type:
        messages = [
            {"role": "user", "content": f"{system_content}\n\n{user_content}"},
            {"role": "assistant", "content": label},
        ]
    else:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": label},
        ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def load_model_for_training(cfg: dict) -> Tuple:
    """Load quantized base model and attach QLoRA adapters."""
    m, lora_cfg = cfg["model"], cfg["lora"]

    logger.info(f"Loading Student Base Model: {m['base_model_id']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=m["base_model_id"],
        max_seq_length=m["max_seq_length"],
        load_in_4bit=m["load_in_4bit"],
        token=os.environ.get("HF_TOKEN"),
    )

    # Replace the hardcoded line with this:
    eos_token = m.get("eos_token")
    if not eos_token:
        raise ValueError(
            f"'eos_token' missing from model config for {m['base_model_id']}. "
            "Add it to your model-specific yaml."
        )
    tokenizer.eos_token = eos_token
    tokenizer.pad_token = tokenizer.eos_token

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
    m = cfg["model"]
    logger.info(f"Loading model from Hub: {m['hub_model_id']}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=m["hub_model_id"],
        max_seq_length=m["max_seq_length"],
        load_in_4bit=m["load_in_4bit"],
        token=os.environ.get("HF_TOKEN"),
    )

    # ADD THIS — same as load_model_for_training
    eos_token = m.get("eos_token")
    if not eos_token:
        raise ValueError(
            f"'eos_token' missing from model config for {m['hub_model_id']}. "
            "Add it to your model-specific yaml."
        )
    tokenizer.eos_token = eos_token
    tokenizer.pad_token = tokenizer.eos_token

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
