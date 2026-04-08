#!/usr/bin/env python
# =============================================================================
# scripts/inference.py
#
# Load fine-tuned model from Hub and classify an IT ticket.
#
# Run: poetry run python scripts/inference.py --config config/model_config.yaml
#      poetry run python scripts/inference.py --ticket "My VPN stopped working."
#      poetry run python scripts/inference.py --ticket "My VPN stopped working." --no-stream
# =============================================================================

import argparse
import logging
import pathlib
import re
import sys
import warnings

import unsloth  # noqa: F401 — must be first
import yaml
from dotenv import load_dotenv
from transformers import TextStreamer

# Suppress transformers v5.2 internal logging format bug
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from utils.training.llm_handler import (  # noqa: E402
    build_system_prompt,
    load_labels_config,
    load_model_for_inference,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
# FIX: Suppress the buggy transformers AttentionMask deprecation warning
logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_inference_prompt(ticket_text: str, tokenizer, labels_cfg: dict) -> str:
    """Build prompt without the assistant turn — model generates that part.

    Mistral/Ministral models do not support a system role; the system content
    is merged into the user turn instead (mirrors the training-time format_prompt
    behaviour in llm_handler.py).
    """
    system_content = build_system_prompt(labels_cfg)
    user_content = f"Input Description: {ticket_text}\nOutput Tag:"

    model_type = getattr(tokenizer, "name_or_path", "").lower()
    if "mistral" in model_type or "ministral" in model_type:
        messages = [{"role": "user", "content": f"{system_content}\n\n{user_content}"}]
    else:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_tag(full_output: str) -> str:
    """
    Pull just the 'Tag: ...' line from the full reasoning output.
    Falls back to the first non-empty line if no Tag: prefix is found.
    """
    text = full_output.strip()
    match = re.search(r"Tag:\s*(.*)", text, re.IGNORECASE)
    if match:
        return match.group(1).splitlines()[0].strip()
    # Fallback: return first non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text


def classify(
    model,
    tokenizer,
    ticket_text: str,
    labels_cfg: dict,
    max_new_tokens: int = 250,  # matches training — reasoning chain needs room
    stream: bool = True,
) -> dict:
    """
    Returns a dict with:
      - 'full_output': complete model response (Reasoning + Tag + Confidence)
      - 'tag':         just the extracted Tag line, for downstream use
    """
    prompt = build_inference_prompt(ticket_text, tokenizer, labels_cfg)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = (
        TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        if stream
        else None
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated = outputs[0][inputs["input_ids"].shape[1] :]
    full_output = tokenizer.decode(generated, skip_special_tokens=True).strip()
    tag = extract_tag(full_output)

    return {"full_output": full_output, "tag": tag}


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument(
        "--ticket",
        default="Support - My YouSee/Web (PC/Mac)/Missing content (e.g. Bills) "
        "missing pay now button, and Customer is logged in with my ID. "
        "Please look into why the 'Pay now' button is not there, should be "
        "possible for the kd to pay via Mit Yousee "
        "Customer is using iPhone and Samsung tablet and he has tried both "
        "the app and the web, Customer uses the right username",
    )
    parser.add_argument("--no-stream", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    labels_cfg = load_labels_config(cfg["data"]["labels_config"])
    model, tokenizer = load_model_for_inference(cfg)

    print(f"\nTicket : {args.ticket}")
    print(f"{'─' * 60}")

    result = classify(
        model,
        tokenizer,
        args.ticket,
        labels_cfg,
        stream=not args.no_stream,
    )

    print(f"{'─' * 60}")
    print(f"Full output:\n{result['full_output']}")
    print(f"{'─' * 60}")
    print(f"Extracted tag: {result['tag']}\n")
