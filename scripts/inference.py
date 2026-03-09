#!/usr/bin/env python
# =============================================================================
# scripts/inference.py
#
# Load fine-tuned model from Hub and classify an IT ticket.
#
# Run: poetry run python scripts/inference.py --config config/model_config.yaml
#      poetry run python scripts/inference.py --ticket "My VPN stopped working."
# =============================================================================

import argparse
import logging
import pathlib
import sys

import yaml
from dotenv import load_dotenv
from transformers import TextStreamer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from utils.models.llm_handler import (  # noqa: E402
    build_system_prompt,
    load_labels_config,
    load_model_for_inference,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_inference_prompt(ticket_text: str, tokenizer, labels_cfg: dict) -> str:
    """Build prompt without the assistant turn — model generates that part."""
    messages = [
        {"role": "system", "content": build_system_prompt(labels_cfg)},
        {"role": "user", "content": f"Input Description: {ticket_text}\nOutput Tag:"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def classify(
    model,
    tokenizer,
    ticket_text: str,
    labels_cfg: dict,
    max_new_tokens: int = 64,
    stream: bool = True,
) -> str:
    prompt = build_inference_prompt(ticket_text, tokenizer, labels_cfg)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True) if stream else None

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument(
        "--ticket",
        default="My laptop won't connect to the VPN after the latest update.",
    )
    parser.add_argument("--no-stream", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    labels_cfg = load_labels_config(cfg["data"]["labels_config"])
    model, tokenizer = load_model_for_inference(cfg)

    result = classify(
        model, tokenizer, args.ticket, labels_cfg, stream=not args.no_stream
    )
    if args.no_stream:
        print(result)
