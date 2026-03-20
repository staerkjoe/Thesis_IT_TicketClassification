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


def extract_label(text: str) -> str:
    """
    Take only the first non-empty line of model output.
    The model is trained to output just the label — anything after
    the first line is rambling that we discard.
    """
    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            return line
    return text.strip()


def classify(
    model,
    tokenizer,
    ticket_text: str,
    labels_cfg: dict,
    max_new_tokens: int = 20,  # labels are short — 20 is plenty
    stream: bool = True,
) -> str:
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
        eos_token_id=tokenizer.eos_token_id,  # stops at EOS — prevents rambling
    )

    generated = outputs[0][inputs["input_ids"].shape[1] :]
    raw = tokenizer.decode(generated, skip_special_tokens=True)
    return extract_label(raw)


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument(
        "--ticket",
        default="description: Support - TV/Video on demand/Streamer or Audio Customer experiences that his "
        "programs lag on the streamer. Category:TV/Video on demand/Streamer or Audio contentProvider:Everything on "
        "YouSee Play pathFromCIP:- issue:Video/picture - Other issues related to the video timeCodeError:00:00 "
        "contentTitle:Everything on YouSee Play TV-Video On Demand (VOD) â€“ Validation, "
        "Reasoning: The customer reports lag while streaming content via YouSee Play on a streamer device, which "
        "aligns with OTT TV service and is a performance-related issue (video playback lag). "
        "Reasoning_Confidence: 0.92 Label_Confidence: 0.90, scientific_confidence: 0.8195, max_path_confidence: 0.9, "
        "consistency: 0.8",
    )
    parser.add_argument("--no-stream", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    labels_cfg = load_labels_config(cfg["data"]["labels_config"])
    model, tokenizer = load_model_for_inference(cfg)

    print(f"\nTicket : {args.ticket}")
    print(f"{'─' * 60}")

    predicted = classify(
        model,
        tokenizer,
        args.ticket,
        labels_cfg,
        stream=not args.no_stream,
    )

    print(f"{'─' * 60}")
    print(f"Predicted label: {predicted}\n")
