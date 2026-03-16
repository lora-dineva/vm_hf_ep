#!/usr/bin/env python3
"""
Convert a text description to an audio file using Kokoro-82M via HF Inference.
Can be run standalone or as part of the pipeline.
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import InferenceClient

from utils import get_token

LOG_PREFIX = "[generate_audio]"

DEFAULT_MODEL = "hexgrad/Kokoro-82M"
DEFAULT_PROVIDER = "fal-ai"


def generate_audio(text: str, token: str, model: str = DEFAULT_MODEL,
                   provider: str = DEFAULT_PROVIDER) -> bytes:
    client = InferenceClient(provider=provider, api_key=token)
    audio = client.text_to_speech(text, model=model)
    return audio


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate audio from a text file using Kokoro-82M (HF Inference)."
    )
    parser.add_argument("input", help="Text file to read (or '-' for stdin)")
    parser.add_argument("-o", "--output", default="output.wav", help="Output audio file path")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    args = parser.parse_args()

    if args.input == "-":
        text = sys.stdin.read().strip()
    else:
        p = Path(args.input)
        if not p.is_file():
            print(f"Error: File not found: {p}", file=sys.stderr)
            sys.exit(1)
        text = p.read_text(encoding="utf-8").strip()

    if not text:
        print("Error: Input text is empty.", file=sys.stderr)
        sys.exit(1)

    token = get_token()
    print(f"{LOG_PREFIX} model={args.model} provider={args.provider}", file=sys.stderr)
    print(f"{LOG_PREFIX} text length: {len(text)} chars", file=sys.stderr)

    audio_bytes = generate_audio(text, token, model=args.model, provider=args.provider)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(audio_bytes)
    print(f"{LOG_PREFIX} saved: {out_path} ({len(audio_bytes)} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
