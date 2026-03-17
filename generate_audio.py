#!/usr/bin/env python3
"""
Convert text descriptions to audio using Kokoro-82M via HF Inference.

Accepts either a text file (single audio) or a scene CSV (one audio per scene).

Usage:
  # Single text file -> single .wav
  python generate_audio.py scenes/styled.txt -o scenes/audio.wav

  # Scene CSV -> one .wav per scene description
  python generate_audio.py scenes/video_scenes.csv --output-dir scenes/audio/
"""

import argparse
import sys
import time
from pathlib import Path

from huggingface_hub import InferenceClient

from utils import get_token, load_scenes_csv, log_run

LOG_PREFIX = "[generate_audio]"

DEFAULT_MODEL = "hexgrad/Kokoro-82M"
DEFAULT_PROVIDER = "fal-ai"


def generate_audio(text: str, token: str, model: str = DEFAULT_MODEL,
                   provider: str = DEFAULT_PROVIDER) -> bytes:
    client = InferenceClient(provider=provider, api_key=token)
    return client.text_to_speech(text, model=model)


def _is_csv(path: Path) -> bool:
    return path.suffix.lower() == ".csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate audio from a text file or scene CSV."
    )
    parser.add_argument("input", help="Text file, or scene CSV for per-scene audio")
    parser.add_argument("-o", "--output", default="output.wav",
                        help="Output .wav path (used for single-file mode)")
    parser.add_argument("--output-dir",
                        help="Output directory for per-scene audio (CSV mode)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    token = get_token()
    print(f"{LOG_PREFIX} model={args.model} provider={args.provider}", file=sys.stderr)

    t_start = time.perf_counter()
    outputs: list[str] = []

    if _is_csv(input_path):
        rows = load_scenes_csv(input_path)
        if not rows:
            print("Error: CSV is empty.", file=sys.stderr)
            sys.exit(1)
        if "description" not in (rows[0] or {}):
            print("Error: CSV must have a 'description' column.", file=sys.stderr)
            sys.exit(1)

        out_dir = Path(args.output_dir) if args.output_dir else input_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = input_path.stem.replace("_scenes", "")

        for row in rows:
            sid = row.get("scene_id", "unknown")
            desc = (row.get("description") or "").strip()
            if not desc:
                print(f"{LOG_PREFIX} scene {sid}: empty description, skipping",
                      file=sys.stderr)
                continue

            out_path = out_dir / f"{stem}_scene_{sid}.wav"
            print(f"{LOG_PREFIX} scene {sid}: {len(desc)} chars -> {out_path.name}",
                  file=sys.stderr)
            audio_bytes = generate_audio(desc, token, model=args.model,
                                         provider=args.provider)
            out_path.write_bytes(audio_bytes)
            outputs.append(str(out_path))

        print(f"{LOG_PREFIX} wrote per-scene audio to {out_dir}", file=sys.stderr)
    else:
        text = input_path.read_text(encoding="utf-8").strip()
        if not text:
            print("Error: Input text is empty.", file=sys.stderr)
            sys.exit(1)

        print(f"{LOG_PREFIX} text length: {len(text)} chars", file=sys.stderr)
        audio_bytes = generate_audio(text, token, model=args.model,
                                     provider=args.provider)

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(audio_bytes)
        outputs.append(str(out_path))
        print(f"{LOG_PREFIX} saved: {out_path} ({len(audio_bytes)} bytes)",
              file=sys.stderr)

    elapsed = time.perf_counter() - t_start
    log_run("generate_audio.py", {
        "input": str(input_path),
        "model": args.model,
        "provider": args.provider,
    }, elapsed, outputs)


if __name__ == "__main__":
    main()
