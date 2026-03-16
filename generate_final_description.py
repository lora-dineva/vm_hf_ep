#!/usr/bin/env python3
"""
Merge per-scene descriptions into one coherent narrative using a Hugging Face
Inference Endpoint (default: Mistral Nemo). Reads the scene CSV produced by
describe_video.py and returns a single audio-description-ready text.
"""

import argparse
import os
import sys
from pathlib import Path

from openai import OpenAI

from utils import get_token, load_config, load_scenes_csv

LOG_PREFIX = "[generate_final_description]"

SYSTEM_PROMPT = """\
You are an expert Audio Describer creating accessibility tracks for visually \
impaired audiences. Your task is to take a chronological sequence of raw, \
AI-generated video shot descriptions and merge them into a single, seamless, \
and natural narrative scene.

Follow these strict rules:
1. MERGE CONTINUITY: Recognize when consecutive shots are part of the same \
physical space or conversation. Describe the scene as a whole.
2. NO CAMERA TERMINOLOGY: Never use words like "shot", "cut", "camera", \
"angle", "frame", or "zooms". Describe the physical world and characters.
3. OBJECTIVE OBSERVATION: Describe exactly what is visible -- actions, settings, \
clothing, facial expressions. Do not guess motivations or internal thoughts.
4. TTS-READY: Output must be immediately ready for Text-to-Speech. No timestamps, \
labels, or conversational filler. Output ONLY the spoken narrative."""

DEFAULT_ENDPOINT = "https://d3hbhiiq7llzxs1q.us-east-1.aws.endpoints.huggingface.cloud"

USER_PROMPT_INTRO = (
    "Please merge the following chronological sequence of shots "
    "into a single continuous audio description:"
)


def build_prompt(rows: list[dict]) -> str:
    parts = [USER_PROMPT_INTRO, ""]
    for row in rows:
        scene_id = row.get("scene_id", "?")
        start = row.get("start_time", "")
        end = row.get("end_time", "")
        desc = (row.get("description") or "").strip()
        parts.append(f"Scene {scene_id} ({start} – {end}): {desc}")
    return "\n\n".join(parts)


def generate_final_description(
    user_prompt: str, base_url: str, token: str,
    model: str, max_tokens: int,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    client = OpenAI(base_url=base_url.rstrip("/") + "/v1", api_key=token)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
    )
    if not resp.choices or not resp.choices[0].message.content:
        print("Error: No content in response.", file=sys.stderr)
        sys.exit(1)
    return resp.choices[0].message.content.strip()


def _resolve_csv(args) -> Path:
    """Determine the CSV path from CLI arguments."""
    if args.csv:
        return Path(args.csv)
    if args.video:
        stem = Path(args.video).stem
        return Path(args.scenes_dir) / f"{stem}_scenes.csv"
    scenes_dir = Path(args.scenes_dir)
    if scenes_dir.is_dir():
        candidates = sorted(scenes_dir.glob("*_scenes.csv"))
        if candidates:
            return candidates[0]
    print(
        "Error: Provide CSV path, --video, or ensure --scenes-dir has a *_scenes.csv",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    cfg = load_config()
    endpoint = os.environ.get("HF_FINAL_ENDPOINT") or cfg.get("final_endpoint") or DEFAULT_ENDPOINT
    model = os.environ.get("HF_FINAL_MODEL") or cfg.get("final_model", "mistralai/Mistral-Nemo-Instruct-2407")
    max_tokens = cfg.get("final_max_tokens", 2048)

    parser = argparse.ArgumentParser(
        description="Generate a merged video description from scene CSV."
    )
    parser.add_argument("csv", nargs="?", default=None)
    parser.add_argument("--video", metavar="PATH")
    parser.add_argument("--scenes-dir", default=cfg.get("scenes_dir", "scenes"))
    parser.add_argument("--endpoint", default=endpoint)
    parser.add_argument("--model", default=model)
    parser.add_argument("--max-tokens", type=int, default=max_tokens)
    parser.add_argument("--output", "-o", metavar="FILE")
    args = parser.parse_args()

    csv_path = _resolve_csv(args)
    if not csv_path.is_file():
        print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_scenes_csv(csv_path)
    if not rows:
        print("Error: CSV is empty.", file=sys.stderr)
        sys.exit(1)
    if "description" not in (rows[0] or {}):
        print("Error: CSV must have a 'description' column. Run describe_video.py first.", file=sys.stderr)
        sys.exit(1)

    prompt = build_prompt(rows)
    token = get_token()

    print(f"{LOG_PREFIX} endpoint={args.endpoint!r} model={args.model!r}", file=sys.stderr)
    print(f"{LOG_PREFIX} scenes={len(rows)} from {csv_path}", file=sys.stderr)

    result = generate_final_description(
        prompt, args.endpoint, token, args.model, args.max_tokens,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result, encoding="utf-8")
        print(f"{LOG_PREFIX} wrote {out_path}", file=sys.stderr)

    print(result)


if __name__ == "__main__":
    main()
