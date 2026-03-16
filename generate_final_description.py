#!/usr/bin/env python3
"""
Generate a single final video description from per-scene descriptions using a
Hugging Face Inference Endpoint (default: Llama 3.3 70B Instruct). Reads the
scene CSV (with description column) and sends the text to the LLM to produce
one coherent summary.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# OpenAI-compatible client for HF Inference Endpoints
from openai import OpenAI

LOG_PREFIX = "[generate_final_description]"

SYSTEM_PROMPT = """You are an expert Audio Describer creating accessibility tracks for visually impaired audiences. Your task is to take a chronological sequence of raw, AI-generated video shot descriptions and merge them into a single, seamless, and natural narrative scene.

Follow these strict rules:
1. MERGE CONTINUITY: Recognize when consecutive shots are part of the same physical space or conversation (e.g., cutting back and forth between two people in a car). Describe the scene as a whole, rather than isolated fragments.
2. NO CAMERA TERMINOLOGY: Never use words like "shot", "cut", "camera", "angle", "frame", or "zooms". Describe the physical world and the characters, not the filmmaking.
3. OBJECTIVE OBSERVATION: Describe exactly what is visible—actions, settings, clothing, and facial expressions. Do not guess character motivations or internal thoughts.
4. TTS-READY: The output must be immediately ready for a Text-to-Speech engine. Do not include timestamps, labels, or conversational filler (e.g., "Here is the description:"). Output ONLY the spoken narrative.

Example Input:
Shot 1: The camera shows a woman driving a car. It is raining outside.
Shot 2: Cut to the back seat. A young boy is looking out the window, looking bored.
Shot 3: Cut back to the woman driving. She turns the steering wheel sharply.
Shot 4: Cut back to the boy, who drops his toy in surprise.

Example Output:
A woman drives a car through the rain. In the back seat, a young boy stares out the window. Suddenly, the woman turns the steering wheel sharply, startling the boy, who drops his toy."""

DEFAULT_ENDPOINT = "https://xe765ikyd0ykgs1e.us-east-1.aws.endpoints.huggingface.cloud"
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_SCENES_DIR = "scenes"
DEFAULT_MAX_TOKENS = 2048


def get_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("Error: Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN in .env", file=sys.stderr)
        sys.exit(1)
    return token


def load_scenes_csv(csv_path: Path) -> list[dict]:
    """Load scene CSV; each row is a dict (scene_id, start_time, end_time, description, ...)."""
    if not csv_path.is_file():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


USER_PROMPT_INTRO = "Please merge the following chronological sequence of shots into a single continuous audio description:"


def build_prompt(rows: list[dict]) -> str:
    """Build the user message: intro + chronological scene descriptions."""
    parts = [USER_PROMPT_INTRO, ""]
    for row in rows:
        scene_id = row.get("scene_id", "?")
        start = row.get("start_time", "")
        end = row.get("end_time", "")
        desc = (row.get("description") or "").strip()
        parts.append(f"Scene {scene_id} ({start} – {end}): {desc}")
    return "\n\n".join(parts)


def generate_final_description(
    user_prompt: str,
    base_url: str,
    token: str,
    model: str,
    max_tokens: int,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Call HF endpoint and return the model's reply."""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate final video description from scene CSV using Llama 3.3 70B (or other HF endpoint)."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default=None,
        help="Path to scene CSV (default: scenes/<stem>_scenes.csv if --video given, else first *_scenes.csv in --scenes-dir)",
    )
    parser.add_argument(
        "--video",
        metavar="PATH",
        help="Video path or name (e.g. test_clip_60s.mp4) to derive CSV: scenes/<stem>_scenes.csv",
    )
    parser.add_argument(
        "--scenes-dir",
        default=DEFAULT_SCENES_DIR,
        help=f"Scenes directory when resolving CSV from --video (default: {DEFAULT_SCENES_DIR})",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_FINAL_ENDPOINT") or DEFAULT_ENDPOINT,
        help=f"HF Inference Endpoint base URL (default: {DEFAULT_ENDPOINT})",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("HF_FINAL_MODEL") or DEFAULT_MODEL,
        help=f"Model name on endpoint (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens for the reply (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        help="Write final description to this file (default: print only)",
    )
    args = parser.parse_args()

    csv_path = None
    if args.csv:
        csv_path = Path(args.csv)
    elif args.video:
        stem = Path(args.video).stem
        csv_path = Path(args.scenes_dir) / f"{stem}_scenes.csv"
    else:
        scenes_dir = Path(args.scenes_dir)
        if scenes_dir.is_dir():
            candidates = sorted(scenes_dir.glob("*_scenes.csv"))
            if candidates:
                csv_path = candidates[0]
        if not csv_path:
            print(
                "Error: Provide CSV path, or --video to use scenes/<stem>_scenes.csv, or ensure --scenes-dir contains *_scenes.csv",
                file=sys.stderr,
            )
            sys.exit(1)

    if not csv_path.is_file():
        print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_scenes_csv(csv_path)
    if not rows:
        print("Error: CSV is empty or has no data rows.", file=sys.stderr)
        sys.exit(1)

    if "description" not in (rows[0] or {}):
        print("Error: CSV must have a 'description' column (run describe_video.py on scenes first).", file=sys.stderr)
        sys.exit(1)

    prompt = build_prompt(rows)
    token = get_token()

    print(f"{LOG_PREFIX} endpoint={args.endpoint!r} model={args.model!r}", file=sys.stderr)
    print(f"{LOG_PREFIX} scenes={len(rows)} from {csv_path}", file=sys.stderr)

    result = generate_final_description(
        prompt,
        args.endpoint,
        token,
        args.model,
        args.max_tokens,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result, encoding="utf-8")
        print(f"{LOG_PREFIX} wrote {out_path}", file=sys.stderr)

    print(result)


if __name__ == "__main__":
    main()
