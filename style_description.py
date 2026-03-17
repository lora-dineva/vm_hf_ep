#!/usr/bin/env python3
"""
Style a raw video description using an LLM (e.g. Mistral).

Auto-detects input type:
  - CSV file  -> merge per-scene descriptions into one narrative
  - Text file -> rewrite a raw full-video description into polished prose

Usage:
  # Style merged scenes
  python style_description.py scenes/video_scenes.csv -o scenes/styled.txt

  # Style a full-video raw description
  python style_description.py scenes/raw.txt -o scenes/styled.txt

  # Override Langfuse prompt label
  python style_description.py scenes/raw.txt --prompt-label experiment_v2 -o out.txt
"""

import argparse
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

from utils import get_token, load_config, load_scenes_csv, log_run, resolve_chat_prompt

LOG_PREFIX = "[style]"

# ---------------------------------------------------------------------------
# Fallback prompts: scene merge
# ---------------------------------------------------------------------------

SCENE_MERGE_SYSTEM = """\
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

SCENE_MERGE_USER = (
    "Merge the following shots into a single continuous audio description. "
    "Output ONLY the final description, no other text."
)

# ---------------------------------------------------------------------------
# Fallback prompts: full-video style
# ---------------------------------------------------------------------------

FULL_VIDEO_SYSTEM = """\
You are an expert Audio Describer creating accessibility tracks for visually \
impaired audiences. You will receive a raw, AI-generated description of a video. \
Rewrite it into a single, seamless, and natural spoken narrative.

Follow these strict rules:
1. CONTINUITY: Present events as a continuous, chronological flow.
2. NO CAMERA TERMINOLOGY: Never use words like "shot", "cut", "camera", \
"angle", "frame", or "zooms". Describe the physical world and characters.
3. OBJECTIVE OBSERVATION: Describe exactly what is visible -- actions, settings, \
clothing, facial expressions. Do not guess motivations or internal thoughts.
4. TTS-READY: Output must be immediately ready for Text-to-Speech. No timestamps, \
labels, or conversational filler. Output ONLY the spoken narrative."""

FULL_VIDEO_USER = (
    "Rewrite the following raw video description into a polished "
    "audio description narrative:\n\n"
)

DEFAULT_ENDPOINT = "https://d3hbhiiq7llzxs1q.us-east-1.aws.endpoints.huggingface.cloud"


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def _build_scene_prompt(rows: list[dict], user_intro: str) -> str:
    parts = [user_intro, ""]
    for row in rows:
        scene_id = row.get("scene_id", "?")
        start = row.get("start_time", "")
        end = row.get("end_time", "")
        desc = (row.get("description") or "").strip()
        parts.append(f"Scene {scene_id} ({start} \u2013 {end}): {desc}")
    return "\n\n".join(parts)


def _is_csv(path: Path) -> bool:
    return path.suffix.lower() == ".csv"


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def style(
    system_prompt: str,
    user_prompt: str,
    base_url: str,
    token: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    client = OpenAI(base_url=base_url.rstrip("/") + "/v1", api_key=token)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if not resp.choices or not resp.choices[0].message.content:
        print("Error: No content in response.", file=sys.stderr)
        sys.exit(1)
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()
    endpoint = (os.environ.get("HF_FINAL_ENDPOINT")
                or cfg.get("final_endpoint")
                or DEFAULT_ENDPOINT)
    model = (os.environ.get("HF_FINAL_MODEL")
             or cfg.get("final_model", "mistralai/Mistral-Nemo-Instruct-2407"))
    max_tokens = cfg.get("final_max_tokens", 2048)
    temperature = cfg.get("final_temperature", 0.7)

    parser = argparse.ArgumentParser(
        description="Style a raw description (text or scene CSV) with an LLM."
    )
    parser.add_argument("input", help="Text file or scene CSV to style")
    parser.add_argument("--endpoint", default=endpoint)
    parser.add_argument("--model", default=model)
    parser.add_argument("--max-tokens", type=int, default=max_tokens)
    parser.add_argument("--temperature", type=float, default=temperature)
    parser.add_argument("--prompt-label", default=cfg.get("prompt_label", "production"),
                        help="Langfuse prompt label (default: from config)")
    parser.add_argument("--prompt-name",
                        help="Override Langfuse prompt name (auto-detected by default)")
    parser.add_argument("-o", "--output", metavar="FILE")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    token = get_token()

    if _is_csv(input_path):
        rows = load_scenes_csv(input_path)
        if not rows:
            print("Error: CSV is empty.", file=sys.stderr)
            sys.exit(1)
        if "description" not in (rows[0] or {}):
            print("Error: CSV must have a 'description' column. "
                  "Run describe_video.py first.", file=sys.stderr)
            sys.exit(1)

        prompt_name = args.prompt_name or cfg.get("prompt_scene_merge", "scene-merge")
        system_prompt, user_intro, _ = resolve_chat_prompt(
            prompt_name, args.prompt_label, SCENE_MERGE_SYSTEM, SCENE_MERGE_USER,
        )
        user_prompt = _build_scene_prompt(rows, user_intro)
        print(f"{LOG_PREFIX} mode=scene-merge scenes={len(rows)} "
              f"prompt={prompt_name!r} label={args.prompt_label!r}", file=sys.stderr)
    else:
        raw = input_path.read_text(encoding="utf-8").strip()
        if not raw:
            print("Error: Input file is empty.", file=sys.stderr)
            sys.exit(1)

        prompt_name = args.prompt_name or cfg.get("prompt_full_video_style", "full-video-style")
        system_prompt, user_prefix, _ = resolve_chat_prompt(
            prompt_name, args.prompt_label, FULL_VIDEO_SYSTEM, FULL_VIDEO_USER,
        )
        user_prompt = user_prefix + raw
        print(f"{LOG_PREFIX} mode=full-video prompt={prompt_name!r} "
              f"label={args.prompt_label!r}", file=sys.stderr)

    print(f"{LOG_PREFIX} endpoint={args.endpoint!r} model={args.model!r}", file=sys.stderr)

    t_start = time.perf_counter()
    result = style(
        system_prompt, user_prompt,
        args.endpoint, token, args.model, args.max_tokens, args.temperature,
    )
    elapsed = time.perf_counter() - t_start

    outputs = []
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result, encoding="utf-8")
        outputs.append(str(out_path))
        print(f"{LOG_PREFIX} wrote: {out_path}", file=sys.stderr)

    log_run("style_description.py", {
        "input": str(input_path),
        "mode": "scene-merge" if _is_csv(input_path) else "full-video",
        "model": args.model,
        "endpoint": args.endpoint,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "prompt_label": args.prompt_label,
        "prompt_name": prompt_name,
    }, elapsed, outputs)

    print(result)


if __name__ == "__main__":
    main()
