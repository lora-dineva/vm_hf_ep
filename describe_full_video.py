#!/usr/bin/env python3
"""
Describe a full video in one or two passes (no scene splitting).

Step 1 (vision):  Send frames to Qwen3-VL for a raw description.
Step 2 (styling): Send the raw text to the styling model for a polished narrative.

Run both steps at once (default), or each step individually:

  # Both steps
  python describe_full_video.py video.mxf -o scenes/out_styled.txt

  # Step 1 only -- save raw output for review
  python describe_full_video.py video.mxf --raw-only -o scenes/out_raw.txt

  # Step 2 only -- style an existing raw file
  python describe_full_video.py --style-from scenes/out_raw.txt -o scenes/out_styled.txt
"""

import argparse
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

from describe_video import extract_frames_b64
from utils import (
    get_token, load_config, resolve_text_prompt, resolve_chat_prompt,
)

LOG_PREFIX = "[full_video]"

FALLBACK_VISION_PROMPT = "Describe everything that happens in this video."

FALLBACK_STYLE_SYSTEM = """\
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

FALLBACK_STYLE_USER = (
    "Rewrite the following raw video description into a polished "
    "audio description narrative:\n\n"
)


# ---------------------------------------------------------------------------
# Step 1: Vision model
# ---------------------------------------------------------------------------

def vision_describe(video_path: str, token: str, cfg: dict) -> dict:
    """Extract frames and get a raw description from the vision model."""
    label = cfg.get("prompt_label", "production")
    vision_name = cfg.get("prompt_full_video_description", "full-video-description")
    vision_prompt, _ = resolve_text_prompt(vision_name, label, FALLBACK_VISION_PROMPT)

    print(f"{LOG_PREFIX} Step 1: vision ({cfg['model']})", file=sys.stderr)
    t0 = time.perf_counter()
    frames_b64 = extract_frames_b64(
        video_path, cfg["fps"], cfg["max_frames"], vision_prompt,
        cfg.get("image_patch_size", 16), cfg.get("jpeg_quality", 85),
    )
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        for b64 in frames_b64
    ]
    content.append({"type": "text", "text": vision_prompt})

    client = OpenAI(base_url=cfg["endpoint"].rstrip("/") + "/v1", api_key=token)
    t1 = time.perf_counter()
    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=[{"role": "user", "content": content}],
        max_tokens=cfg["max_tokens"],
    )
    api_time = time.perf_counter() - t1

    if not resp.choices or not resp.choices[0].message.content:
        print("Error: No content from vision model.", file=sys.stderr)
        sys.exit(1)

    raw = resp.choices[0].message.content.strip()
    total = time.perf_counter() - t0
    print(f"{LOG_PREFIX}   {len(frames_b64)} frames, {len(raw.split())} words, {api_time:.1f}s (total {total:.1f}s)", file=sys.stderr)

    return {"raw": raw, "frames": len(frames_b64), "vision_time": total}


# ---------------------------------------------------------------------------
# Step 2: Styling model
# ---------------------------------------------------------------------------

def style_description(
    raw: str, token: str, cfg: dict,
    final_endpoint: str, final_model: str, final_max_tokens: int = 2048,
) -> dict:
    """Take a raw description and produce a polished narrative."""
    label = cfg.get("prompt_label", "production")
    style_name = cfg.get("prompt_full_video_style", "full-video-style")
    style_system, style_user_prefix, _ = resolve_chat_prompt(
        style_name, label, FALLBACK_STYLE_SYSTEM, FALLBACK_STYLE_USER,
    )

    print(f"{LOG_PREFIX} Step 2: styling ({final_model})", file=sys.stderr)
    client = OpenAI(base_url=final_endpoint.rstrip("/") + "/v1", api_key=token)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=final_model,
        messages=[
            {"role": "system", "content": style_system},
            {"role": "user", "content": style_user_prefix + raw},
        ],
        max_tokens=final_max_tokens,
    )
    style_time = time.perf_counter() - t0

    if not resp.choices or not resp.choices[0].message.content:
        print("Error: No content from styling model.", file=sys.stderr)
        sys.exit(1)

    styled = resp.choices[0].message.content.strip()
    print(f"{LOG_PREFIX}   {len(styled.split())} words, {style_time:.1f}s", file=sys.stderr)

    return {"styled": styled, "style_time": style_time}


# ---------------------------------------------------------------------------
# Combined (backwards-compatible)
# ---------------------------------------------------------------------------

def describe_full_video(
    video_path: str,
    token: str,
    cfg: dict,
    final_endpoint: str,
    final_model: str,
    final_max_tokens: int = 2048,
) -> dict:
    """Run both steps. Returns dict with raw, styled, and timing info."""
    t0 = time.perf_counter()
    vis = vision_describe(video_path, token, cfg)
    sty = style_description(
        vis["raw"], token, cfg, final_endpoint, final_model, final_max_tokens,
    )
    return {
        "raw": vis["raw"],
        "styled": sty["styled"],
        "frames": vis["frames"],
        "vision_time": vis["vision_time"],
        "style_time": sty["style_time"],
        "total_time": time.perf_counter() - t0,
    }


def main() -> None:
    cfg = load_config()
    endpoint = os.environ.get("HF_ENDPOINT") or cfg["endpoint"]
    final_endpoint = (os.environ.get("HF_FINAL_ENDPOINT")
                      or cfg.get("final_endpoint")
                      or "https://d3hbhiiq7llzxs1q.us-east-1.aws.endpoints.huggingface.cloud")
    final_model = (os.environ.get("HF_FINAL_MODEL")
                   or cfg.get("final_model", "mistralai/Mistral-Nemo-Instruct-2407"))

    parser = argparse.ArgumentParser(
        description="Describe a full video (vision + styling). Steps can run separately."
    )
    parser.add_argument("video", nargs="?", default=cfg.get("video", "test_clip_60s.mp4"))
    parser.add_argument("--endpoint", default=endpoint)
    parser.add_argument("--model", default=cfg["model"])
    parser.add_argument("--max-tokens", type=int, default=cfg["max_tokens"])
    parser.add_argument("--fps", type=float, default=cfg["fps"])
    parser.add_argument("--max-frames", type=int, default=cfg["max_frames"])
    parser.add_argument("--final-endpoint", default=final_endpoint)
    parser.add_argument("--final-model", default=final_model)
    parser.add_argument("--final-max-tokens", type=int, default=cfg.get("final_max_tokens", 2048))
    parser.add_argument("-o", "--output", metavar="FILE")
    parser.add_argument("--raw-only", action="store_true",
                        help="Run only Step 1 (vision). Save raw output and stop.")
    parser.add_argument("--style-from", metavar="FILE",
                        help="Run only Step 2 (styling) on an existing raw description file.")
    args = parser.parse_args()

    cfg["endpoint"] = args.endpoint
    cfg["model"] = args.model
    cfg["max_tokens"] = args.max_tokens
    cfg["fps"] = args.fps
    cfg["max_frames"] = args.max_frames

    token = get_token()

    # --- Step 2 only: style an existing raw file ---
    if args.style_from:
        raw_path = Path(args.style_from)
        if not raw_path.is_file():
            print(f"Error: Raw file not found: {raw_path}", file=sys.stderr)
            sys.exit(1)
        raw = raw_path.read_text(encoding="utf-8").strip()
        sty = style_description(
            raw, token, cfg,
            args.final_endpoint, args.final_model, args.final_max_tokens,
        )
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(sty["styled"], encoding="utf-8")
            print(f"{LOG_PREFIX} wrote: {out}", file=sys.stderr)
        print(sty["styled"])
        return

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"Error: Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    # --- Step 1 only: raw vision output ---
    if args.raw_only:
        vis = vision_describe(str(video_path), token, cfg)
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(vis["raw"], encoding="utf-8")
            print(f"{LOG_PREFIX} wrote: {out}", file=sys.stderr)
        print(vis["raw"])
        return

    # --- Both steps ---
    result = describe_full_video(
        str(video_path), token, cfg,
        args.final_endpoint, args.final_model, args.final_max_tokens,
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result["styled"], encoding="utf-8")
        print(f"{LOG_PREFIX} wrote: {out}", file=sys.stderr)
        raw_path = out.with_name(out.stem.replace("_styled", "_raw") + out.suffix)
        if raw_path == out:
            raw_path = out.with_stem(out.stem + "_raw")
        raw_path.write_text(result["raw"], encoding="utf-8")
        print(f"{LOG_PREFIX} wrote: {raw_path}", file=sys.stderr)

    print(result["styled"])


if __name__ == "__main__":
    main()
