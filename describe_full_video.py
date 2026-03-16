#!/usr/bin/env python3
"""
Describe a full video in one pass (no scene splitting).

Step 1: Send all frames to Qwen3-VL for a raw description.
Step 2: Send the raw text to the styling model for a polished narrative.

Can be used as an importable module or run standalone.
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


def describe_full_video(
    video_path: str,
    token: str,
    cfg: dict,
    final_endpoint: str,
    final_model: str,
    final_max_tokens: int = 2048,
) -> dict:
    """Run both steps. Returns dict with raw, styled, and timing info."""
    label = cfg.get("prompt_label", "production")
    t0 = time.perf_counter()

    # Resolve vision prompt from Langfuse
    vision_name = cfg.get("prompt_full_video_description", "full-video-description")
    vision_prompt, _ = resolve_text_prompt(vision_name, label, FALLBACK_VISION_PROMPT)

    # Step 1: Vision model
    print(f"{LOG_PREFIX} Step 1: vision ({cfg['model']})", file=sys.stderr)
    frames_b64 = extract_frames_b64(
        video_path, cfg["fps"], cfg["max_frames"], vision_prompt,
        cfg.get("image_patch_size", 16), cfg.get("jpeg_quality", 85),
    )
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        for b64 in frames_b64
    ]
    content.append({"type": "text", "text": vision_prompt})

    vision_client = OpenAI(
        base_url=cfg["endpoint"].rstrip("/") + "/v1", api_key=token,
    )
    t1 = time.perf_counter()
    resp = vision_client.chat.completions.create(
        model=cfg["model"],
        messages=[{"role": "user", "content": content}],
        max_tokens=cfg["max_tokens"],
    )
    vision_time = time.perf_counter() - t1

    if not resp.choices or not resp.choices[0].message.content:
        print("Error: No content from vision model.", file=sys.stderr)
        sys.exit(1)

    raw = resp.choices[0].message.content.strip()
    print(f"{LOG_PREFIX}   {len(frames_b64)} frames, {len(raw.split())} words, {vision_time:.1f}s", file=sys.stderr)

    # Resolve style prompt from Langfuse
    style_name = cfg.get("prompt_full_video_style", "full-video-style")
    style_system, style_user_prefix, _ = resolve_chat_prompt(
        style_name, label, FALLBACK_STYLE_SYSTEM, FALLBACK_STYLE_USER,
    )

    # Step 2: Styling model
    print(f"{LOG_PREFIX} Step 2: styling ({final_model})", file=sys.stderr)
    style_client = OpenAI(
        base_url=final_endpoint.rstrip("/") + "/v1", api_key=token,
    )
    t2 = time.perf_counter()
    resp2 = style_client.chat.completions.create(
        model=final_model,
        messages=[
            {"role": "system", "content": style_system},
            {"role": "user", "content": style_user_prefix + raw},
        ],
        max_tokens=final_max_tokens,
    )
    style_time = time.perf_counter() - t2
    total = time.perf_counter() - t0

    if not resp2.choices or not resp2.choices[0].message.content:
        print("Error: No content from styling model.", file=sys.stderr)
        sys.exit(1)

    styled = resp2.choices[0].message.content.strip()
    print(f"{LOG_PREFIX}   {len(styled.split())} words, {style_time:.1f}s (total {total:.1f}s)", file=sys.stderr)

    return {
        "raw": raw,
        "styled": styled,
        "frames": len(frames_b64),
        "vision_time": vision_time,
        "style_time": style_time,
        "total_time": total,
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
        description="Describe a full video in one pass (vision + styling)."
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
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"Error: Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    cfg["endpoint"] = args.endpoint
    cfg["model"] = args.model
    cfg["max_tokens"] = args.max_tokens
    cfg["fps"] = args.fps
    cfg["max_frames"] = args.max_frames

    token = get_token()
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
