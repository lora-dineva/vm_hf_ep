#!/usr/bin/env python3
"""
Describe a full video in a single pass (no scene splitting).

Sends sampled frames to a vision-language model (Qwen3-VL) and outputs
the raw description text.

Usage:
  python describe_full_video.py video.mxf -o scenes/raw.txt
"""

import argparse
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

from describe_video import extract_frames_b64
from utils import get_token, load_config, log_run, resolve_text_prompt

LOG_PREFIX = "[full_video]"

FALLBACK_VISION_PROMPT = "Describe everything that happens in this video."


def vision_describe(video_path: str, token: str, cfg: dict) -> dict:
    """Extract frames and get a raw description from the vision model."""
    label = cfg.get("prompt_label", "production")
    vision_name = cfg.get("prompt_full_video_description", "full-video-description")
    vision_prompt, _ = resolve_text_prompt(vision_name, label, FALLBACK_VISION_PROMPT)

    print(f"{LOG_PREFIX} vision ({cfg['model']})", file=sys.stderr)
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
    print(f"{LOG_PREFIX}   {len(frames_b64)} frames, {len(raw.split())} words, "
          f"{api_time:.1f}s (total {total:.1f}s)", file=sys.stderr)

    return {"raw": raw, "frames": len(frames_b64), "vision_time": total}


def main() -> None:
    cfg = load_config()
    endpoint = os.environ.get("HF_ENDPOINT") or cfg["endpoint"]

    parser = argparse.ArgumentParser(
        description="Describe a full video via a vision-language model."
    )
    parser.add_argument("video", nargs="?", default=cfg.get("video", "test_clip_60s.mp4"))
    parser.add_argument("--endpoint", default=endpoint)
    parser.add_argument("--model", default=cfg["model"])
    parser.add_argument("--max-tokens", type=int, default=cfg["max_tokens"])
    parser.add_argument("--fps", type=float, default=cfg["fps"])
    parser.add_argument("--max-frames", type=int, default=cfg["max_frames"])
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
    t_start = time.perf_counter()
    result = vision_describe(str(video_path), token, cfg)
    elapsed = time.perf_counter() - t_start

    outputs = []
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result["raw"], encoding="utf-8")
        outputs.append(str(out))
        print(f"{LOG_PREFIX} wrote: {out}", file=sys.stderr)

    log_run("describe_full_video.py", {
        "video": str(video_path),
        "model": cfg["model"],
        "endpoint": cfg["endpoint"],
        "fps": cfg["fps"],
        "max_frames": cfg["max_frames"],
        "max_tokens": cfg["max_tokens"],
        "frames_extracted": result["frames"],
    }, elapsed, outputs)

    print(result["raw"])


if __name__ == "__main__":
    main()
