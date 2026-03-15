#!/usr/bin/env python3
"""
Send a video file to a Hugging Face Inference Endpoint (Qwen3-VL) and print
the model's description. Video processing uses qwen-vl-utils (fps/resize sampling).
Defaults and overrides: config.yaml (and CLI overrides config).
"""

import argparse
import base64
import io
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

LOG_PREFIX = "[describe_video]"

from dotenv import load_dotenv
load_dotenv()

from langfuse import get_client
from langfuse.openai import OpenAI as LangfuseOpenAI
from qwen_vl_utils import process_vision_info

# -----------------------------------------------------------------------------
# Config (config.yaml + env overrides)
# -----------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
CHAT_PATH = "/v1/chat/completions"

_DEFAULTS = {
    "video": "test_clip_60s.mp4",
    "frames_dir": "frames",
    "endpoint": "https://rjk11aiy6oefykan.us-east-1.aws.endpoints.huggingface.cloud",
    "model": "Qwen/Qwen3-VL-8B-Instruct",
    "max_tokens": 1024,
    "fps": 2.0,
    "max_frames": 120,
    "image_patch_size": 16,
    "jpeg_quality": 85,
    "prompt_name": "video-description",
    "prompt_label": "production",
    "fallback_prompt": "Describe the video.",
}


def load_config(path: Optional[Path] = None) -> dict:
    """Load config from YAML; env and CLI can override later."""
    p = path or CONFIG_PATH
    out = _DEFAULTS.copy()
    if not p.is_file():
        return out
    try:
        import yaml
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if v is not None and k in out:
                out[k] = v
    except Exception as e:
        print(f"{LOG_PREFIX} Could not load config {p}: {e}", file=sys.stderr)
    return out


def get_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("Error: Set HF_TOKEN in .env or HUGGING_FACE_HUB_TOKEN.", file=sys.stderr)
        sys.exit(1)
    return token


def _langfuse_configured() -> bool:
    return bool(
        os.environ.get("LANGFUSE_SECRET_KEY") and os.environ.get("LANGFUSE_PUBLIC_KEY")
    )


def resolve_prompt(
    cfg: dict,
    ab_labels: Optional[List[str]] = None,
) -> tuple[str, Any]:
    """(prompt_text, langfuse_prompt_or_none). Uses fallback on error or when Langfuse not configured."""
    fallback = (cfg["fallback_prompt"] or _DEFAULTS["fallback_prompt"]).strip()
    if not _langfuse_configured():
        return (fallback, None)
    try:
        langfuse = get_client()
        name = cfg["prompt_name"]
        if ab_labels and len(ab_labels) >= 2:
            prompts = [langfuse.get_prompt(name, label=lbl.strip()) for lbl in ab_labels]
            selected = random.choice(prompts)
        else:
            selected = langfuse.get_prompt(name, label=cfg["prompt_label"] or "production")
        raw = selected.compile() if hasattr(selected, "compile") else str(selected.prompt)
        if isinstance(raw, list):
            parts = [m.get("content", "") for m in raw if isinstance(m, dict) and m.get("role") == "user"]
            raw = parts[-1] if parts else fallback
        return (str(raw).strip() or fallback, selected)
    except Exception as e:
        print(f"{LOG_PREFIX} Langfuse prompt fetch failed, using fallback: {e}", file=sys.stderr)
        return (fallback, None)


def _video_to_frames(
    path: str,
    fps: float,
    max_frames: int,
    prompt_text: str,
    image_patch_size: int,
) -> List[Any]:
    """Load video with qwen-vl-utils; return list of (C,H,W) tensors (uint8-ready, max max_frames)."""
    if not os.path.isfile(path):
        print(f"Error: Video file not found: {path}", file=sys.stderr)
        sys.exit(1)
    file_url = "file://" + os.path.abspath(path).replace("\\", "/")
    messages = [[{"role": "user", "content": [{"type": "video", "video": file_url, "fps": fps}, {"type": "text", "text": prompt_text}]}]]
    try:
        _, video_inputs, _ = process_vision_info(
            messages, return_video_kwargs=True, return_video_metadata=True, image_patch_size=image_patch_size
        )
    except Exception as e:
        print(f"Error: qwen-vl-utils failed: {e}", file=sys.stderr)
        sys.exit(1)
    if not video_inputs:
        print("Error: No video output from process_vision_info.", file=sys.stderr)
        sys.exit(1)
    item = video_inputs[0]
    tensor = item[0] if isinstance(item, tuple) else item
    T = tensor.shape[0]
    return [tensor[i] for i in range(min(T, max_frames))]


def extract_frames_qwen_vl_utils(
    path: str,
    fps: float,
    max_frames: int,
    prompt_text: str,
    image_patch_size: int = 16,
    jpeg_quality: int = 85,
) -> List[str]:
    """Return base64 JPEG strings for each sampled frame."""
    try:
        from PIL import Image as PILImage
    except ImportError as e:
        print(f"Error: Pillow required: {e}", file=sys.stderr)
        sys.exit(1)
    t0 = time.perf_counter()
    frames = _video_to_frames(path, fps, max_frames, prompt_text, image_patch_size)
    print(f"{LOG_PREFIX} process_vision_info: {time.perf_counter() - t0:.2f}s", file=sys.stderr)
    out = []
    for frame in frames:
        arr = frame.permute(1, 2, 0).cpu().numpy().clip(0, 255).astype("uint8")
        buf = io.BytesIO()
        PILImage.fromarray(arr).save(buf, format="JPEG", quality=jpeg_quality)
        out.append(base64.standard_b64encode(buf.getvalue()).decode("ascii"))
    print(f"{LOG_PREFIX} encode {len(out)} frames: {time.perf_counter() - t0:.2f}s", file=sys.stderr)
    return out


def extract_frames_to_dir(
    path: str,
    output_dir: str,
    fps: float,
    max_frames: int,
    prompt_text: str,
    image_patch_size: int = 16,
    jpeg_quality: int = 85,
    open_folder: bool = True,
) -> List[str]:
    """Save sampled frames as JPEGs; return list of paths."""
    try:
        from PIL import Image as PILImage
    except ImportError as e:
        print(f"Error: Pillow required: {e}", file=sys.stderr)
        sys.exit(1)
    t0 = time.perf_counter()
    frames = _video_to_frames(path, fps, max_frames, prompt_text, image_patch_size)
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, frame in enumerate(frames):
        arr = frame.permute(1, 2, 0).cpu().numpy().clip(0, 255).astype("uint8")
        p = os.path.join(output_dir, f"frame_{i:03d}.jpg")
        PILImage.fromarray(arr).save(p, format="JPEG", quality=jpeg_quality)
        paths.append(p)
    print(f"{LOG_PREFIX} saved {len(paths)} frames to {os.path.abspath(output_dir)} in {time.perf_counter() - t0:.2f}s", file=sys.stderr)
    if open_folder:
        abs_out = os.path.abspath(output_dir)
        if sys.platform == "win32":
            os.startfile(abs_out)
        elif sys.platform == "darwin":
            subprocess.run(["open", abs_out], check=False)
        else:
            subprocess.run(["xdg-open", abs_out], check=False)
    return paths


def describe_video(
    video_path: str,
    base_url: str,
    token: str,
    cfg: dict,
    prompt_text: str,
    langfuse_prompt: Any = None,
    trace_metadata: Optional[dict] = None,
) -> str:
    """Call HF endpoint with sampled frames; return description text."""
    t0 = time.perf_counter()
    frames_b64 = extract_frames_qwen_vl_utils(
        video_path,
        cfg["fps"],
        cfg["max_frames"],
        prompt_text,
        cfg.get("image_patch_size", 16),
        cfg.get("jpeg_quality", 85),
    )
    content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}} for b64 in frames_b64]
    content.append({"type": "text", "text": prompt_text})
    print(f"{LOG_PREFIX} video processing: {len(content) - 1} image(s)", file=sys.stderr)

    client = LangfuseOpenAI(base_url=base_url.rstrip("/") + "/v1", api_key=token)
    kwargs = {"model": cfg["model"], "messages": [{"role": "user", "content": content}], "max_tokens": cfg["max_tokens"]}
    if langfuse_prompt is not None:
        kwargs["langfuse_prompt"] = langfuse_prompt
    if trace_metadata:
        kwargs["metadata"] = trace_metadata

    t1 = time.perf_counter()
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        print(f"Error: Request failed: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"{LOG_PREFIX} API: {time.perf_counter() - t1:.2f}s, total: {time.perf_counter() - t0:.2f}s", file=sys.stderr)

    if not resp.choices or not resp.choices[0].message.content:
        print("Error: No content in response.", file=sys.stderr)
        sys.exit(1)
    return resp.choices[0].message.content.strip()


def main() -> None:
    cfg = load_config()
    cfg["endpoint"] = os.environ.get("HF_ENDPOINT") or cfg["endpoint"]

    parser = argparse.ArgumentParser(description="Describe a video via Hugging Face VL endpoint.")
    parser.add_argument("video", nargs="?", default=cfg["video"], help="Video file path")
    parser.add_argument("--endpoint", default=cfg["endpoint"], help="Endpoint base URL")
    parser.add_argument("--model", default=cfg["model"], help="Model name")
    parser.add_argument("--max-tokens", type=int, default=cfg["max_tokens"], help="Max tokens")
    parser.add_argument("--fps", type=float, default=cfg["fps"], help="Frame sampling FPS")
    parser.add_argument("--max-frames", type=int, default=cfg["max_frames"], help="Max frames")
    parser.add_argument("--frames-dir", default=cfg["frames_dir"], help="Output dir for --extract-frames-only")
    parser.add_argument("--extract-frames-only", action="store_true", help="Only extract frames to disk, no API")
    parser.add_argument("--prompt-name", default=cfg["prompt_name"], help="Langfuse prompt name")
    parser.add_argument("--prompt-label", default=cfg["prompt_label"], help="Langfuse prompt label")
    parser.add_argument("--ab-prompt-labels", metavar="LABELS", help='A/B prompts: comma-separated labels')
    parser.add_argument("--model-b", help="Second model for A/B (use with --ab-test-models)")
    parser.add_argument("--ab-test-models", action="store_true", help="A/B test --model vs --model-b")
    parser.add_argument("--config", type=Path, help="Config YAML path (default: config.yaml next to script)")
    args = parser.parse_args()

    cfg = load_config(args.config if args.config else None)
    # Env / CLI overrides
    cfg["video"] = args.video
    cfg["endpoint"] = args.endpoint
    cfg["model"] = args.model
    cfg["max_tokens"] = args.max_tokens
    cfg["fps"] = args.fps
    cfg["max_frames"] = args.max_frames
    cfg["frames_dir"] = args.frames_dir
    cfg["prompt_name"] = args.prompt_name
    cfg["prompt_label"] = args.prompt_label

    if args.extract_frames_only:
        prompt_text = (cfg.get("fallback_prompt") or _DEFAULTS["fallback_prompt"]).strip()
        paths = extract_frames_to_dir(
            args.video, cfg["frames_dir"], cfg["fps"], cfg["max_frames"], prompt_text,
            cfg.get("image_patch_size", 16), cfg.get("jpeg_quality", 85), open_folder=True
        )
        for p in paths:
            print(p)
        return

    token = get_token()
    ab_labels = [x.strip() for x in (args.ab_prompt_labels or "").split(",") if x.strip()] if args.ab_prompt_labels else None
    prompt_text, langfuse_prompt = resolve_prompt(cfg, ab_labels=ab_labels if ab_labels and len(ab_labels) >= 2 else None)

    model = cfg["model"]
    trace_metadata = {}
    if args.ab_test_models and args.model_b:
        use_a = random.random() < 0.5
        model = args.model if use_a else args.model_b
        trace_metadata["ab_model"] = "a" if use_a else "b"
        print(f"{LOG_PREFIX} A/B model: {trace_metadata['ab_model']} -> {model}", file=sys.stderr)
    cfg["model"] = model

    print(f"{LOG_PREFIX} video={args.video!r}", file=sys.stderr)
    t0 = time.perf_counter()
    out = describe_video(args.video, cfg["endpoint"], token, cfg, prompt_text, langfuse_prompt, trace_metadata or None)
    print(f"{LOG_PREFIX} wall: {time.perf_counter() - t0:.2f}s", file=sys.stderr)
    print(out)

    if _langfuse_configured():
        try:
            get_client().flush()
        except Exception:
            pass


if __name__ == "__main__":
    main()
