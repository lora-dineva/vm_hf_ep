#!/usr/bin/env python3
"""Shared utilities for the video description pipeline."""

import csv
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv

load_dotenv()


def get_token() -> str:
    """Return HF_TOKEN from environment, exit if missing."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("Error: Set HF_TOKEN in .env or environment.", file=sys.stderr)
        sys.exit(1)
    return token


def load_scenes_csv(csv_path: Path) -> List[dict]:
    """Load scene CSV into a list of dicts keyed by header columns."""
    if not csv_path.is_file():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_scenes_csv(
    csv_path: Path, rows: List[dict], fieldnames: List[str]
) -> None:
    """Write rows to a scene CSV with the given column order."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_config(path: Optional[Path] = None) -> dict:
    """Load config.yaml merged with built-in defaults. Returns a plain dict."""
    defaults = {
        "video": "test_clip_60s.mp4",
        "frames_dir": "frames",
        "scenes_dir": "scenes",
        "endpoint": "https://rjk11aiy6oefykan.us-east-1.aws.endpoints.huggingface.cloud",
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "max_tokens": 1024,
        "fps": 2.0,
        "max_frames": 120,
        "image_patch_size": 16,
        "jpeg_quality": 85,
        # Langfuse prompt names
        "prompt_label": "production",
        "prompt_scene_description": "video-description",
        "prompt_scene_merge": "scene-merge",
        "prompt_full_video_description": "full-video-description",
        "prompt_full_video_style": "full-video-style",
        # Final description (merge) settings
        "final_endpoint": "",
        "final_model": "mistralai/Mistral-Nemo-Instruct-2407",
        "final_max_tokens": 2048,
        # Scene splitting
        "scene_detector": "transnetv2",
    }
    config_path = path or (Path(__file__).resolve().parent / "config.yaml")
    out = defaults.copy()
    if not config_path.is_file():
        return out
    try:
        import yaml

        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if v is not None and k in out:
                out[k] = v
    except Exception as e:
        print(f"Warning: Could not load {config_path}: {e}", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# Langfuse prompt resolution
# ---------------------------------------------------------------------------

def langfuse_configured() -> bool:
    return bool(
        os.environ.get("LANGFUSE_SECRET_KEY") and os.environ.get("LANGFUSE_PUBLIC_KEY")
    )


def resolve_text_prompt(
    name: str,
    label: str,
    fallback: str,
) -> tuple[str, Any]:
    """Fetch a text prompt from Langfuse. Returns (prompt_text, langfuse_prompt_obj_or_None)."""
    if not langfuse_configured():
        return (fallback, None)
    try:
        from langfuse import get_client
        langfuse = get_client()
        prompt = langfuse.get_prompt(name, label=label)
        raw = prompt.compile() if hasattr(prompt, "compile") else str(prompt.prompt)
        if isinstance(raw, list):
            parts = [m.get("content", "") for m in raw
                     if isinstance(m, dict) and m.get("role") == "user"]
            raw = parts[-1] if parts else fallback
        text = str(raw).strip() or fallback
        return (text, prompt)
    except Exception as e:
        print(f"[langfuse] Failed to fetch prompt '{name}': {e}", file=sys.stderr)
        return (fallback, None)


def resolve_chat_prompt(
    name: str,
    label: str,
    fallback_system: str,
    fallback_user: str,
) -> tuple[str, str, Any]:
    """Fetch a chat prompt from Langfuse. Returns (system_text, user_text, langfuse_prompt_obj_or_None)."""
    if not langfuse_configured():
        return (fallback_system, fallback_user, None)
    try:
        from langfuse import get_client
        langfuse = get_client()
        prompt = langfuse.get_prompt(name, label=label, type="chat")
        raw = prompt.compile() if hasattr(prompt, "compile") else prompt.prompt
        if isinstance(raw, list):
            system = ""
            user = ""
            for msg in raw:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    system = content
                elif role == "user":
                    user = content
            system = system.strip() or fallback_system
            user = user.strip() or fallback_user
            return (system, user, prompt)
        return (fallback_system, fallback_user, prompt)
    except Exception as e:
        print(f"[langfuse] Failed to fetch chat prompt '{name}': {e}", file=sys.stderr)
        return (fallback_system, fallback_user, None)


def ensure_ffmpeg() -> bool:
    """Ensure ffmpeg is available (system PATH or imageio-ffmpeg bundle).

    Sets scenedetect's FFMPEG_PATH when using the bundled binary.
    Returns True if ffmpeg is ready.
    """
    from scenedetect.video_splitter import is_ffmpeg_available

    if is_ffmpeg_available():
        return True
    try:
        import imageio_ffmpeg
        import scenedetect.video_splitter as vs

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            vs.FFMPEG_PATH = exe
            return True
    except Exception:
        pass
    return False


def get_ffmpeg_exe() -> Optional[str]:
    """Return full path to an ffmpeg binary, or None."""
    import shutil

    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return str(Path(exe).resolve())
    except Exception:
        pass
    return None
