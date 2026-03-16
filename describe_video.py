#!/usr/bin/env python3
"""
Describe video scenes via a Hugging Face Inference Endpoint (Qwen3-VL).
Frames are sampled with qwen-vl-utils. Config lives in config.yaml; CLI overrides it.
"""

import argparse
import base64
import io
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

from langfuse.openai import OpenAI as LangfuseOpenAI
from qwen_vl_utils import process_vision_info

from utils import (
    get_token, load_config, load_scenes_csv, write_scenes_csv,
    langfuse_configured, resolve_text_prompt,
)

LOG_PREFIX = "[describe_video]"
SCENE_FILENAME_RE = re.compile(r"-Scene-(\d+)\.mp4$", re.IGNORECASE)
FALLBACK_PROMPT = "Describe the video."


def scene_id_from_path(path: Path) -> Optional[int]:
    m = SCENE_FILENAME_RE.search(path.name)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def _video_to_frames(
    path: str, fps: float, max_frames: int,
    prompt_text: str, image_patch_size: int,
) -> list:
    if not os.path.isfile(path):
        print(f"Error: Video file not found: {path}", file=sys.stderr)
        sys.exit(1)
    file_url = "file://" + os.path.abspath(path).replace("\\", "/")
    messages = [[{
        "role": "user",
        "content": [
            {"type": "video", "video": file_url, "fps": fps},
            {"type": "text", "text": prompt_text},
        ],
    }]]
    try:
        _, video_inputs, _ = process_vision_info(
            messages, return_video_kwargs=True,
            return_video_metadata=True, image_patch_size=image_patch_size,
        )
    except Exception as e:
        print(f"Error: qwen-vl-utils failed: {e}", file=sys.stderr)
        sys.exit(1)
    if not video_inputs:
        print("Error: No video output from process_vision_info.", file=sys.stderr)
        sys.exit(1)
    item = video_inputs[0]
    tensor = item[0] if isinstance(item, tuple) else item
    return [tensor[i] for i in range(min(tensor.shape[0], max_frames))]


def extract_frames_b64(
    path: str, fps: float, max_frames: int,
    prompt_text: str, image_patch_size: int = 16, jpeg_quality: int = 85,
) -> List[str]:
    """Return base64 JPEG strings for each sampled frame."""
    from PIL import Image as PILImage

    t0 = time.perf_counter()
    frames = _video_to_frames(path, fps, max_frames, prompt_text, image_patch_size)
    print(f"{LOG_PREFIX} process_vision_info: {time.perf_counter() - t0:.2f}s", file=sys.stderr)
    out = []
    for frame in frames:
        arr = frame.permute(1, 2, 0).cpu().numpy().clip(0, 255).astype("uint8")
        buf = io.BytesIO()
        PILImage.fromarray(arr).save(buf, format="JPEG", quality=jpeg_quality)
        out.append(base64.standard_b64encode(buf.getvalue()).decode("ascii"))
    print(f"{LOG_PREFIX} encoded {len(out)} frames: {time.perf_counter() - t0:.2f}s", file=sys.stderr)
    return out


def extract_frames_to_dir(
    path: str, output_dir: str, fps: float, max_frames: int,
    prompt_text: str, image_patch_size: int = 16,
    jpeg_quality: int = 85, open_folder: bool = True,
) -> List[str]:
    """Save sampled frames as JPEGs to disk; return list of paths."""
    from PIL import Image as PILImage

    t0 = time.perf_counter()
    frames = _video_to_frames(path, fps, max_frames, prompt_text, image_patch_size)
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, frame in enumerate(frames):
        arr = frame.permute(1, 2, 0).cpu().numpy().clip(0, 255).astype("uint8")
        p = os.path.join(output_dir, f"frame_{i:03d}.jpg")
        PILImage.fromarray(arr).save(p, format="JPEG", quality=jpeg_quality)
        paths.append(p)
    elapsed = time.perf_counter() - t0
    print(f"{LOG_PREFIX} saved {len(paths)} frames to {os.path.abspath(output_dir)} in {elapsed:.2f}s", file=sys.stderr)
    if open_folder:
        abs_out = os.path.abspath(output_dir)
        if sys.platform == "win32":
            os.startfile(abs_out)
        elif sys.platform == "darwin":
            subprocess.run(["open", abs_out], check=False)
        else:
            subprocess.run(["xdg-open", abs_out], check=False)
    return paths


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def describe_video(
    video_path: str, base_url: str, token: str, cfg: dict,
    prompt_text: str, langfuse_prompt: Any = None,
    trace_metadata: Optional[dict] = None,
) -> str:
    """Send sampled frames to HF endpoint; return description text."""
    t0 = time.perf_counter()
    frames_b64 = extract_frames_b64(
        video_path, cfg["fps"], cfg["max_frames"], prompt_text,
        cfg.get("image_patch_size", 16), cfg.get("jpeg_quality", 85),
    )
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        for b64 in frames_b64
    ]
    content.append({"type": "text", "text": prompt_text})
    print(f"{LOG_PREFIX} prepared {len(frames_b64)} frame(s)", file=sys.stderr)

    client = LangfuseOpenAI(base_url=base_url.rstrip("/") + "/v1", api_key=token)
    kwargs: dict[str, Any] = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": content}],
        "max_tokens": cfg["max_tokens"],
    }
    if langfuse_prompt is not None:
        kwargs["langfuse_prompt"] = langfuse_prompt
    if trace_metadata:
        kwargs["metadata"] = trace_metadata

    t1 = time.perf_counter()
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        print(f"Error: API request failed: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"{LOG_PREFIX} API: {time.perf_counter() - t1:.2f}s, total: {time.perf_counter() - t0:.2f}s", file=sys.stderr)

    if not resp.choices or not resp.choices[0].message.content:
        print("Error: No content in response.", file=sys.stderr)
        sys.exit(1)
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()
    cfg["endpoint"] = os.environ.get("HF_ENDPOINT") or cfg["endpoint"]

    parser = argparse.ArgumentParser(description="Describe a video via Hugging Face VL endpoint.")
    parser.add_argument("video", nargs="?", default=None)
    parser.add_argument("--scenes-dir", default=cfg.get("scenes_dir", "scenes"))
    parser.add_argument("--endpoint", default=cfg["endpoint"])
    parser.add_argument("--model", default=cfg["model"])
    parser.add_argument("--max-tokens", type=int, default=cfg["max_tokens"])
    parser.add_argument("--fps", type=float, default=cfg["fps"])
    parser.add_argument("--max-frames", type=int, default=cfg["max_frames"])
    parser.add_argument("--frames-dir", default=cfg["frames_dir"])
    parser.add_argument("--extract-frames-only", action="store_true")
    parser.add_argument("--prompt-label", default=cfg["prompt_label"])
    parser.add_argument("--ab-prompt-labels", metavar="LABELS")
    parser.add_argument("--model-b")
    parser.add_argument("--ab-test-models", action="store_true")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["scenes_dir"] = args.scenes_dir
    cfg["video"] = args.video
    cfg["endpoint"] = args.endpoint
    cfg["model"] = args.model
    cfg["max_tokens"] = args.max_tokens
    cfg["fps"] = args.fps
    cfg["max_frames"] = args.max_frames
    cfg["frames_dir"] = args.frames_dir
    cfg["prompt_label"] = args.prompt_label

    # Resolve video paths
    if args.video and Path(args.video).is_file():
        video_paths = [Path(args.video)]
    else:
        scenes_dir = Path(cfg["scenes_dir"])
        if not scenes_dir.is_dir():
            print(f"Error: No video given and scenes dir not found: {scenes_dir}", file=sys.stderr)
            sys.exit(1)
        video_paths = sorted(scenes_dir.glob("*.mp4"))
        if not video_paths:
            print(f"Error: No .mp4 files in {scenes_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"{LOG_PREFIX} describing {len(video_paths)} scene(s) from {scenes_dir}", file=sys.stderr)

    if args.extract_frames_only:
        for vp in video_paths:
            out_dir = str(Path(cfg["frames_dir"]) / vp.stem) if len(video_paths) > 1 else cfg["frames_dir"]
            paths = extract_frames_to_dir(
                str(vp), out_dir, cfg["fps"], cfg["max_frames"], FALLBACK_PROMPT,
                cfg.get("image_patch_size", 16), cfg.get("jpeg_quality", 85),
                open_folder=(vp == video_paths[-1]),
            )
            for p in paths:
                print(p)
        return

    token = get_token()

    # Resolve prompt from Langfuse (or fallback)
    prompt_name = cfg.get("prompt_scene_description", "video-description")
    label = cfg["prompt_label"]

    if args.ab_prompt_labels:
        ab_labels = [x.strip() for x in args.ab_prompt_labels.split(",") if x.strip()]
        if len(ab_labels) >= 2:
            chosen_label = random.choice(ab_labels)
            prompt_text, langfuse_prompt = resolve_text_prompt(prompt_name, chosen_label, FALLBACK_PROMPT)
        else:
            prompt_text, langfuse_prompt = resolve_text_prompt(prompt_name, label, FALLBACK_PROMPT)
    else:
        prompt_text, langfuse_prompt = resolve_text_prompt(prompt_name, label, FALLBACK_PROMPT)

    model = cfg["model"]
    trace_metadata: dict = {}
    if args.ab_test_models and args.model_b:
        use_a = random.random() < 0.5
        model = args.model if use_a else args.model_b
        trace_metadata["ab_model"] = "a" if use_a else "b"
        print(f"{LOG_PREFIX} A/B model: {trace_metadata['ab_model']} -> {model}", file=sys.stderr)
    cfg["model"] = model

    # Load scene CSV if describing a directory of scenes
    scene_rows: List[dict] = []
    scene_csv_path: Optional[Path] = None
    if len(video_paths) > 1:
        first_stem = video_paths[0].stem
        if "-Scene-" in first_stem:
            prefix = first_stem.rsplit("-Scene-", 1)[0]
            scene_csv_path = Path(cfg["scenes_dir"]) / f"{prefix}_scenes.csv"
            scene_rows = load_scenes_csv(scene_csv_path)
            if scene_rows:
                print(f"{LOG_PREFIX} will update {scene_csv_path} with descriptions", file=sys.stderr)

    descriptions_by_id: dict[int, str] = {}
    for idx, vp in enumerate(video_paths, start=1):
        print(f"{LOG_PREFIX} [{idx}/{len(video_paths)}] {vp.name}", file=sys.stderr)
        t0 = time.perf_counter()
        out = describe_video(str(vp), cfg["endpoint"], token, cfg, prompt_text, langfuse_prompt, trace_metadata or None)
        print(f"{LOG_PREFIX} wall: {time.perf_counter() - t0:.2f}s", file=sys.stderr)
        print(f"[{vp.name}] {out}")
        sid = scene_id_from_path(vp)
        if sid is not None:
            descriptions_by_id[sid] = out

    if scene_csv_path and scene_rows and descriptions_by_id:
        fieldnames = list(scene_rows[0].keys())
        if "description" not in fieldnames:
            fieldnames.append("description")
        for row in scene_rows:
            sid = row.get("scene_id")
            if sid is not None:
                try:
                    sid_int = int(sid)
                    if sid_int in descriptions_by_id:
                        row["description"] = descriptions_by_id[sid_int]
                except (TypeError, ValueError):
                    pass
        write_scenes_csv(scene_csv_path, scene_rows, fieldnames)
        print(f"{LOG_PREFIX} updated {scene_csv_path} with {len(descriptions_by_id)} description(s)", file=sys.stderr)

    if langfuse_configured():
        try:
            from langfuse import get_client
            get_client().flush()
        except Exception:
            pass


if __name__ == "__main__":
    main()
