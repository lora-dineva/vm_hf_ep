#!/usr/bin/env python3
"""
Detect scenes with TransNetV2 and split into separate files.
Output format matches split_scenes.py (same CSV columns and clip naming).
Requires transnetv2-pytorch. Uses ffmpeg for splitting (PATH or imageio-ffmpeg).
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import torch
    from transnetv2_pytorch import TransNetV2
except ImportError:
    print("Error: Install transnetv2-pytorch: pip install transnetv2-pytorch", file=sys.stderr)
    sys.exit(1)

from scenedetect import FrameTimecode
from scenedetect.video_splitter import split_video_ffmpeg

from utils import ensure_ffmpeg, get_ffmpeg_exe

DEFAULT_VIDEO = "Kilavuz.mxf"
DEFAULT_OUTPUT_DIR = "scenes"


def _setup_ffmpeg() -> None:
    """Put the ffmpeg binary on PATH and patch ffmpeg-python to use it by full path."""
    exe = get_ffmpeg_exe()
    if not exe:
        return
    ffmpeg_dir = str(Path(exe).resolve().parent)
    if ffmpeg_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    try:
        import ffmpeg._run as _run
        _orig = _run.run_async
        def _patched(stream_spec, cmd="ffmpeg", **kwargs):
            if cmd == "ffmpeg":
                cmd = exe
            return _orig(stream_spec, cmd=cmd, **kwargs)
        _run.run_async = _patched
    except Exception:
        pass


def _get_video_fps(video_path: str, ffmpeg_exe: str) -> float:
    """Extract FPS via ffmpeg -i. Fallback 25.0."""
    try:
        result = subprocess.run(
            [ffmpeg_exe, "-i", video_path],
            capture_output=True, text=True, timeout=10,
        )
        text = result.stderr or ""
        m = re.search(r"(\d+(?:\.\d+)?)\s+fps", text)
        if m:
            return float(m.group(1))
        m = re.search(r"(\d+(?:\.\d+)?)\s+tbr", text)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return 25.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect scenes with TransNetV2 and split video."
    )
    parser.add_argument("video", nargs="?", default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", "-o", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", "-t", type=float, default=0.9)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    _setup_ffmpeg()
    ffmpeg_exe = get_ffmpeg_exe()
    if not ffmpeg_exe:
        print("Error: ffmpeg not found. Install ffmpeg or imageio-ffmpeg.", file=sys.stderr)
        sys.exit(1)

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    if not args.list_only and not ensure_ffmpeg():
        print("Error: ffmpeg is required for splitting.", file=sys.stderr)
        sys.exit(1)

    model = TransNetV2(device=args.device)
    video_str = str(video_path)

    real_fps = _get_video_fps(video_str, ffmpeg_exe)
    model.get_video_fps = lambda _path: real_fps
    print(f"Detected video FPS: {real_fps}", file=sys.stderr)

    with torch.no_grad():
        results = model.analyze_video(video_str, threshold=args.threshold)

    raw_scenes = results["scenes"]
    fps = float(results["fps"])

    if not raw_scenes:
        print("No scenes detected.", file=sys.stderr)
        sys.exit(0)

    MIN_SCENE_DURATION = 1.0  # seconds
    scenes = [s for s in raw_scenes if float(s["end_time"]) - float(s["start_time"]) >= MIN_SCENE_DURATION]
    dropped = len(raw_scenes) - len(scenes)
    if dropped:
        print(f"Dropped {dropped} scene(s) shorter than {MIN_SCENE_DURATION}s.", file=sys.stderr)

    if not scenes:
        print("No scenes remaining after minimum duration filter.", file=sys.stderr)
        sys.exit(0)

    print(f"Detected {len(scenes)} scene(s) (TransNetV2, threshold={args.threshold}).")
    for i, s in enumerate(scenes, start=1):
        print(f"  Scene {i}: {float(s['start_time']):.3f}s - {float(s['end_time']):.3f}s")

    scene_list = []
    for s in scenes:
        start_tc = FrameTimecode(timecode=float(s["start_time"]), fps=fps)
        end_tc = FrameTimecode(timecode=float(s["end_time"]), fps=fps)
        scene_list.append((start_tc, end_tc))

    output_dir = Path(args.output_dir)
    csv_path = output_dir / f"{video_path.stem}_scenes.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "start_time", "end_time"])
        for i, (start_tc, end_tc) in enumerate(scene_list, start=1):
            w.writerow([i, start_tc.get_timecode(), end_tc.get_timecode()])
    print(f"Wrote scene list to {csv_path}")

    if args.list_only:
        return

    split_video_ffmpeg(
        video_str, scene_list,
        output_dir=str(output_dir),
        output_file_template="$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4",
    )
    print(f"Wrote {len(scene_list)} scene file(s) to {output_dir.absolute()}")


if __name__ == "__main__":
    main()
