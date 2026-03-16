#!/usr/bin/env python3
"""
Detect scenes with TransNetV2 and split into separate files.
Output format matches split_scenes.py (same CSV columns and clip naming) so
describe_video.py can use the results. Requires transnetv2-pytorch (which
bundles model weights). Uses ffmpeg for splitting (PATH or imageio-ffmpeg).
"""

import argparse
import csv
import os
import sys
from pathlib import Path

# TransNetV2 — constructor auto-loads bundled weights and calls eval()
try:
    import torch
    from transnetv2_pytorch import TransNetV2
except ImportError:
    print(
        "Error: transnetv2-pytorch is required. Install with: pip install transnetv2-pytorch",
        file=sys.stderr,
    )
    sys.exit(1)

# PySceneDetect only for FrameTimecode + ffmpeg splitter (same as split_scenes.py)
from scenedetect import FrameTimecode
from scenedetect.video_splitter import split_video_ffmpeg, is_ffmpeg_available

DEFAULT_VIDEO = "test_clip_60s.mp4"
DEFAULT_OUTPUT_DIR = "scenes"


def _get_ffmpeg_exe() -> str | None:
    """Return full path to ffmpeg (from PATH or imageio-ffmpeg), or None."""
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


def _prepend_ffmpeg_to_path() -> None:
    """Put ffmpeg directory on PATH so ffmpeg-python and PySceneDetect can find it."""
    exe = _get_ffmpeg_exe()
    if exe:
        ffmpeg_dir = str(Path(exe).resolve().parent)
        if ffmpeg_dir not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")


def _patch_ffmpeg_python_to_use_exe(ffmpeg_exe: str) -> None:
    """Make ffmpeg-python use the given executable by full path (avoids PATH issues on Windows)."""
    try:
        import ffmpeg._run as _run

        _orig_run_async = _run.run_async.__wrapped__ if hasattr(_run.run_async, "__wrapped__") else _run.run_async

        def _patched_run_async(stream_spec, cmd="ffmpeg", **kwargs):
            if cmd == "ffmpeg":
                cmd = ffmpeg_exe
            return _orig_run_async(stream_spec, cmd=cmd, **kwargs)

        _run.run_async = _patched_run_async
    except Exception:
        pass

    pass  # ffprobe patch not needed; FPS is handled via _get_video_fps below


def _get_video_fps(video_path: str, ffmpeg_exe: str) -> float:
    """Extract FPS using ffmpeg -i (works without ffprobe). Fallback 25.0."""
    import re
    import subprocess

    try:
        result = subprocess.run(
            [ffmpeg_exe, "-i", video_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # ffmpeg -i exits with 1 but prints info to stderr
        text = result.stderr or ""
        # Match patterns like "25 fps", "50 fps", "29.97 fps", "25 tbr"
        m = re.search(r"(\d+(?:\.\d+)?)\s+fps", text)
        if m:
            return float(m.group(1))
        m = re.search(r"(\d+(?:\.\d+)?)\s+tbr", text)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return 25.0


def _ensure_ffmpeg_for_splitting() -> bool:
    """Return True if ffmpeg is available for PySceneDetect splitting."""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect scenes with TransNetV2 and split video (same CSV/clip format as split_scenes.py)."
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=DEFAULT_VIDEO,
        help=f"Input video path (default: {DEFAULT_VIDEO})",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for scene files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.5,
        help="Detection threshold 0–1 (default: 0.5)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device for TransNetV2 (default: auto)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only print scene boundaries and write CSV, do not split",
    )
    args = parser.parse_args()

    # TransNetV2 (and ffmpeg.probe) use ffmpeg-python which shells out to ffmpeg/ffprobe
    _prepend_ffmpeg_to_path()
    ffmpeg_exe = _get_ffmpeg_exe()
    if not ffmpeg_exe:
        print(
            "Error: ffmpeg not found. Install ffmpeg on PATH or install imageio-ffmpeg.",
            file=sys.stderr,
        )
        sys.exit(1)
    _patch_ffmpeg_python_to_use_exe(ffmpeg_exe)

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    if not args.list_only and not _ensure_ffmpeg_for_splitting():
        print(
            "Error: ffmpeg is required for splitting. Install ffmpeg or reinstall imageio-ffmpeg.",
            file=sys.stderr,
        )
        sys.exit(1)

    # TransNetV2 constructor auto-loads bundled weights and calls eval()
    model = TransNetV2(device=args.device)

    video_str = str(video_path)

    # Get reliable FPS via ffmpeg -i (imageio-ffmpeg doesn't ship ffprobe,
    # so TransNetV2.get_video_fps falls back to 25.0 which doubles timestamps
    # on 50 fps video). Monkey-patch the model so analyze_video uses the real FPS.
    real_fps = _get_video_fps(video_str, ffmpeg_exe)
    model.get_video_fps = lambda _path: real_fps
    print(f"Detected video FPS: {real_fps}", file=sys.stderr)

    with torch.no_grad():
        results = model.analyze_video(video_str, threshold=args.threshold)

    scenes = results["scenes"]
    fps = float(results["fps"])

    if not scenes:
        print("No scenes detected.", file=sys.stderr)
        sys.exit(0)

    print(f"Detected {len(scenes)} scene(s) (TransNetV2, threshold={args.threshold}).")
    for i, s in enumerate(scenes, start=1):
        start_s = float(s["start_time"])
        end_s = float(s["end_time"])
        print(f"  Scene {i}: {start_s:.3f}s – {end_s:.3f}s")

    # Build list of (start, end) FrameTimecode for split_video_ffmpeg
    scene_list = []
    for s in scenes:
        start_tc = FrameTimecode(timecode=float(s["start_time"]), fps=fps)
        end_tc = FrameTimecode(timecode=float(s["end_time"]), fps=fps)
        scene_list.append((start_tc, end_tc))

    # Write CSV (same format as split_scenes.py)
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

    # Split with same template as split_scenes.py
    split_video_ffmpeg(
        video_str,
        scene_list,
        output_dir=str(output_dir),
        output_file_template="$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4",
    )
    print(f"Wrote {len(scene_list)} scene file(s) to {output_dir.absolute()}")


if __name__ == "__main__":
    main()
