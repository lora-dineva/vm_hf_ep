#!/usr/bin/env python3
"""
Detect scenes in a video with PySceneDetect and split into separate files.
Uses system ffmpeg if on PATH, otherwise the imageio-ffmpeg bundled binary. Default input: test_clip_60s.mp4.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

from scenedetect import detect, AdaptiveDetector, ContentDetector, ThresholdDetector
from scenedetect.video_splitter import split_video_ffmpeg, is_ffmpeg_available

def _ensure_ffmpeg_available() -> bool:
    """Return True if ffmpeg is available (on PATH or via imageio-ffmpeg). Sets video_splitter.FFMPEG_PATH when using bundle."""
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

DEFAULT_VIDEO = "test_clip_60s.mp4"
DEFAULT_OUTPUT_DIR = "scenes"

DETECTORS = {
    "adaptive": AdaptiveDetector,
    "content": ContentDetector,
    "threshold": ThresholdDetector,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect scenes and split video into one file per scene (uses ffmpeg from PATH or imageio-ffmpeg)."
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
        "--detector",
        "-d",
        choices=list(DETECTORS),
        default="adaptive",
        help="Detector: adaptive (default), content, or threshold",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only print scene boundaries (start/end), do not split",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    if not args.list_only and not _ensure_ffmpeg_available():
        print(
            "Error: ffmpeg is required for splitting but was not found. Install ffmpeg and ensure it is on your PATH, or reinstall imageio-ffmpeg.",
            file=sys.stderr,
        )
        sys.exit(1)

    detector_class = DETECTORS[args.detector]
    detector = detector_class()
    scene_list = detect(str(video_path), detector)

    if not scene_list:
        print("No scenes detected.", file=sys.stderr)
        sys.exit(0)

    print(f"Detected {len(scene_list)} scene(s).")
    for i, (start, end) in enumerate(scene_list, start=1):
        print(f"  Scene {i}: {start.get_timecode()} - {end.get_timecode()}")

    csv_dir = Path(args.output_dir) if not args.list_only else video_path.parent
    csv_path = csv_dir / f"{video_path.stem}_scenes.csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "start_time", "end_time"])
        for i, (start, end) in enumerate(scene_list, start=1):
            w.writerow([i, start.get_timecode(), end.get_timecode()])
    print(f"Wrote scene list to {csv_path}")

    if args.list_only:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_video_ffmpeg(
        str(video_path),
        scene_list,
        output_dir=output_dir,
        output_file_template="$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4",
    )
    print(f"Wrote {len(scene_list)} scene file(s) to {output_dir.absolute()}")


if __name__ == "__main__":
    main()
