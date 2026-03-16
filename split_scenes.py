#!/usr/bin/env python3
"""
Detect scenes with PySceneDetect and split into separate files.
Uses system ffmpeg if on PATH, otherwise the imageio-ffmpeg bundled binary.
"""

import argparse
import csv
import sys
from pathlib import Path

from scenedetect import detect, AdaptiveDetector, ContentDetector, ThresholdDetector
from scenedetect.video_splitter import split_video_ffmpeg

from utils import ensure_ffmpeg

DEFAULT_VIDEO = "test_clip_60s.mp4"
DEFAULT_OUTPUT_DIR = "scenes"

DETECTORS = {
    "adaptive": AdaptiveDetector,
    "content": ContentDetector,
    "threshold": ThresholdDetector,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect scenes and split video into one file per scene."
    )
    parser.add_argument("video", nargs="?", default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", "-o", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--detector", "-d", choices=list(DETECTORS), default="adaptive")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    if not args.list_only and not ensure_ffmpeg():
        print("Error: ffmpeg is required for splitting but was not found.", file=sys.stderr)
        sys.exit(1)

    if args.detector == "adaptive":
        detector = AdaptiveDetector(min_content_val=25.0, min_scene_len=30)
    else:
        detector = DETECTORS[args.detector]()
    scene_list = detect(str(video_path), detector)

    if not scene_list:
        print("No scenes detected.", file=sys.stderr)
        sys.exit(0)

    print(f"Detected {len(scene_list)} scene(s).")
    for i, (start, end) in enumerate(scene_list, start=1):
        print(f"  Scene {i}: {start.get_timecode()} - {end.get_timecode()}")

    output_dir = Path(args.output_dir)
    csv_path = output_dir / f"{video_path.stem}_scenes.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "start_time", "end_time"])
        for i, (start, end) in enumerate(scene_list, start=1):
            w.writerow([i, start.get_timecode(), end.get_timecode()])
    print(f"Wrote scene list to {csv_path}")

    if args.list_only:
        return

    split_video_ffmpeg(
        str(video_path), scene_list,
        output_dir=output_dir,
        output_file_template="$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4",
    )
    print(f"Wrote {len(scene_list)} scene file(s) to {output_dir.absolute()}")


if __name__ == "__main__":
    main()
