#!/usr/bin/env python3
"""
Test different PySceneDetect parameter combinations to compare scene counts.
Useful when the default split is too granular (too many short scenes).
Only runs detection and prints results; does not split video or write files.

Run: python test_scene_detection.py [video]
(May take 1–3 minutes for a 60 s video; runs several detector configs.)
"""

import argparse
import sys
from pathlib import Path

from scenedetect import detect, AdaptiveDetector, ContentDetector

# Default video (same as split_scenes.py)
DEFAULT_VIDEO = "test_clip_60s.mp4"


def run_tests(video_path: Path) -> None:
    """Run scene detection with several parameter presets and print a comparison."""
    video_str = str(video_path)
    results = []

    # -------------------------------------------------------------------------
    # Baseline: current split_scenes.py default (very granular)
    # -------------------------------------------------------------------------
    detector = AdaptiveDetector()
    scene_list = detect(video_str, detector)
    results.append(("AdaptiveDetector() [default, current behavior]", len(scene_list), scene_list))

    # -------------------------------------------------------------------------
    # AdaptiveDetector: longer minimum scene length → fewer scenes
    # -------------------------------------------------------------------------
    # ~1.25 s at 24 fps
    detector = AdaptiveDetector(min_scene_len=30)
    scene_list = detect(video_str, detector)
    results.append(("AdaptiveDetector(min_scene_len=30)", len(scene_list), scene_list))

    # ~2 s at 24 fps
    detector = AdaptiveDetector(min_scene_len=48)
    scene_list = detect(video_str, detector)
    results.append(("AdaptiveDetector(min_scene_len=48)", len(scene_list), scene_list))

    # ~3 s at 24 fps
    detector = AdaptiveDetector(min_scene_len=72)
    scene_list = detect(video_str, detector)
    results.append(("AdaptiveDetector(min_scene_len=72)", len(scene_list), scene_list))

    # -------------------------------------------------------------------------
    # AdaptiveDetector: higher threshold → fewer cuts (less sensitive)
    # -------------------------------------------------------------------------
    detector = AdaptiveDetector(adaptive_threshold=5.0)
    scene_list = detect(video_str, detector)
    results.append(("AdaptiveDetector(adaptive_threshold=5.0)", len(scene_list), scene_list))

    detector = AdaptiveDetector(adaptive_threshold=6.0, min_scene_len=24)
    scene_list = detect(video_str, detector)
    results.append(
        ("AdaptiveDetector(adaptive_threshold=6.0, min_scene_len=24)", len(scene_list), scene_list)
    )

    # -------------------------------------------------------------------------
    # AdaptiveDetector: require stronger content change
    # -------------------------------------------------------------------------
    detector = AdaptiveDetector(min_content_val=25.0, min_scene_len=30)
    scene_list = detect(video_str, detector)
    results.append(
        (
            "AdaptiveDetector(min_content_val=25.0, min_scene_len=30)",
            len(scene_list),
            scene_list,
        )
    )

    # -------------------------------------------------------------------------
    # ContentDetector: higher threshold = less sensitive → fewer scenes
    # -------------------------------------------------------------------------
    detector = ContentDetector(threshold=40.0)
    scene_list = detect(video_str, detector)
    results.append(("ContentDetector(threshold=40.0)", len(scene_list), scene_list))

    detector = ContentDetector(threshold=50.0, min_scene_len=30)
    scene_list = detect(video_str, detector)
    results.append(
        ("ContentDetector(threshold=50.0, min_scene_len=30)", len(scene_list), scene_list)
    )

    # -------------------------------------------------------------------------
    # Print summary table
    # -------------------------------------------------------------------------
    print(f"Video: {video_path}")
    print()
    name_width = max(len(r[0]) for r in results)
    print(f"{'Configuration':<{name_width}}  Scenes")
    print("-" * (name_width + 10))
    for name, count, _ in results:
        print(f"{name:<{name_width}}  {count}")

    # Optionally print scene boundaries for each preset (verbose)
    print()
    print("Scene boundaries (start – end) per configuration:")
    print()
    for name, count, scene_list in results:
        print(f"  {name}")
        if not scene_list:
            print("    (none)")
        else:
            for i, (start, end) in enumerate(scene_list, start=1):
                print(f"    Scene {i}: {start.get_timecode()} – {end.get_timecode()}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test scene detection with different parameters; only reports scene counts and boundaries."
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=DEFAULT_VIDEO,
        help=f"Video path (default: {DEFAULT_VIDEO})",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    run_tests(video_path)


if __name__ == "__main__":
    main()
