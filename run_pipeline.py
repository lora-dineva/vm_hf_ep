#!/usr/bin/env python3
"""
End-to-end video description pipeline.

Steps:
  1. Split video into scenes  (split_scenes.py / split_scenes_transnetv2.py)
  2. Describe each scene       (describe_video.py)
  3. Merge into final narrative (generate_final_description.py)
  4. Write a Markdown report with run parameters and output, export to PDF.
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from utils import load_config, load_scenes_csv

LOG_PREFIX = "[pipeline]"


def _run(cmd: list[str], label: str) -> subprocess.CompletedProcess:
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"{LOG_PREFIX} {label}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"{LOG_PREFIX} FAILED: {label} (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)
    return result


def _write_report(
    report_path: Path,
    cfg: dict,
    args,
    scene_count: int,
    scene_descriptions: list[dict],
    final_description: str,
    elapsed_s: float,
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Video Description Report",
        "",
        f"Generated: {now}",
        "",
        "---",
        "",
        "## Run Parameters",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
        f"| Video | `{cfg.get('video', 'N/A')}` |",
        f"| Scene detector | `{args.detector}` |",
        f"| Scenes detected | {scene_count} |",
        f"| Description model | `{cfg['model']}` |",
        f"| Description endpoint | `{cfg['endpoint']}` |",
        f"| FPS (sampling) | {cfg['fps']} |",
        f"| Max frames | {cfg['max_frames']} |",
        f"| Max tokens (scene) | {cfg['max_tokens']} |",
        f"| Final model | `{args.final_model}` |",
        f"| Final endpoint | `{args.final_endpoint}` |",
        f"| Final max tokens | {args.final_max_tokens} |",
        f"| Total time | {elapsed_s:.1f}s |",
        "",
        "---",
        "",
        "## Per-Scene Descriptions",
        "",
    ]
    for row in scene_descriptions:
        sid = row.get("scene_id", "?")
        start = row.get("start_time", "")
        end = row.get("end_time", "")
        desc = (row.get("description") or "").strip()
        lines.append(f"### Scene {sid} ({start} - {end})")
        lines.append("")
        lines.append(desc)
        lines.append("")

    lines += [
        "---",
        "",
        "## Final Description",
        "",
        final_description,
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"{LOG_PREFIX} report: {report_path}", file=sys.stderr)


def _export_pdf(md_path: Path) -> None:
    pdf_path = md_path.with_suffix(".pdf")
    try:
        from md2pdf.core import md2pdf as convert
        convert(str(pdf_path), md_file_path=str(md_path))
        print(f"{LOG_PREFIX} PDF: {pdf_path}", file=sys.stderr)
    except ImportError:
        print(f"{LOG_PREFIX} md2pdf not installed, skipping PDF export.", file=sys.stderr)
    except Exception as e:
        print(f"{LOG_PREFIX} PDF export failed: {e}", file=sys.stderr)


def main() -> None:
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description="Run the full pipeline: split -> describe -> merge -> report."
    )
    parser.add_argument("video", nargs="?", default=cfg.get("video", "test_clip_60s.mp4"))
    parser.add_argument("--scenes-dir", default=cfg.get("scenes_dir", "scenes"))
    parser.add_argument(
        "--detector", choices=("adaptive", "content", "threshold", "transnetv2"),
        default=cfg.get("scene_detector", "adaptive"),
    )
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT") or cfg["endpoint"])
    parser.add_argument("--model", default=cfg["model"])
    parser.add_argument("--max-tokens", type=int, default=cfg["max_tokens"])
    parser.add_argument("--fps", type=float, default=cfg["fps"])
    parser.add_argument("--max-frames", type=int, default=cfg["max_frames"])
    parser.add_argument("--final-endpoint", default=(
        os.environ.get("HF_FINAL_ENDPOINT")
        or cfg.get("final_endpoint")
        or "https://d3hbhiiq7llzxs1q.us-east-1.aws.endpoints.huggingface.cloud"
    ))
    parser.add_argument("--final-model", default=(
        os.environ.get("HF_FINAL_MODEL")
        or cfg.get("final_model", "mistralai/Mistral-Nemo-Instruct-2407")
    ))
    parser.add_argument("--final-max-tokens", type=int, default=cfg.get("final_max_tokens", 2048))
    parser.add_argument("--report", default="report.md", help="Output report path")
    parser.add_argument("--skip-split", action="store_true", help="Skip scene splitting (reuse existing scenes)")
    args = parser.parse_args()

    cfg["video"] = args.video
    cfg["endpoint"] = args.endpoint
    cfg["model"] = args.model
    cfg["max_tokens"] = args.max_tokens
    cfg["fps"] = args.fps
    cfg["max_frames"] = args.max_frames

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"Error: Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    scenes_dir = Path(args.scenes_dir)
    stem = video_path.stem
    csv_path = scenes_dir / f"{stem}_scenes.csv"

    t_start = time.perf_counter()

    # --- Step 1: Split scenes ---
    if not args.skip_split:
        if args.detector == "transnetv2":
            _run(
                [sys.executable, "split_scenes_transnetv2.py", str(video_path),
                 "-o", str(scenes_dir)],
                "Split scenes (TransNetV2)",
            )
        else:
            _run(
                [sys.executable, "split_scenes.py", str(video_path),
                 "-o", str(scenes_dir), "-d", args.detector],
                f"Split scenes ({args.detector})",
            )
    else:
        print(f"{LOG_PREFIX} skipping split, reusing {scenes_dir}", file=sys.stderr)

    if not csv_path.is_file():
        print(f"Error: Expected scene CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    scene_rows = load_scenes_csv(csv_path)
    scene_count = len(scene_rows)
    print(f"{LOG_PREFIX} {scene_count} scene(s) in {csv_path}", file=sys.stderr)

    # --- Step 2: Describe each scene ---
    _run(
        [sys.executable, "describe_video.py",
         "--scenes-dir", str(scenes_dir),
         "--endpoint", args.endpoint,
         "--model", args.model,
         "--max-tokens", str(args.max_tokens),
         "--fps", str(args.fps),
         "--max-frames", str(args.max_frames)],
        "Describe scenes",
    )

    # Reload CSV (now has descriptions)
    scene_rows = load_scenes_csv(csv_path)

    # --- Step 3: Final merged description ---
    final_output_path = scenes_dir / f"{stem}_final.txt"
    _run(
        [sys.executable, "generate_final_description.py",
         str(csv_path),
         "--endpoint", args.final_endpoint,
         "--model", args.final_model,
         "--max-tokens", str(args.final_max_tokens),
         "-o", str(final_output_path)],
        "Generate final description",
    )

    final_description = final_output_path.read_text(encoding="utf-8").strip() if final_output_path.is_file() else ""

    elapsed = time.perf_counter() - t_start

    # --- Step 4: Report ---
    report_path = Path(args.report)
    _write_report(report_path, cfg, args, scene_count, scene_rows, final_description, elapsed)
    _export_pdf(report_path)

    print(f"\n{LOG_PREFIX} Done in {elapsed:.1f}s", file=sys.stderr)
    print(f"{LOG_PREFIX} Report: {report_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
