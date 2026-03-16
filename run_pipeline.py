#!/usr/bin/env python3
"""
End-to-end video description pipeline.

Runs two approaches on the same video and produces a comparison report:

  Approach A -- Scene-based:
    1. Split video into scenes
    2. Describe each scene with Qwen3-VL
    3. Merge scene descriptions with the styling model

  Approach B -- Full video:
    1. Send entire video to Qwen3-VL (single pass)
    2. Style the raw output with the styling model

  Then:
    - Generate audio from each final description
    - Write a Markdown + PDF report comparing both approaches
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
    scene_final: str,
    fullvid_raw: str,
    fullvid_styled: str,
    scene_audio: Path | None,
    fullvid_audio: Path | None,
    elapsed_s: float,
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Video Description Report",
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
        f"| Vision model | `{cfg['model']}` |",
        f"| Vision endpoint | `{cfg['endpoint']}` |",
        f"| FPS (sampling) | {cfg['fps']} |",
        f"| Max frames | {cfg['max_frames']} |",
        f"| Max tokens (vision) | {cfg['max_tokens']} |",
        f"| Styling model | `{args.final_model}` |",
        f"| Styling endpoint | `{args.final_endpoint}` |",
        f"| Styling max tokens | {args.final_max_tokens} |",
        f"| TTS model | `{args.tts_model}` |",
        f"| Total time | {elapsed_s:.1f}s |",
        "",
        "---",
        "",
        "# Approach A: Scene-Based Pipeline",
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
        "## Scene-Based Final Description",
        "",
        scene_final,
        "",
    ]
    if scene_audio:
        lines.append(f"Audio: `{scene_audio}`")
        lines.append("")

    lines += [
        "---",
        "",
        "# Approach B: Full Video (Single Pass)",
        "",
        "## Raw Vision Output",
        "",
        fullvid_raw,
        "",
        "## Styled Final Description",
        "",
        fullvid_styled,
        "",
    ]
    if fullvid_audio:
        lines.append(f"Audio: `{fullvid_audio}`")
        lines.append("")

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


def _generate_audio(text_path: Path, audio_path: Path, args) -> Path | None:
    """Generate audio, return path on success or None."""
    try:
        _run(
            [sys.executable, "generate_audio.py",
             str(text_path), "-o", str(audio_path),
             "--model", args.tts_model, "--provider", args.tts_provider],
            f"Generate audio -> {audio_path.name}",
        )
        return audio_path if audio_path.is_file() else None
    except SystemExit:
        return None


def main() -> None:
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description="Run both scene-based and full-video description, then compare."
    )
    parser.add_argument("video", nargs="?", default=cfg.get("video", "test_clip_60s.mp4"))
    parser.add_argument("--scenes-dir", default=cfg.get("scenes_dir", "scenes"))
    parser.add_argument(
        "--detector", choices=("adaptive", "content", "threshold", "transnetv2"),
        default=cfg.get("scene_detector", "transnetv2"),
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
    parser.add_argument("--final-temperature", type=float, default=cfg.get("final_temperature", 0.7))
    parser.add_argument("--tts-model", default=cfg.get("tts_model", "hexgrad/Kokoro-82M"))
    parser.add_argument("--tts-provider", default=cfg.get("tts_provider", "fal-ai"))
    parser.add_argument("--no-audio", action="store_true", help="Skip audio generation")
    parser.add_argument("--report", default="report.md")
    parser.add_argument("--skip-split", action="store_true", help="Reuse existing scenes")
    parser.add_argument("--scenes-only", action="store_true", help="Run only Approach A (scene-based)")
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

    # ================================================================
    # APPROACH A: Scene-based pipeline
    # ================================================================

    # A1: Split scenes
    if not args.skip_split:
        if args.detector == "transnetv2":
            _run(
                [sys.executable, "split_scenes_transnetv2.py", str(video_path),
                 "-o", str(scenes_dir)],
                "A1: Split scenes (TransNetV2)",
            )
        else:
            _run(
                [sys.executable, "split_scenes.py", str(video_path),
                 "-o", str(scenes_dir), "-d", args.detector],
                f"A1: Split scenes ({args.detector})",
            )
    else:
        print(f"{LOG_PREFIX} skipping split, reusing {scenes_dir}", file=sys.stderr)

    if not csv_path.is_file():
        print(f"Error: Scene CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    scene_rows = load_scenes_csv(csv_path)
    scene_count = len(scene_rows)
    print(f"{LOG_PREFIX} {scene_count} scene(s)", file=sys.stderr)

    # A2: Describe each scene
    _run(
        [sys.executable, "describe_video.py",
         "--scenes-dir", str(scenes_dir),
         "--endpoint", args.endpoint,
         "--model", args.model,
         "--max-tokens", str(args.max_tokens),
         "--fps", str(args.fps),
         "--max-frames", str(args.max_frames)],
        "A2: Describe scenes",
    )
    scene_rows = load_scenes_csv(csv_path)

    # A3: Merge into final description
    scene_final_path = scenes_dir / f"{stem}_scene_final.txt"
    _run(
        [sys.executable, "generate_final_description.py",
         str(csv_path),
         "--endpoint", args.final_endpoint,
         "--model", args.final_model,
         "--max-tokens", str(args.final_max_tokens),
         "--temperature", str(args.final_temperature),
         "-o", str(scene_final_path)],
        "A3: Merge scene descriptions",
    )
    scene_final = scene_final_path.read_text(encoding="utf-8").strip() if scene_final_path.is_file() else ""

    # ================================================================
    # APPROACH B: Full video (single pass)
    # ================================================================

    fullvid_raw = ""
    fullvid_styled = ""
    if not args.scenes_only:
        fullvid_raw_path = scenes_dir / f"{stem}_fullvid_raw.txt"
        fullvid_styled_path = scenes_dir / f"{stem}_fullvid_styled.txt"

        _run(
            [sys.executable, "describe_full_video.py", str(video_path),
             "--endpoint", args.endpoint,
             "--model", args.model,
             "--max-tokens", str(args.max_tokens),
             "--fps", str(args.fps),
             "--max-frames", str(args.max_frames),
             "--final-endpoint", args.final_endpoint,
             "--final-model", args.final_model,
             "--final-max-tokens", str(args.final_max_tokens),
             "-o", str(fullvid_styled_path)],
            "B: Full video description",
        )

        fullvid_styled = fullvid_styled_path.read_text(encoding="utf-8").strip() if fullvid_styled_path.is_file() else ""
        fullvid_raw = fullvid_raw_path.read_text(encoding="utf-8").strip() if fullvid_raw_path.is_file() else ""
    else:
        print(f"{LOG_PREFIX} skipping Approach B (--scenes-only)", file=sys.stderr)

    # ================================================================
    # AUDIO
    # ================================================================

    scene_audio = None
    fullvid_audio = None
    if not args.no_audio:
        if scene_final:
            scene_audio = _generate_audio(
                scene_final_path, scenes_dir / f"{stem}_scene_audio.wav", args)
        if not args.scenes_only and fullvid_styled:
            fullvid_audio = _generate_audio(
                fullvid_styled_path, scenes_dir / f"{stem}_fullvid_audio.wav", args)
    else:
        print(f"{LOG_PREFIX} skipping audio", file=sys.stderr)

    elapsed = time.perf_counter() - t_start

    # ================================================================
    # REPORT
    # ================================================================

    report_path = Path(args.report)
    _write_report(
        report_path, cfg, args, scene_count, scene_rows,
        scene_final, fullvid_raw, fullvid_styled,
        scene_audio, fullvid_audio, elapsed,
    )
    _export_pdf(report_path)

    print(f"\n{LOG_PREFIX} Done in {elapsed:.1f}s", file=sys.stderr)
    print(f"{LOG_PREFIX} Report: {report_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
