# vm_hf_ep

Automated video description pipeline using vision-language models on Hugging Face Inference Endpoints.

Given a video file, the pipeline:

1. Splits it into scenes (PySceneDetect or TransNetV2).
2. Describes each scene with Qwen3-VL.
3. Merges scene descriptions into a single narrative suitable for audio description / accessibility.
4. Produces a Markdown report (with PDF export) containing run parameters and output.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file from `.env.example`. At minimum, set `HF_TOKEN`.

Edit `config.yaml` to adjust defaults (video path, endpoints, models, sampling parameters). CLI arguments override config values.

## Usage

### Full pipeline (recommended)

```bash
python run_pipeline.py video.mp4
python run_pipeline.py video.mp4 --detector content --report my_report.md
python run_pipeline.py video.mp4 --skip-split          # reuse existing scenes
python run_pipeline.py video.mp4 --detector transnetv2  # use TransNetV2 instead
```

This runs all steps and writes `report.md` + `report.pdf`.

### Individual scripts

Split scenes:

```bash
python split_scenes.py video.mp4 -o scenes
python split_scenes.py video.mp4 --list-only
```

Describe scenes:

```bash
python describe_video.py video.mp4
python describe_video.py --scenes-dir scenes
```

Merge into final description:

```bash
python generate_final_description.py scenes/video_scenes.csv
python generate_final_description.py --video video.mp4 -o final.txt
```

Convert Markdown to PDF:

```bash
python md_to_pdf.py report.md
```

## Project structure

```
config.yaml                  - pipeline configuration
utils.py                     - shared helpers (token, CSV, config, ffmpeg)
run_pipeline.py              - end-to-end orchestrator
split_scenes.py              - scene detection (PySceneDetect)
split_scenes_transnetv2.py   - scene detection (TransNetV2, optional)
describe_video.py            - per-scene description via Qwen3-VL
generate_final_description.py - merge scene descriptions into one narrative
md_to_pdf.py                 - Markdown to PDF conversion
```

## Optional: Langfuse

Set `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_BASE_URL` in `.env` to enable prompt management, token tracking, and A/B testing through [Langfuse](https://langfuse.com).
