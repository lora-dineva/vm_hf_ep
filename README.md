# vm_hf_ep

Automated video description pipeline using vision-language models on Hugging Face Inference Endpoints.

Each pipeline run produces two descriptions of the same video and compares them:

- **Approach A (scene-based):** Split into scenes, describe each scene, merge into one narrative.
- **Approach B (full video):** Send the entire video in one pass, then style the raw output.

Both outputs go through a styling model and can be converted to audio.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file from `.env.example`. At minimum, set `HF_TOKEN`.

Edit `config.yaml` to adjust defaults (endpoints, models, sampling, TTS).

## Usage

### Full pipeline (recommended)

```bash
python run_pipeline.py video.mp4
python run_pipeline.py video.mp4 --detector adaptive
python run_pipeline.py video.mp4 --skip-split        # reuse existing scenes
python run_pipeline.py video.mp4 --no-audio           # skip TTS
```

Outputs `report.md` + `report.pdf` with parameters, per-scene descriptions, and both final descriptions side by side.

### Individual scripts

```bash
python split_scenes.py video.mp4 -o scenes
python split_scenes_transnetv2.py video.mp4 -o scenes
python describe_video.py --scenes-dir scenes
python describe_full_video.py video.mp4 -o output.txt
python generate_final_description.py scenes/video_scenes.csv -o final.txt
python generate_audio.py final.txt -o output.wav
python md_to_pdf.py report.md
```

## Project structure

```
config.yaml                  - pipeline configuration
utils.py                     - shared helpers (token, CSV, config, ffmpeg)
run_pipeline.py              - end-to-end orchestrator (both approaches)
split_scenes.py              - scene detection (PySceneDetect)
split_scenes_transnetv2.py   - scene detection (TransNetV2)
describe_video.py            - per-scene description via Qwen3-VL
describe_full_video.py       - full-video description (vision + styling)
generate_final_description.py - merge scene descriptions into one narrative
generate_audio.py            - text-to-speech (Kokoro-82M)
md_to_pdf.py                 - Markdown to PDF conversion
```

## Optional: Langfuse

Set `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_BASE_URL` in `.env` to enable prompt management and tracing through [Langfuse](https://langfuse.com).
