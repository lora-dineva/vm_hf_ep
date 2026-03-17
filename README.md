# vm_hf_ep

Automated video description pipeline using vision-language models on Hugging Face Inference Endpoints.

Two independent approaches for generating descriptions, plus shared styling and audio steps.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file from `.env.example`. At minimum, set `HF_TOKEN`.

Edit `config.yaml` to adjust defaults (endpoints, models, sampling, TTS).

## Approach 1: Full Video

Describe the entire video in a single pass with Qwen3-VL, then style and generate audio.

```bash
# 1. Vision: raw description
python describe_full_video.py video.mxf -o scenes/raw.txt

# 2. Style with LLM
python style_description.py scenes/raw.txt -o scenes/styled.txt

# 3. Audio
python generate_audio.py scenes/styled.txt -o scenes/audio.wav
```

## Approach 2: Scene-Based

Split the video into scenes, describe each one, then style and generate per-scene audio.

```bash
# 1. Split into scenes
python split_scenes_transnetv2.py video.mxf -o scenes

# 2. Describe each scene
python describe_video.py --scenes-dir scenes

# 3. Style (merge scenes into one narrative)
python style_description.py scenes/video_scenes.csv -o scenes/styled.txt

# 4. Audio (one .wav per scene)
python generate_audio.py scenes/video_scenes.csv --output-dir scenes/audio/
```

## Styling Options

`style_description.py` auto-detects the input type (CSV = scene merge, text = full-video rewrite).

```bash
# Override Langfuse prompt label
python style_description.py input.txt --prompt-label experiment_v2 -o out.txt

# Override Langfuse prompt name
python style_description.py input.csv --prompt-name my-custom-prompt -o out.txt

# Adjust model parameters
python style_description.py input.txt --temperature 0.3 --max-tokens 4096 -o out.txt
```

## Run Log

Every script run is recorded in `runs.jsonl` (one JSON object per line) with:
- timestamp, script name, parameters, processing time, output file paths

```bash
# View recent runs
python -c "import json, pathlib; [print(json.dumps(json.loads(l), indent=2)) for l in pathlib.Path('runs.jsonl').read_text().splitlines()[-3:]]"
```

## Project Structure

```
config.yaml                - pipeline configuration
utils.py                   - shared helpers (token, CSV, config, Langfuse, ffmpeg, run log)
describe_full_video.py     - full-video description via Qwen3-VL (vision only)
split_scenes.py            - scene detection (PySceneDetect)
split_scenes_transnetv2.py - scene detection (TransNetV2)
describe_video.py          - per-scene description via Qwen3-VL
style_description.py       - style any description with an LLM (scene merge or full-video)
generate_audio.py          - text-to-speech (single file or per-scene CSV)
```

## Langfuse Integration

Set `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_BASE_URL` in `.env` to enable prompt management and tracing through [Langfuse](https://langfuse.com).

Prompt names and labels are configured in `config.yaml` and can be overridden via CLI (`--prompt-label`, `--prompt-name`).
