# Video Audio Description Pipeline - Report

## Overview

This report documents the automated video audio-description pipeline built to generate accessibility narration tracks for video content. The pipeline was tested on **Kilavuz.mxf** ("The Guide"), a ~15.5-minute Turkish short film (1920x1080, 25 fps).

The pipeline follows a scene-based approach: split the video into scenes, describe each scene with a vision-language model, generate per-scene text-to-speech audio, and combine everything into a final video with narration preceding each scene.

---

## Pipeline Steps

### 1. Scene Splitting (TransNetV2)

Scenes were detected using **TransNetV2**, a deep-learning shot boundary detection model.

| Parameter | Value |
|---|---|
| Model | `transnetv2-pytorch` |
| Threshold | 0.9 (default) |
| Min scene duration | 1.0s (scenes shorter than this are dropped) |
| Device | auto |
| Source video | `Kilavuz.mxf` |

**Result:** 36 scenes detected, ranging from 2.5s (scene 4) to 99.5s (scene 21). Scene boundaries and timecodes were written to `scenes/Kilavuz_scenes.csv`, and each scene was split into an individual `.mp4` file.

### 2. Scene Description (Qwen3-VL)

Each scene clip was sampled into frames and sent to a **Qwen3-VL-8B-Instruct** vision-language model hosted on a Hugging Face Inference Endpoint.

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen3-VL-8B-Instruct` |
| Endpoint | HF Inference Endpoint (us-east-1) |
| FPS (frame sampling) | 1.0 |
| Max frames | 2000 |
| Max tokens | 400 |
| Image patch size | 16 |
| JPEG quality | 85 |
| Prompt label | `production` |
| Scenes processed | 36 |
| Total processing time | **1415.7s (~23.6 min)** |

**Langfuse prompt** (name: `video-description`, label: `production`):

> You are an expert Audio Describer creating accessibility tracks for visually impaired audiences. Describe scenes concisely. Focus on characters and actions. NO CAMERA TERMINOLOGY: NEVER use words like "shot", "cut", "camera", "angle", "frame", "pan" or "zooms".Give short description.

Descriptions were written back into the `description` column of `Kilavuz_scenes.csv`.

### 3. Audio Generation (Kokoro-82M)

Per-scene descriptions were converted to speech using the **Kokoro-82M** text-to-speech model via HF Inference (fal-ai provider).

| Parameter | Value |
|---|---|
| Model | `hexgrad/Kokoro-82M` |
| Provider | fal-ai |
| Input | `scenes/Kilavuz_scenes.csv` (36 scenes) |
| Output | 36 `.wav` files in `scenes/audio/` |
| Total processing time | **144.2s (~2.4 min)** |

### 4. Combining Video and Audio

Scene videos and audio narrations were combined into a single output video. Each scene is preceded by its narration played over a black screen.

| Parameter | Value |
|---|---|
| Resolution | 1920x1080 |
| FPS | 25.0 |
| Audio format | AAC, 48 kHz stereo |
| Scenes | 36 (all with audio) |
| Output | `final.mp4` |
| Total processing time | **806.8s (~13.4 min)** |

## Total Processing Time

| Step | Duration |
|---|---|
| Scene splitting (TransNetV2) | ~1 min (local GPU) |
| Scene description (Qwen3-VL) | 23.6 min |
| Audio generation (Kokoro-82M) | 2.4 min |
| Video + audio combination | 13.4 min |
| **Total** | **~40.4 min** |

---

## Constraints and Known Limitations

### FPS set to 1.0

Frame sampling was set to **1 FPS** (instead of the more typical 2-4 FPS) to speed up processing. This means faster extraction and fewer frames sent to the model per scene, reducing both latency and inference cost. The trade-off is that rapid actions or brief visual details between sampled seconds may be missed.

### Continuity across scenes

The current pipeline describes each scene **independently**, with no awareness of what happened before or after. This leads to a loss of spatial and narrative continuity:

- Characters are described by appearance each time ("a man in a suit") rather than by name or consistent identifier.
- The same location may be described differently across scenes.
- Cause-and-effect relationships spanning scene boundaries are not captured.

### Potential improvements for continuity

Depending on the final use case and available inputs, further preprocessing can address these gaps:

- **Script availability:** If a screenplay or subtitle file is available, it can be passed as context to the description model to anchor character names, locations, and plot points.
- **Face recognition:** A face detection/recognition pass (e.g., InsightFace, DeepFace) could identify recurring characters across scenes and assign consistent names or labels.
- **Iterative text generation:** A final LLM pass over all scene descriptions can enforce cross-scene consistency, resolve character references, and smooth transitions. The existing `style_description.py` scene-merge step partially addresses this but could be extended with multi-turn refinement.

---

## Current Costs

| Category | Cost |
|---|---|
| Development hours | ~30 hours |
| Cursor (IDE / AI assistant) | $50 |
| Model inference (HF endpoints + fal-ai) | $50 |
| **Total** | **~$100 + 30h development time** |
