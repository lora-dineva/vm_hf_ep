# Video Description PoC

## What Was Done

A 60-second video clip was automatically described by a vision–language model. The pipeline:

1. **Video processing** — Frames were sampled from the source video using Qwen’s vision utilities (fps-based sampling and resize).
2. **Encoding** — Sampled frames were encoded as JPEGs and sent to a remote inference API.
3. **Description** — The model produced a single, coherent text description of the full clip from the provided frames.

No manual annotation or editing was applied; the description is model output only.

---

## Models and Approaches


| Component                 | Choice                                                                                                                                           |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Vision–language model** | **Qwen3-VL-8B-Instruct** (Hugging Face Inference Endpoint)                                                                                       |
| **Frame sampling**        | **qwen-vl-utils** (`process_vision_info`) — same sampling/resize logic as official Qwen-VL (fps, patch size, token budget)                       |
| **Video decoding**        | **decord**                                                                                                                                       |
| **API**                   | Hugging Face Inference Endpoint — OpenAI-compatible `/v1/chat/completions` with multimodal messages (multiple `image_url` entries + text prompt) |


**Run configuration (this run):**

- Backend: `qwen-vl-utils`
- Sampling: **2 fps**, up to **120 frames**
- Prompt: *"Describe the video."*

---

## Results

### Timing (this run)


| Stage                               | Time       |
| ----------------------------------- | ---------- |
| qwen-vl-utils `process_vision_info` | 244.46 s   |
| Encode 120 frames to JPEG           | 3.30 s     |
| Video processing (total)            | 247.96 s   |
| API request                         | 43.24 s    |
| **Total (wall)**                    | **~291 s** |


### Model Output (verbatim)

The video appears to be a cinematic trailer or opening sequence for a film or production titled **"KILAVUZ"** (which translates to "Guide" or "Navigator" in English).

The sequence begins with a stylized, abstract animation featuring glowing particles that coalesce into a logo for **"ZEMBEREK,"** which then fades into a black screen. This is followed by the official seal of the **"T.C. Kültür ve Turizm Bakanlığı"** (Ministry of Culture and Tourism of the Republic of Turkey), indicating state support for the project. The sequence concludes with the **"TRT SINEMA"** logo, signifying a co-production with TRT (Turkish Radio and Television Corporation).

The main narrative portion of the video depicts a tense, atmospheric scene inside a car. Two men are in the front seats, driving along a desolate, rural road through a dry, golden field under a bright blue sky with scattered clouds. The driver is focused on the road, while the passenger, wearing white earbuds, interacts with the car's infotainment system. The rearview mirror reflects the driver's intense, focused expression. The journey is interrupted when the car approaches a roadblock with warning signs, causing the driver to slow down and the passenger to look at him with concern.

The scene then cuts to a rear view of a third man, an older gentleman with a white beard, sitting in the back seat, looking out the window at the passing landscape, which includes wind turbines.

The video concludes with a powerful, symbolic shot: a lone man in a suit is seen running across a vast, harvested field, heading towards a distant hill under the same expansive sky. This imagery strongly suggests themes of pursuit, escape, or a journey toward an uncertain destination.

The final frame reveals the film's title, **"KILAVUZ,"** displayed in bold white letters against a black background, accompanied by a stylized arrow logo. The overall tone is suspenseful and cinematic, hinting at a thriller or drama set in a rural, Turkish landscape.

## Summary

- **Input:** 60 s video (trailer / opening for “KILAVUZ”).
- **Method:** 120 frames at 2 fps via qwen-vl-utils → Qwen3-VL-8B on Hugging Face Inference Endpoint.
- **Model License:** Apache 2.0
- **Output:** A single, structured text description covering logos, setting, characters, and narrative tone, suitable for indexing or accessibility.

---

### Hugging Face Inference Endpoint Instance Details

- **Provider:** AWS
- **Region:** us-east-1
- **Hardware:** 1 × Nvidia A100 GPU (80 GB VRAM)

## Model Selection

The Qwen3-VL-8B model was selected for this task due to its strong multimodal reasoning capabilities and proven performance on visual-linguistic understanding benchmarks. Video frames were processed using available tools (`qwen-vl-utils`, with fallback to `opencv`), ensuring high-quality and consistent extraction of representative images from the video.

Key considerations for model and tool selection:

- **Capability:** Qwen3-VL-8B supports both image and video understanding. It is adept at generating detailed, structured textual descriptions from complex visual input, making it suitable for long-form, high-fidelity video summarization.
- **Video Frame Processing:** The pipeline uses `qwen-vl-utils` to extract up to 120 frames at 2 fps, maintaining coverage and temporal diversity.
- **License:** The model is distributed under the Apache 2.0 License, which permits free use in commercial applications.

---

### Approaches for Handling Longer Videos and Improving Output

1. **Evaluate Model Performance on Longer Inputs**
  - Systematically test the vision–language pipeline with longer video content (e.g., multi-minute clips or feature-length material).
  - Identify any degradation in accuracy, coherence, or descriptive ability as input length increases.
  - Consider metrics such as segmentation quality, recall of temporal events, and completeness of narrative coverage.
2. **Post-process Descriptions for Length, Structure, and Style**
  - Feed initial raw descriptions through a secondary, specialized language model:
    - **Length control:** Adjust output to a target word or sentence count (e.g., generate summaries of different lengths optimized for catalogs, accessibility, or social media).
    - **Style alignment:** Standardize tone and phrasing, or tailor to a specific audience (e.g., educational, cinematic, accessibility-focused).
    - **Segmentation:** Explore splitting video into logical scenes or intervals, describing each separately; aggregate or join as needed for the desired output format.
  - **Open question:** Should description granularity adapt to the video's duration (e.g., 1 summary per N seconds or per scene)?
3. **Support for Multilingual Output**
  - Automatically translate finalized descriptions into Turkish (or other target languages) using high-quality machine translation models.
  - Consider augmenting with post-editing by human translators for improved accuracy in key deployments.
  - Support bidirectional translation to enable both Turkish and English (and, potentially, additional languages).
4. **Additional Ideas**
  - **Interactive/iterative editing:** Allow a human reviewer to quickly revise the model’s output via a lightweight interface before finalization.
  - **Rich metadata output:** Complement textual descriptions with timestamps, scene boundaries, or entity tags for easier downstream use.
  - **Summarization at multiple levels:** Provide both short and detailed descriptions per user/application needs.
  - **Feedback loop:** Use human ratings on summaries to fine-tune the model or post-processing pipeline over time.

---

