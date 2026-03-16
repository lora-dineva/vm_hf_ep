# Video Description PoC

## What Was Done

The same 60-second source video was first split into **13 scenes** (via `split_scenes.py`). Each scene clip was then described independently by the vision–language model. The pipeline:

1. **Scene input** — Scene clips (e.g. `test_clip_60s-Scene-001.mp4` … `Scene-013.mp4`) from the `scenes` directory.
2. **Video processing** — For each clip, frames were sampled using Qwen’s vision utilities (fps-based sampling and resize).
3. **Encoding** — Sampled frames were encoded as JPEGs and sent to the remote inference API.
4. **Description** — The model produced one text description per scene; descriptions were written into `scenes/test_clip_60s_scenes.csv` in a `description` column.

No manual annotation or editing was applied; all descriptions are model output only.

---

## Models and Approaches


| Component                 | Choice                                                                                                                                           |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Vision–language model** | **Qwen/Qwen3-VL-8B-Instruct** (Hugging Face Inference Endpoint)                                                                                  |
| **Frame sampling**        | **qwen-vl-utils** (`process_vision_info`) — same sampling/resize logic as official Qwen-VL (fps, patch size, token budget)                       |
| **Video decoding**        | **decord**                                                                                                                                       |
| **API**                   | Hugging Face Inference Endpoint — OpenAI-compatible `/v1/chat/completions` with multimodal messages (multiple `image_url` entries + text prompt) |


**Run configuration (this run):**

- Backend: `qwen-vl-utils`
- Sampling: **2 fps**, up to **120 frames** per scene
- Prompt: *"Describe the video."* (fallback from config; Langfuse prompt used if configured)

### Scene splitter

Scene boundaries were computed and the source video was split **before** description using `split_scenes.py`:

| Aspect            | Choice                                                                 |
| ----------------- | ---------------------------------------------------------------------- |
| **Library**       | **PySceneDetect** — scene detection and boundary list                  |
| **Splitting**     | **ffmpeg** — system binary if on PATH, otherwise **imageio-ffmpeg**    |
| **Detector**      | **Adaptive** (default); alternatives: `content`, `threshold`            |
| **Output CSV**    | `{video_stem}_scenes.csv` in the output dir — columns: `scene_id`, `start_time`, `end_time` |
| **Output clips**  | One MP4 per scene: `$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4` (e.g. `test_clip_60s-Scene-001.mp4`) |

For this run, the 60 s source was split into **13 scenes** with the default **adaptive** detector. The resulting CSV (`scenes/test_clip_60s_scenes.csv`) was later updated by `describe_video.py` with a `description` column. Use `python split_scenes.py --list-only` to print scene boundaries without writing files.

---

## Results

### Timing (this run)

13 scene clips were processed sequentially. Totals below are summed over all 13 scenes (from terminal log).

| Stage                               | Time (total) | Per-scene range   |
| ----------------------------------- | ------------ | ----------------- |
| qwen-vl-utils `process_vision_info` | 26.44 s      | 0.28 s – 5.06 s   |
| Encode frames to JPEG               | 27.50 s      | 0.32 s – 5.30 s   |
| Video processing (total)            | ~54 s        | 1.92 s – 9.67 s   |
| API request                         | 35.71 s      | 1.58 s – 4.35 s   |
| **Total (wall)**                    | **~63.6 s**  | 2.83 s – 9.67 s   |


### Model Output (verbatim)

Per-scene descriptions from `scenes/test_clip_60s_scenes.csv`:

**Scene 1** (00:00:00.000 – 00:00:00.440)  
The screen remains completely black throughout the entire sequence. There are no visual elements, actions, or settings to describe. The scene is silent and devoid of any imagery.

**Scene 2** (00:00:00.440 – 00:00:00.700)  
In a vast, star-strewn void, two luminous streams—one glowing amber, the other a vibrant green—flow with graceful, opposing arcs. They swirl and intertwine at their center, a dazzling dance of light, while countless tiny sparks drift like cosmic dust around them, painting a scene of serene, celestial motion.

**Scene 3** (00:00:00.700 – 00:00:01.700)  
A cosmic swirl of golden and green light emerges from the darkness, forming the name "ZEMBEREK" as if forged from starlight. The swirling energy coalesces into a radiant, concentric circular logo, glowing warmly against the star-speckled void. The name settles into place, a beacon of elegance and mystery, signaling the beginning of a cinematic journey.

**Scene 4** (00:00:01.700 – 00:00:14.600)  
The video opens with a cosmic, star-filled scene where the name "ZEMBEREK" emerges from a glowing, swirling vortex of orange and yellow light, evoking a sense of mystery and grandeur. This fades into a stark black screen, which then reveals the official emblem of the Turkish Ministry of Culture and Tourism — a red, circular seal with a traditional knot design and stars — slowly illuminating and pulsing with energy. Below it, the text "T.C. Kültür ve Turizm Bakanlığı tarafından desteklenmiştir" (Supported by the Ministry of Culture and Tourism) appears, grounding the production in national cultural support. The scene transitions to the TRT Sinema logo, featuring a film clapperboard, accompanied by the text "TRT Ortaklığıyla / In Coproduction with," indicating a collaborative cinematic effort. The video concludes with a final black screen, leaving a sense of anticipation and artistic professionalism.

**Scene 5** (00:00:14.600 – 00:00:24.280)  
Two men are driving along a quiet, rural road under a bright, partly cloudy sky. The driver, focused on the road, occasionally glances at the passenger. The passenger, wearing earbuds, is engaged with the car's infotainment system, tapping and swiping its screen. Their journey through the open countryside is calm and contemplative, marked by the steady hum of the engine and the passing scenery.

**Scene 6** (00:00:24.280 – 00:00:28.600)  
A man with a gray beard sits in the back of a car, gazing out the window. Outside, a lone wind turbine stands in a vast, golden field under a bright blue sky with scattered clouds. As the car moves forward, the landscape drifts past, revealing more trees and the distant horizon. He appears contemplative, lost in thought as he observes the serene, open countryside.

**Scene 7** (00:00:28.600 – 00:00:31.160)  
A man drives a car along a cracked, lonely road through a vast, sun-baked field. His passenger sits quietly beside him. In the rearview mirror, the driver's focused face is reflected, hinting at a journey of quiet contemplation or purpose. The open sky and empty landscape suggest solitude, stretching ahead into the distance.

**Scene 8** (00:00:31.160 – 00:00:33.720)  
A man in a suit sits in the back of a moving car, gazing out the window at a vast, sunlit landscape. His expression is contemplative as the scenery blurs past, suggesting a journey through open country under a bright, cloud-dappled sky.

**Scene 9** (00:00:33.720 – 00:00:42.920)  
Two men are driving a car along a quiet, rural road under a bright, sunny sky. The driver, focused ahead, steers steadily as the passenger, wearing earbuds, looks out the window. As they approach a roadblock with warning signs, the driver slows down and glances toward the passenger, their expressions turning serious and alert. The scene suggests they've encountered an unexpected obstacle, and a moment of tense, silent communication unfolds between them.

**Scene 10** (00:00:42.920 – 00:00:48.000)  
The screen begins in darkness. A sleek, white arrow icon emerges, pointing right. Letters appear one by one, forming the word "KILAVUZ." The final letter "Z" completes the name, and the arrow icon slides smoothly to the right, leaving the word "KILAVUZ" centered on the black screen. The logo fades out, leaving the screen black once more.

**Scene 11** (00:00:48.000 – 00:00:53.280)  
A lone figure walks steadily uphill across a vast, sun-baked field of dry, golden stubble. The path they follow is marked by deep tire tracks winding toward the crest of the hill. Above, a wide, clear blue sky is dotted with scattered, fluffy white clouds. The scene is quiet and expansive, evoking a sense of solitude, determination, and peaceful endurance as the person moves toward the distant horizon.

**Scene 12** (00:00:53.280 – 00:00:59.080)  
A man in a suit, holding his jacket, runs determinedly across a vast, golden field of harvested wheat. His focused expression and steady pace suggest urgency or purpose as he moves toward the camera, the expansive, textured field stretching out behind him under a clear sky.

**Scene 13** (00:00:59.080 – 00:01:00.000)  
A lone road winds through endless golden fields under a vast, clear blue sky. The camera glides forward, following the path as it curves gently into the horizon, flanked by rows of sunflowers and dry grass. The scene is peaceful, quiet, and expansive — a solitary journey through the open countryside.

## Summary

- **Input:** 13 scene clips from the 60 s source video (trailer / opening for "KILAVUZ"), from `split_scenes.py`.
- **Method:** Per-scene: frames at 2 fps (up to 120 per clip) via qwen-vl-utils → Qwen3-VL-8B on Hugging Face Inference Endpoint; descriptions written to CSV.
- **Model License:** Apache 2.0
- **Output:** One description per scene in `scenes/test_clip_60s_scenes.csv` (columns: scene_id, start_time, end_time, description), suitable for scene-level indexing or accessibility.

---

### Hugging Face Inference Endpoint Instance Details

- **Provider:** AWS
- **Region:** us-east-1
- **Hardware:** 1 × Nvidia A100 GPU (80 GB VRAM)

## Model Selection

The Qwen3-VL-8B model was selected for this task due to its strong multimodal reasoning capabilities and proven performance on visual-linguistic understanding benchmarks. Video frames were processed using available tools (`qwen-vl-utils`, with fallback to `opencv`), ensuring high-quality and consistent extraction of representative images from the video.

Key considerations for model and tool selection:

- **Capability:** Qwen3-VL-8B supports both image and video understanding. It is adept at generating detailed, structured textual descriptions from complex visual input, making it suitable for long-form, high-fidelity video summarization.
- **Video Frame Processing:** The pipeline uses `qwen-vl-utils` to extract up to 120 frames at 2 fps per clip, maintaining coverage and temporal diversity.
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
   - **Interactive/iterative editing:** Allow a human reviewer to quickly revise the model's output via a lightweight interface before finalization.
   - **Rich metadata output:** Complement textual descriptions with timestamps, scene boundaries, or entity tags for easier downstream use.
   - **Summarization at multiple levels:** Provide both short and detailed descriptions per user/application needs.
   - **Feedback loop:** Use human ratings on summaries to fine-tune the model or post-processing pipeline over time.

---

