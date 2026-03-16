# vm_hf_ep

Get a text description of a video using Qwen3-VL on a Hugging Face Inference Endpoint. Supports [Langfuse](https://langfuse.com) for prompt management, token tracking, and A/B testing.

**Setup**

```bash
pip install -r requirements.txt
```

- **Config:** Edit `config.yaml` for default video path, endpoint, model, max_tokens, fps, max_frames, prompt name/label, fallback prompt, and optional A/B labels. CLI overrides config.
- **Secrets:** Create a `.env` (see `.env.example`). Required: `HF_TOKEN`. Optional: `HF_ENDPOINT`, and Langfuse keys for tracing and prompt management.

**Run**

```bash
python describe_video.py video.mp4
python describe_video.py video.mp4 --fps 2 --max-frames 120
python describe_video.py video.mp4 --extract-frames-only
python describe_video.py --config path/to/config.yaml video.mp4
```

**Split video into scenes (PySceneDetect):** `python split_scenes.py [video] --output-dir scenes`. Use `--list-only` to only print scene boundaries. Uses system ffmpeg if on PATH, otherwise the imageio-ffmpeg bundled binary.