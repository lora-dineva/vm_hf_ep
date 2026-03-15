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

**Langfuse: prompt and A/B testing**

- Create a prompt named `video-description` in Langfuse (text type, e.g. "Describe the video.") and assign a label (e.g. `production`). The script fetches it by default; without Langfuse env it uses a built-in fallback.
- **Prompt A/B:** Create two versions with labels e.g. `prod-a` and `prod-b`, then run with `--ab-prompt-labels "prod-a,prod-b"` to randomly use one per run; compare metrics in Langfuse by prompt version.
- **Model A/B:** Use `--model-b <model_name>` and `--ab-test-models` to randomly call one of two models; filter traces by metadata `ab_model` (a or b) in Langfuse to compare cost and performance.
