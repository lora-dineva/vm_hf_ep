# vm_hf_ep

Get a text description of a video using Qwen3-VL on a Hugging Face Inference Endpoint.

**Setup**

```bash
pip install -r requirements.txt
```

Create a `.env` file with `HF_TOKEN=your_hf_token` (from [Hugging Face settings](https://huggingface.co/settings/tokens)).

**Run**

```bash
python describe_video.py video.mp4
python describe_video.py video.mp4 --backend qwen-vl-utils --fps 2 --max-frames 120
python describe_video.py video.mp4 --extract-frames-only
```
