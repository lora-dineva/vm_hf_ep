#!/usr/bin/env python3
"""
Send a video file to a Hugging Face Inference Endpoint (Qwen3-VL) and print
the model's description of what is happening in the clip.

Video processing can use qwen-vl-utils (Qwen-style fps/resize) or opencv
(single middle frame). Use --backend qwen-vl-utils when installed.
"""

import argparse
import base64
import io
import os
import subprocess
import sys
import time
from typing import List

LOG_PREFIX = "[describe_video]"

import cv2
import requests
from dotenv import load_dotenv

load_dotenv()

# Optional: qwen-vl-utils for Qwen-style video sampling (fps, resize)
try:
    from qwen_vl_utils import process_vision_info
    _QWEN_VL_UTILS_AVAILABLE = True
except ImportError:
    process_vision_info = None
    _QWEN_VL_UTILS_AVAILABLE = False

DEFAULT_VIDEO = "test_clip_60s.mp4"
DEFAULT_ENDPOINT = "https://rjk11aiy6oefykan.us-east-1.aws.endpoints.huggingface.cloud"
CHAT_PATH = "/v1/chat/completions"
PROMPT = """
Describe the video.
"""

def get_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("Error: Set HF_TOKEN in .env or HUGGING_FACE_HUB_TOKEN.", file=sys.stderr)
        sys.exit(1)
    return token


def extract_frames_qwen_vl_utils(
    path: str,
    fps: float = 2.0,
    max_frames: int = 120,
    image_patch_size: int = 16,
) -> List[str]:
    """
    Use qwen-vl-utils to load and sample the video (fps, resize), then return
    a list of base64 JPEG strings for each frame. Compatible with Qwen3-VL sampling.
    """
    if not _QWEN_VL_UTILS_AVAILABLE or process_vision_info is None:
        raise RuntimeError("qwen-vl-utils is not installed")
    if not os.path.isfile(path):
        print(f"Error: Video file not found: {path}", file=sys.stderr)
        sys.exit(1)
    abs_path = os.path.abspath(path)
    # qwen-vl-utils does video_path[7:] for "file://", so we must not use "file:///C:/..."
    # or it becomes "/C:/..." on Windows. Use "file://" + path so remainder is C:/... or /...
    path_for_url = abs_path.replace("\\", "/")
    file_url = "file://" + path_for_url
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": file_url,
                        "fps": fps,
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
    ]
    t0 = time.perf_counter()
    try:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
            return_video_metadata=True,
            image_patch_size=image_patch_size,
        )
    except Exception as e:
        print(f"Error: qwen-vl-utils failed: {e}", file=sys.stderr)
        sys.exit(1)
    if not video_inputs or len(video_inputs) == 0:
        print("Error: No video output from process_vision_info.", file=sys.stderr)
        sys.exit(1)
    video_item = video_inputs[0]
    if isinstance(video_item, tuple):
        video_tensor = video_item[0]
    else:
        video_tensor = video_item
    # Shape (T, C, H, W), float in [0, 1] after resize
    try:
        from PIL import Image as PILImage
    except ImportError as e:
        print(f"Error: qwen-vl-utils path needs Pillow: {e}", file=sys.stderr)
        sys.exit(1)
    t_process = time.perf_counter() - t0
    print(f"{LOG_PREFIX} qwen-vl-utils process_vision_info: {t_process:.2f}s", file=sys.stderr)
    T = video_tensor.shape[0]
    frames_b64: List[str] = []
    t_encode = time.perf_counter()
    for i in range(min(T, max_frames)):
        frame = video_tensor[i]  # (C, H, W)
        arr = frame.permute(1, 2, 0).cpu().numpy()
        # qwen-vl-utils returns float in [0, 255] (from torchvision/decord/torchcodec), not [0, 1]
        arr = arr.clip(0, 255).astype("uint8")
        buf = io.BytesIO()
        PILImage.fromarray(arr).save(buf, format="JPEG", quality=85)
        frames_b64.append(base64.standard_b64encode(buf.getvalue()).decode("ascii"))
    print(f"{LOG_PREFIX} encode {len(frames_b64)} frames to JPEG: {time.perf_counter() - t_encode:.2f}s", file=sys.stderr)
    return frames_b64


def extract_frames_qwen_vl_utils_to_dir(
    path: str,
    output_dir: str = "frames",
    fps: float = 2.0,
    max_frames: int = 120,
    image_patch_size: int = 16,
    open_folder: bool = True,
) -> List[str]:
    """
    Use qwen-vl-utils to load/sample the video and save each frame as a JPEG file.
    Returns the list of saved file paths. Optionally opens the output folder.
    """
    if not _QWEN_VL_UTILS_AVAILABLE or process_vision_info is None:
        raise RuntimeError("qwen-vl-utils is not installed")
    if not os.path.isfile(path):
        print(f"Error: Video file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        from PIL import Image as PILImage
    except ImportError as e:
        print(f"Error: Pillow required: {e}", file=sys.stderr)
        sys.exit(1)
    abs_path = os.path.abspath(path)
    path_for_url = abs_path.replace("\\", "/")
    file_url = "file://" + path_for_url
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": file_url, "fps": fps},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
    ]
    t0 = time.perf_counter()
    try:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
            return_video_metadata=True,
            image_patch_size=image_patch_size,
        )
    except Exception as e:
        print(f"Error: qwen-vl-utils failed: {e}", file=sys.stderr)
        sys.exit(1)
    if not video_inputs or len(video_inputs) == 0:
        print("Error: No video output from process_vision_info.", file=sys.stderr)
        sys.exit(1)
    video_item = video_inputs[0]
    video_tensor = video_item[0] if isinstance(video_item, tuple) else video_item
    T = video_tensor.shape[0]
    os.makedirs(output_dir, exist_ok=True)
    saved_paths: List[str] = []
    n_save = min(T, max_frames)
    for i in range(n_save):
        frame = video_tensor[i]
        arr = frame.permute(1, 2, 0).cpu().numpy()
        # qwen-vl-utils returns float in [0, 255], not [0, 1]; do not multiply by 255
        arr = arr.clip(0, 255).astype("uint8")
        out_path = os.path.join(output_dir, f"frame_{i:03d}.jpg")
        PILImage.fromarray(arr).save(out_path, format="JPEG", quality=85)
        saved_paths.append(out_path)
    print(f"{LOG_PREFIX} saved {len(saved_paths)} frames to {os.path.abspath(output_dir)} in {time.perf_counter() - t0:.2f}s", file=sys.stderr)
    if open_folder:
        abs_out = os.path.abspath(output_dir)
        if sys.platform == "win32":
            os.startfile(abs_out)
        elif sys.platform == "darwin":
            subprocess.run(["open", abs_out], check=False)
        else:
            subprocess.run(["xdg-open", abs_out], check=False)
    return saved_paths


def extract_frame_base64(path: str, frame_index: int | None = None) -> str:
    """Analyze the video and extract all significant events. Output the results strictly as a CSV with the following headers: Timestamp_Start, Timestamp_End, Event_Description, Confidence_Score.
    """
    t0 = time.perf_counter()
    if not os.path.isfile(path):
        print(f"Error: Video file not found: {path}", file=sys.stderr)
        sys.exit(1)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1
        idx = frame_index if frame_index is not None else total_frames // 2
        idx = min(max(0, idx), total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            print("Error: Could not read any frame from video.", file=sys.stderr)
            sys.exit(1)
        _, buf = cv2.imencode(".jpg", frame)
        out = base64.standard_b64encode(buf.tobytes()).decode("ascii")
        print(f"{LOG_PREFIX} opencv extract 1 frame: {time.perf_counter() - t0:.2f}s", file=sys.stderr)
        return out
    finally:
        cap.release()


def describe_video(
    video_path: str,
    base_url: str,
    token: str,
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    max_tokens: int = 1024,
    backend: str = "opencv",
    video_fps: float = 2.0,
    max_frames: int = 120,
) -> str:
    url = base_url.rstrip("/") + CHAT_PATH
    # Endpoint expects image data; send frame(s) as JPEG (one or multiple).
    t_total = time.perf_counter()
    content: List[dict] = []
    t_video = time.perf_counter()
    if backend == "qwen-vl-utils" and _QWEN_VL_UTILS_AVAILABLE:
        try:
            frames_b64 = extract_frames_qwen_vl_utils(
                video_path, fps=video_fps, max_frames=max_frames
            )
            for b64 in frames_b64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
        except RuntimeError:
            backend = "opencv"
    if backend == "opencv" or not content:
        frame_b64 = extract_frame_base64(video_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
        })
    content.append({"type": "text", "text": PROMPT})
    print(f"{LOG_PREFIX} video processing ({backend}): {time.perf_counter() - t_video:.2f}s, {len(content) - 1} image(s)", file=sys.stderr)

    messages = [{"role": "user", "content": content}]

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    t_api = time.perf_counter()
    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    print(f"{LOG_PREFIX} API request: {time.perf_counter() - t_api:.2f}s", file=sys.stderr)

    if not resp.ok:
        print(f"Error: Request failed with status {resp.status_code}", file=sys.stderr)
        try:
            body = resp.json()
            print(body, file=sys.stderr)
        except Exception:
            print(resp.text[:500], file=sys.stderr)
        sys.exit(1)

    data = resp.json()
    if "choices" not in data or not data["choices"]:
        print("Error: Unexpected response shape (no choices).", file=sys.stderr)
        print(data, file=sys.stderr)
        sys.exit(1)

    message = data["choices"][0].get("message")
    if not message or "content" not in message:
        print("Error: No message content in response.", file=sys.stderr)
        print(data, file=sys.stderr)
        sys.exit(1)

    print(f"{LOG_PREFIX} total: {time.perf_counter() - t_total:.2f}s", file=sys.stderr)
    return message["content"].strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Get a description of a video clip from a Hugging Face VL endpoint."
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=DEFAULT_VIDEO,
        help=f"Path to video file (default: {DEFAULT_VIDEO})",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"Endpoint base URL (default: {DEFAULT_ENDPOINT})",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Model name for the request",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max tokens to generate (default: 1024)",
    )
    parser.add_argument(
        "--backend",
        choices=("opencv", "qwen-vl-utils"),
        default="opencv",
        help="Video processing: opencv (single middle frame) or qwen-vl-utils (fps sampling, requires qwen-vl-utils)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Target FPS for frame sampling when using --backend qwen-vl-utils (default: 2.0)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=120,
        help="Max frames when using qwen-vl-utils (default: 120)",
    )
    parser.add_argument(
        "--extract-frames-only",
        action="store_true",
        help="Only extract frames with qwen-vl-utils, save as JPEGs, and open the folder (no API call)",
    )
    parser.add_argument(
        "--frames-dir",
        default="frames",
        help="Output directory for --extract-frames-only (default: frames)",
    )
    args = parser.parse_args()

    if args.extract_frames_only:
        print(f"{LOG_PREFIX} video={args.video!r} fps={args.fps} max_frames={args.max_frames}", file=sys.stderr)
        paths = extract_frames_qwen_vl_utils_to_dir(
            path=args.video,
            output_dir=args.frames_dir,
            fps=args.fps,
            max_frames=args.max_frames,
            open_folder=True,
        )
        for p in paths:
            print(p)
        return

    token = get_token()
    print(f"{LOG_PREFIX} video={args.video!r} backend={args.backend}", file=sys.stderr)
    t0 = time.perf_counter()
    description = describe_video(
        video_path=args.video,
        base_url=args.endpoint,
        token=token,
        model=args.model,
        max_tokens=args.max_tokens,
        backend=args.backend,
        video_fps=args.fps,
        max_frames=args.max_frames,
    )
    print(f"{LOG_PREFIX} wall time: {time.perf_counter() - t0:.2f}s", file=sys.stderr)
    print(description)


if __name__ == "__main__":
    main()
