"""RGB JPEG decode and h264 encode for RoboTwin → LeRobot."""

from __future__ import annotations

from pathlib import Path

import av
import cv2
import numpy as np


def decode_jpeg_rgb(raw_bytes) -> np.ndarray:
    """Decode a HDF5 ``observation/*/rgb`` JPEG to a correct-RGB HxWx3 array.

    RoboTwin stores SAPIEN RGB pixels via ``cv2.imencode`` without a channel
    swap. ``cv2.imdecode(..., IMREAD_COLOR)`` therefore returns an array whose
    channel order is already true RGB. Do **not** colour-convert the result.
    """
    img = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode JPEG frame")
    return img


def encode_video_rgb(frames_rgb, out_path: Path, fps: float) -> None:
    """Encode correct-RGB uint8 HxWx3 frames to h264/yuv420p mp4 (no cvtColor)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(out_path), mode="w")
    height, width = frames_rgb[0].shape[:2]
    stream = container.add_stream("h264", rate=round(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    for img_rgb in frames_rgb:
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(img_rgb), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
