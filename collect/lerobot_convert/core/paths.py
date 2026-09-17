"""Shared LeRobot path helpers."""

from __future__ import annotations

from pathlib import Path

CHUNK_SIZE = 1000


def chunk_dir(episode_index: int) -> str:
    return f"chunk-{episode_index // CHUNK_SIZE:03d}"


def data_path(root: Path, episode_index: int) -> Path:
    return root / "data" / chunk_dir(episode_index) / f"episode_{episode_index:06d}.parquet"


def video_path(root: Path, episode_index: int, cam_name: str) -> Path:
    return (
        root
        / "videos"
        / chunk_dir(episode_index)
        / f"observation.images.{cam_name}"
        / f"episode_{episode_index:06d}.mp4"
    )
