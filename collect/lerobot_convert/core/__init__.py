"""Core convert utilities (stable across data layouts).

Keep wrappers (``convert_scenes``, uploads, pipelines)
outside this package. Adapt wrappers for new sources; change ``core`` only when
the RoboTwin episode → parquet/video contract changes.
"""

from .paths import CHUNK_SIZE, chunk_dir, data_path, video_path
from .resample import SOURCE_FPS, TARGET_FPS, resample_windows
from .robotwin import (
    build_episode_dataframe,
    convert_robotwin_episode,
    load_robotwin_episode,
    peek_length,
    write_episode_videos,
)
from .schema import (
    CAMERA_MAP,
    CANONICAL_COLUMNS,
    CANONICAL_COLUMNS_NO_CLEAN,
    EXPERT_DATA_SOURCE,
    ROLLOUT_DATA_SOURCE,
    canonical_columns,
    decode_pairs_bytes,
    flatten_cv,
    union_pairs,
)
from .stats import column_stats
from .video import decode_jpeg_rgb, encode_video_rgb

__all__ = [
    "CAMERA_MAP",
    "CANONICAL_COLUMNS",
    "CANONICAL_COLUMNS_NO_CLEAN",
    "CHUNK_SIZE",
    "EXPERT_DATA_SOURCE",
    "ROLLOUT_DATA_SOURCE",
    "SOURCE_FPS",
    "TARGET_FPS",
    "build_episode_dataframe",
    "canonical_columns",
    "chunk_dir",
    "column_stats",
    "convert_robotwin_episode",
    "data_path",
    "decode_jpeg_rgb",
    "decode_pairs_bytes",
    "encode_video_rgb",
    "flatten_cv",
    "load_robotwin_episode",
    "peek_length",
    "resample_windows",
    "union_pairs",
    "video_path",
    "write_episode_videos",
]
