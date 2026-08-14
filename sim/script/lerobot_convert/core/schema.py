"""RoboTwin → LeRobot column schema and small HDF5 helpers."""

from __future__ import annotations

import json

import numpy as np

# HDF5 camera key -> target LeRobot camera name.
CAMERA_MAP = {
    "countertop_camera": "countertop",
    "left_camera": "left",
    "right_camera": "right",
}

ROLLOUT_DATA_SOURCE = "robopro_rollouts"
EXPERT_DATA_SOURCE = "roboreal_lerobot_expert"

CANONICAL_COLUMNS = [
    "action",
    "observation.state",
    "task_index",
    "episode_index",
    "frame_index",
    "timestamp",
    "index",
    "observation.countertop.extrinsic_cv",
    "observation.countertop.intrinsic_cv",
    "observation.left.extrinsic_cv",
    "observation.left.intrinsic_cv",
    "observation.right.extrinsic_cv",
    "observation.right.intrinsic_cv",
    "collision",
    "collision_impulse",
    "collision_pairs",
    "contact",
    "contact_impulse",
    "contact_pairs",
    "success",
    "clean_success",
    "data_source",
    "source_detail",
]

CANONICAL_COLUMNS_NO_CLEAN = [c for c in CANONICAL_COLUMNS if c != "clean_success"]


def canonical_columns(df, *, include_clean_success: bool = True):
    """Reorder ``df`` to the canonical column order; error if the set differs."""
    want_list = CANONICAL_COLUMNS if include_clean_success else CANONICAL_COLUMNS_NO_CLEAN
    have = set(df.columns)
    want = set(want_list)
    if have != want:
        raise ValueError(
            f"column set mismatch: missing={want - have} extra={have - want}"
        )
    return df[want_list]


def decode_pairs_bytes(raw) -> list:
    if raw is None:
        return []
    txt = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
    txt = txt.strip()
    if not txt:
        return []
    return json.loads(txt)


def union_pairs(pairs_arr, start: int, end: int) -> str:
    merged: set = set()
    for i in range(start, end):
        merged.update(decode_pairs_bytes(pairs_arr[i]))
    return json.dumps(sorted(merged))


def flatten_cv(mat: np.ndarray) -> np.ndarray:
    """Row-major flatten for extrinsic (3,4) / intrinsic (3,3) matrices."""
    return np.ascontiguousarray(mat, dtype=np.float32).ravel(order="C")
