"""Shared NaN-safe column stats for LeRobot episodes_stats.jsonl."""

from __future__ import annotations

import numpy as np
import pandas as pd


def column_stats(series: pd.Series, *, is_array: bool) -> dict:
    if is_array:
        mat = np.stack([np.asarray(v, dtype=np.float64) for v in series.to_numpy()])
        valid = ~np.isnan(mat).any(axis=1)
        if not valid.any():
            n = mat.shape[1]
            return {
                "min": [None] * n,
                "max": [None] * n,
                "mean": [None] * n,
                "std": [None] * n,
                "count": [0],
            }
        sub = mat[valid]
        return {
            "min": sub.min(axis=0).tolist(),
            "max": sub.max(axis=0).tolist(),
            "mean": sub.mean(axis=0).tolist(),
            "std": sub.std(axis=0).tolist(),
            "count": [int(valid.sum())],
        }
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    valid = ~np.isnan(arr)
    if not valid.any():
        return {"min": [None], "max": [None], "mean": [None], "std": [None], "count": [0]}
    sub = arr[valid]
    return {
        "min": [float(sub.min())],
        "max": [float(sub.max())],
        "mean": [float(sub.mean())],
        "std": [float(sub.std())],
        "count": [int(valid.sum())],
    }


# Back-compat alias used by older call sites.
_column_stats = column_stats
