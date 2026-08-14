"""Frame-selection helpers for RoboTwin → LeRobot conversion.

``TARGET_FPS == SOURCE_FPS``, so conversion is **1:1**: one LeRobot frame per
raw HDF5 row, and one raw row is one policy command (the recorder captures a
frame per ``take_action`` call). Frame index therefore equals command index,
which is what makes fixed-length action chunks line up with the policy's own
chunk boundaries.

Historical note — this used to downsample 30 → 25 fps via ``np.array_split``,
which does NOT interleave: it emits all the oversized groups first. For a
229-row episode that put every dropped frame in the first 76 rows, so an
episode ran at 15 fps for its first ~19.5% and 30 fps afterwards, while
``timestamp`` claimed a uniform 25 fps (up to ~1.3 s of drift). Worse, the
switch frame scaled with episode length, and episode length correlates with the
success label — so any feature read at a fixed frame index picked up a
label-correlated artifact. ``resample_windows`` below now spaces windows
evenly, so that failure mode cannot return even if the two fps ever differ
again.
"""

from __future__ import annotations

import numpy as np

SOURCE_FPS = 30
TARGET_FPS = 30  # 1:1 with the raw recording; see module docstring


def resample_windows(old_len: int, old_fps: float, new_fps: float):
    """Return ``(new_len, windows, reps)``.

    ``windows`` partitions ``range(old_len)`` into ``new_len`` **evenly spaced**
    groups; ``reps`` is the middle source index of each group. When
    ``new_fps == old_fps`` this is the identity: ``windows[i] == (i, i + 1)``
    and ``reps[i] == i``.
    """
    new_len = max(1, round(old_len * new_fps / old_fps))
    new_len = min(new_len, old_len)
    # linspace edges spread the larger groups across the whole episode;
    # np.array_split would front-load every one of them.
    edges = np.linspace(0, old_len, new_len + 1).round().astype(int)
    windows = [(int(edges[i]), int(edges[i + 1])) for i in range(new_len)]
    reps = [(s + e - 1) // 2 for s, e in windows]
    return new_len, windows, reps
