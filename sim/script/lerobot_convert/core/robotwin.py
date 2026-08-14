"""Load + write one RoboTwin HDF5 episode (shared by rollouts / scenes)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from . import paths, resample, schema, video


def peek_length(hdf5_path: Path, fail_max_length: int | None = None) -> int:
    """Cheaply read the output frame count without decoding images.

    Conversion is 1:1 with the raw rows, so this is the raw length unless
    ``fail_max_length`` caps an unsuccessful episode.
    """
    with h5py.File(hdf5_path, "r") as f:
        old_len = f["collision"].shape[0]
        success = bool(f.attrs.get("success", False))
    new_len, _, _ = resample.resample_windows(
        old_len, resample.SOURCE_FPS, resample.TARGET_FPS
    )
    if fail_max_length is not None and not success:
        new_len = min(new_len, int(fail_max_length))
    return new_len


def load_robotwin_episode(
    hdf5_path: Path,
    *,
    fail_max_length: int | None = None,
) -> dict:
    """Decode one HDF5 into per-frame tensors + RGB frames (1:1 with raw rows).

    ``state[t]`` is the drive target recorded before command ``t`` executes and
    ``action[t] = state[t+1]`` is the target command ``t`` left behind — i.e.
    the model's action for step ``t``.

    Returns a dict with: new_len, state, action, collision_*, contact_*,
    cam_data, success, generator, seed.
    """
    with h5py.File(hdf5_path, "r") as f:
        old_len = f["collision"].shape[0]
        new_len, windows, reps = resample.resample_windows(
            old_len, resample.SOURCE_FPS, resample.TARGET_FPS
        )

        vector = f["joint_action/vector"][:]
        resampled_vector = vector[reps]
        state = resampled_vector
        action = np.empty_like(resampled_vector)
        action[:-1] = resampled_vector[1:]
        action[-1] = resampled_vector[-1]

        collision = f["collision"][:]
        collision_impulse = f["collision_impulse"][:]
        collision_pairs_raw = f["collision_pairs"][:]
        contact = f["contact"][:]
        contact_impulse = f["contact_impulse"][:]
        contact_pairs_raw = f["contact_pairs"][:]

        collision_r = np.array([collision[s:e].max() for s, e in windows], dtype=np.uint8)
        collision_impulse_r = np.array(
            [collision_impulse[s:e].max() for s, e in windows], dtype=np.float32
        )
        collision_pairs_r = [
            schema.union_pairs(collision_pairs_raw, s, e) for s, e in windows
        ]
        contact_r = np.array([contact[s:e].max() for s, e in windows], dtype=np.uint8)
        contact_impulse_r = np.array(
            [contact_impulse[s:e].max() for s, e in windows], dtype=np.float32
        )
        contact_pairs_r = [
            schema.union_pairs(contact_pairs_raw, s, e) for s, e in windows
        ]

        cam_data = {}
        for hdf5_cam, cam_name in schema.CAMERA_MAP.items():
            rgb_ds = f[f"observation/{hdf5_cam}/rgb"]
            extr_ds = f[f"observation/{hdf5_cam}/extrinsic_cv"]
            intr_ds = f[f"observation/{hdf5_cam}/intrinsic_cv"]
            frames = [video.decode_jpeg_rgb(rgb_ds[i]) for i in reps]
            extr = np.stack([schema.flatten_cv(extr_ds[i]) for i in reps])
            intr = np.stack([schema.flatten_cv(intr_ds[i]) for i in reps])
            cam_data[cam_name] = {"frames": frames, "extrinsic": extr, "intrinsic": intr}

        success = bool(f.attrs.get("success", False))
        generator = str(f.attrs.get("generator", ""))
        seed = int(f.attrs.get("seed", -1))

    if fail_max_length is not None and not success and new_len > fail_max_length:
        new_len = int(fail_max_length)
        state = state[:new_len]
        action = action[:new_len]
        action[-1] = state[-1]
        collision_r = collision_r[:new_len]
        collision_impulse_r = collision_impulse_r[:new_len]
        collision_pairs_r = collision_pairs_r[:new_len]
        contact_r = contact_r[:new_len]
        contact_impulse_r = contact_impulse_r[:new_len]
        contact_pairs_r = contact_pairs_r[:new_len]
        for cam_name in cam_data:
            cam_data[cam_name]["frames"] = cam_data[cam_name]["frames"][:new_len]
            cam_data[cam_name]["extrinsic"] = cam_data[cam_name]["extrinsic"][:new_len]
            cam_data[cam_name]["intrinsic"] = cam_data[cam_name]["intrinsic"][:new_len]

    return {
        "new_len": new_len,
        "state": state,
        "action": action,
        "collision_r": collision_r,
        "collision_impulse_r": collision_impulse_r,
        "collision_pairs_r": collision_pairs_r,
        "contact_r": contact_r,
        "contact_impulse_r": contact_impulse_r,
        "contact_pairs_r": contact_pairs_r,
        "cam_data": cam_data,
        "success": success,
        "generator": generator,
        "seed": seed,
    }


def build_episode_dataframe(
    ep: dict,
    *,
    episode_index: int,
    task_index: int,
    global_index_start: int,
    data_source: str,
    source_detail: str,
    extra_columns: dict | None = None,
    include_clean_success: bool = False,
    clean_success: bool = False,
    column_order: list[str] | None = None,
) -> pd.DataFrame:
    """Build the per-frame parquet table from a loaded episode dict."""
    new_len = ep["new_len"]
    frame_index = np.arange(new_len, dtype=np.int64)
    timestamp = frame_index / resample.TARGET_FPS
    global_index = global_index_start + frame_index

    data = {
        "action": list(ep["action"].astype(np.float32)),
        "observation.state": list(ep["state"].astype(np.float32)),
        "task_index": np.full(new_len, task_index, dtype=np.int64),
        "episode_index": np.full(new_len, episode_index, dtype=np.int64),
        "frame_index": frame_index,
        "timestamp": timestamp.astype(np.float64),
        "index": global_index,
    }
    for cam_name, d in ep["cam_data"].items():
        data[f"observation.{cam_name}.extrinsic_cv"] = list(d["extrinsic"])
        data[f"observation.{cam_name}.intrinsic_cv"] = list(d["intrinsic"])

    data["collision"] = pd.array(ep["collision_r"], dtype="UInt8")
    data["collision_impulse"] = pd.array(ep["collision_impulse_r"], dtype="Float32")
    data["collision_pairs"] = ep["collision_pairs_r"]
    data["contact"] = pd.array(ep["contact_r"], dtype="UInt8")
    data["contact_impulse"] = pd.array(ep["contact_impulse_r"], dtype="Float32")
    data["contact_pairs"] = ep["contact_pairs_r"]
    data["success"] = np.full(new_len, ep["success"], dtype=bool)
    if include_clean_success:
        data["clean_success"] = np.full(new_len, clean_success, dtype=bool)
    data["data_source"] = [data_source] * new_len
    data["source_detail"] = [source_detail] * new_len

    if extra_columns:
        for key, values in extra_columns.items():
            data[key] = values

    df = pd.DataFrame(data)
    if column_order is not None:
        have, want = set(df.columns), set(column_order)
        if have != want:
            raise ValueError(
                f"column set mismatch: missing={want - have} extra={have - want}"
            )
        return df[column_order]
    return schema.canonical_columns(df, include_clean_success=include_clean_success)


def write_episode_videos(cam_data: dict, out_root: Path, episode_index: int) -> None:
    for cam_name, d in cam_data.items():
        out_video = paths.video_path(out_root, episode_index, cam_name)
        video.encode_video_rgb(d["frames"], out_video, resample.TARGET_FPS)


def convert_robotwin_episode(
    hdf5_path: Path,
    *,
    episode_index: int,
    task_index: int,
    global_index_start: int,
    out_root: Path,
    data_source: str,
    source_detail: str,
    fail_max_length: int | None = None,
    extra_columns: dict | None = None,
    include_clean_success: bool = False,
    clean_success: bool = False,
    column_order: list[str] | None = None,
) -> dict:
    """Load HDF5 → write parquet + RGB videos; return episode record fields."""
    ep = load_robotwin_episode(hdf5_path, fail_max_length=fail_max_length)
    df = build_episode_dataframe(
        ep,
        episode_index=episode_index,
        task_index=task_index,
        global_index_start=global_index_start,
        data_source=data_source,
        source_detail=source_detail,
        extra_columns=extra_columns,
        include_clean_success=include_clean_success,
        clean_success=clean_success,
        column_order=column_order,
    )

    out_parquet = paths.data_path(out_root, episode_index)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    write_episode_videos(ep["cam_data"], out_root, episode_index)

    return {
        "episode_index": episode_index,
        "length": ep["new_len"],
        "success": ep["success"],
        "seed": ep["seed"],
        "generator": ep["generator"],
        "source_detail": source_detail,
        "df": df,
    }
