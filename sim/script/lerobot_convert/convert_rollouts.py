"""Convert robopro_rollouts RoboTwin HDF5 episodes into LeRobot (1:1, 30 fps).

Discovery / labeling live here; HDF5 decode + parquet/video write live in
``lerobot_convert.core``.
"""

from __future__ import annotations

from pathlib import Path

from . import common


def list_rollout_episodes(task_dir: Path, *, exclude_expert: bool = True):
    """Return sorted (config_dir_name, hdf5_path) for all rollout episodes of a task.

    Config dirs are discovered dynamically -- every immediate subdir of
    ``task_dir`` that has a ``data/`` folder of ``episode*.hdf5``. Ordering is
    (config_dir, episode number) for stable episode_index assignment.
    """
    items = []
    for config_dir in sorted(p for p in task_dir.iterdir() if (p / "data").is_dir()):
        if exclude_expert and "expert" in config_dir.name.lower():
            continue
        for fp in sorted(
            (config_dir / "data").glob("episode*.hdf5"),
            key=lambda p: int(p.stem.replace("episode", "")),
        ):
            items.append((config_dir.name, fp))
    return items


def peek_length(hdf5_path: Path, fail_max_length: int | None = None) -> int:
    return common.peek_length(hdf5_path, fail_max_length)


def convert_episode(
    hdf5_path: Path,
    rollout_dir_name: str,
    episode_index: int,
    task_index: int,
    global_index_start: int,
    out_root: Path,
    task_name: str = "",
    clean_seed_keys: set | None = None,
    *,
    include_clean_success: bool = False,
    data_source: str | None = None,
    fail_max_length: int | None = None,
):
    """Convert one HDF5 episode; writes parquet + 3 videos under out_root."""
    source_label = data_source if data_source is not None else common.ROLLOUT_DATA_SOURCE
    ep = common.load_robotwin_episode(hdf5_path, fail_max_length=fail_max_length)

    detail = (
        f"{task_name}/{rollout_dir_name}/{ep['generator']}"
        if task_name
        else f"{rollout_dir_name}/{ep['generator']}"
    )

    clean_success = False
    if include_clean_success and clean_seed_keys:
        from .clean_success import density_of

        density = density_of(rollout_dir_name)
        clean_success = (task_name, density, ep["seed"]) in clean_seed_keys

    df = common.build_episode_dataframe(
        ep,
        episode_index=episode_index,
        task_index=task_index,
        global_index_start=global_index_start,
        data_source=source_label,
        source_detail=detail,
        include_clean_success=include_clean_success,
        clean_success=clean_success,
    )
    out_parquet = common.data_path(out_root, episode_index)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    common.write_episode_videos(ep["cam_data"], out_root, episode_index)

    return {
        "episode_index": episode_index,
        "length": ep["new_len"],
        "task_text": None,
        "df": df,
    }
