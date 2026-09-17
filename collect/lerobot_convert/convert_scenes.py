"""Convert a scene-organised RoboTwin HDF5 dataset into ONE flat LeRobot v2.1
dataset that preserves the scene structure via per-episode metadata.

Source layout is two levels deep::

    <root>/<tier>/seed<N>/data/episode<M>.hdf5   # tier in {clean, d6..d15}
    <root>/<tier>/seed<N>/video/episode<M>.mp4   # ignored (single view, native fps)
    <root>/manifest*.json                        # copied verbatim to out/manifests/

A *scene* is a ``(tier, seed)`` folder. Every scene holds a mixture of success
and fail rollouts. The output is a single LeRobot dataset (episodes flattened,
contiguous ``episode_index``) where each episode additionally carries
``scene_index / tier / seed / draw / scene_id`` -- in the parquet AND in
``meta/episodes.jsonl`` -- and ``meta/scenes.jsonl`` lists every scene with its
episode indices and success/fail split. This keeps the scene grouping fully
recoverable while remaining a standard, loader-compatible LeRobot dataset.

Episode decode/encode is in :mod:`lerobot_convert.core`; this module owns scene
discovery, scene metadata columns, and ``meta/scenes.jsonl``.

Usage (from RoboPRO repo root, env with cv2/av/h5py):
    PYTHONPATH=collect python -m lerobot_convert.convert_scenes \
        --src /path/to/<task>_38scene_... \
        --out ./tmp_lerobot_rgb_smoke \
        --task-text "put the milk tea on the shelf" --limit 2 --overwrite
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import common
from .core.stats import column_stats

DATA_SOURCE = "robotwin_rollouts"

# Base schema (from common.CANONICAL_COLUMNS_NO_CLEAN) + scene metadata columns.
SCENE_COLUMNS = ["scene_index", "tier", "seed", "draw", "scene_id"]
CANONICAL_COLUMNS = common.CANONICAL_COLUMNS_NO_CLEAN + SCENE_COLUMNS

_TIER_RE = re.compile(r"^d(\d+)$")


def _tier_sort_key(tier: str) -> tuple[int, int]:
    """``clean`` first, then ``d6, d7, ... d15`` in numeric order."""
    if tier == "clean":
        return (0, 0)
    m = _TIER_RE.match(tier)
    return (1, int(m.group(1))) if m else (2, hash(tier) & 0xFFFF)


def _episode_num(hdf5_path: Path) -> int:
    return int(hdf5_path.stem.replace("episode", ""))


def discover_scenes(src: Path) -> list[dict]:
    """Return scenes in canonical order, each a dict with tier/seed/scene_id and
    a sorted list of (draw, hdf5_path) episodes.

    A scene is any ``<tier>/seed<N>`` dir containing ``data/episode*.hdf5``.
    ``draw`` is the episode number within the scene (episode<M> -> M).
    """
    scenes = []
    for tier_dir in sorted(
        (p for p in src.iterdir() if p.is_dir()),
        key=lambda p: _tier_sort_key(p.name),
    ):
        tier = tier_dir.name
        for seed_dir in sorted(
            (p for p in tier_dir.iterdir() if p.is_dir() and (p / "data").is_dir()),
            key=lambda p: int(p.name.replace("seed", "")),
        ):
            seed = int(seed_dir.name.replace("seed", ""))
            episodes = sorted(
                ((_episode_num(fp), fp) for fp in (seed_dir / "data").glob("episode*.hdf5")),
                key=lambda t: t[0],
            )
            if not episodes:
                continue
            scenes.append(
                {
                    "tier": tier,
                    "seed": seed,
                    "scene_id": f"{tier}/seed{seed}",
                    "episodes": episodes,
                }
            )
    return scenes


def peek_length(hdf5_path: Path, fail_max_length: int | None) -> int:
    return common.peek_length(hdf5_path, fail_max_length)


def convert_episode(
    *,
    hdf5_path: Path,
    scene: dict,
    scene_index: int,
    draw: int,
    episode_index: int,
    task_index: int,
    global_index_start: int,
    out_root: Path,
    fail_max_length: int | None,
) -> dict:
    """Convert one HDF5 episode; write its parquet + one mp4 per camera."""
    ep = common.load_robotwin_episode(hdf5_path, fail_max_length=fail_max_length)
    new_len = ep["new_len"]
    source_detail = f"{scene['scene_id']}/episode{draw}/{ep['generator']}"

    df = common.build_episode_dataframe(
        ep,
        episode_index=episode_index,
        task_index=task_index,
        global_index_start=global_index_start,
        data_source=DATA_SOURCE,
        source_detail=source_detail,
        include_clean_success=False,
        extra_columns={
            "scene_index": np.full(new_len, scene_index, dtype=np.int64),
            "tier": [scene["tier"]] * new_len,
            "seed": np.full(new_len, scene["seed"], dtype=np.int64),
            "draw": np.full(new_len, draw, dtype=np.int64),
            "scene_id": [scene["scene_id"]] * new_len,
        },
        column_order=CANONICAL_COLUMNS,
    )

    out_parquet = common.data_path(out_root, episode_index)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    common.write_episode_videos(ep["cam_data"], out_root, episode_index)

    return {
        "episode_index": episode_index,
        "length": new_len,
        "success": ep["success"],
        "scene_index": scene_index,
        "scene_id": scene["scene_id"],
        "tier": scene["tier"],
        "seed": scene["seed"],
        "draw": draw,
        "source_detail": source_detail,
        "df": df,
    }


# ---------------------------------------------------------------------------
# meta writers (self-contained; the base schema matches common's NO_CLEAN order)
# ---------------------------------------------------------------------------

ARRAY_COLUMNS = {
    "action": 14,
    "observation.state": 14,
    "observation.countertop.extrinsic_cv": 12,
    "observation.countertop.intrinsic_cv": 9,
    "observation.left.extrinsic_cv": 12,
    "observation.left.intrinsic_cv": 9,
    "observation.right.extrinsic_cv": 12,
    "observation.right.intrinsic_cv": 9,
}
SCALAR_INT_COLUMNS = ["task_index", "episode_index", "frame_index", "index", "scene_index", "seed", "draw"]
SCALAR_FLOAT_COLUMNS = ["timestamp"]
SCALAR_BOOL_COLUMNS = ["success"]
SCALAR_LABEL_COLUMNS = {
    "collision": "uint8",
    "collision_impulse": "float32",
    "contact": "uint8",
    "contact_impulse": "float32",
}
STRING_COLUMNS = ["collision_pairs", "contact_pairs", "data_source", "source_detail", "tier", "scene_id"]


def write_meta(out_root: Path, *, task_text: str, records: list[dict], scenes: list[dict],
               total_frames: int) -> None:
    meta = out_root / "meta"
    meta.mkdir(parents=True, exist_ok=True)

    # tasks.jsonl -- single task.
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": task_text}) + "\n")

    # episodes.jsonl -- scene metadata + outcome per episode.
    with open(meta / "episodes.jsonl", "w") as fh:
        for r in records:
            fh.write(json.dumps({
                "episode_index": r["episode_index"],
                "tasks": [task_text],
                "length": r["length"],
                "scene_index": r["scene_index"],
                "scene_id": r["scene_id"],
                "tier": r["tier"],
                "seed": r["seed"],
                "draw": r["draw"],
                "success": bool(r["success"]),
                "data_source": DATA_SOURCE,
                "source_detail": r["source_detail"],
                "fps": common.TARGET_FPS,
            }) + "\n")

    # episodes_stats.jsonl
    numeric_cols = (list(ARRAY_COLUMNS) + SCALAR_INT_COLUMNS + SCALAR_FLOAT_COLUMNS
                    + SCALAR_BOOL_COLUMNS + list(SCALAR_LABEL_COLUMNS))
    with open(meta / "episodes_stats.jsonl", "w") as fh:
        for r in records:
            df = r["df"]
            stats = {c: column_stats(df[c], is_array=c in ARRAY_COLUMNS)
                     for c in numeric_cols if c in df.columns}
            fh.write(json.dumps({"episode_index": r["episode_index"], "stats": stats}) + "\n")

    # scenes.jsonl -- the preserved structure: each scene -> its episodes + split.
    with open(meta / "scenes.jsonl", "w") as fh:
        for s in scenes:
            eps = [r for r in records if r["scene_index"] == s["scene_index"]]
            fh.write(json.dumps({
                "scene_index": s["scene_index"],
                "scene_id": s["scene_id"],
                "tier": s["tier"],
                "seed": s["seed"],
                "episode_indices": [r["episode_index"] for r in eps],
                "draws": [r["draw"] for r in eps],
                "n_episodes": len(eps),
                "n_success": sum(1 for r in eps if r["success"]),
                "n_fail": sum(1 for r in eps if not r["success"]),
            }) + "\n")

    # info.json
    write_info(meta / "info.json", records=records, total_frames=total_frames,
               sample_df=records[0]["df"])

    # modality.json
    write_modality(meta / "modality.json")


def write_info(path: Path, *, records: list[dict], total_frames: int, sample_df: pd.DataFrame) -> None:
    total_episodes = len(records)
    n_cams = len(common.CAMERA_MAP)
    total_chunks = max(1, (total_episodes + common.CHUNK_SIZE - 1) // common.CHUNK_SIZE)

    features: dict = {}
    for col, dim in ARRAY_COLUMNS.items():
        features[col] = {"dtype": "float32", "shape": [dim], "names": None}
    for col in SCALAR_INT_COLUMNS:
        features[col] = {"dtype": "int64", "shape": [1], "names": None}
    for col in SCALAR_FLOAT_COLUMNS:
        features[col] = {"dtype": "float64", "shape": [1], "names": None}
    for col in SCALAR_BOOL_COLUMNS:
        features[col] = {"dtype": "bool", "shape": [1], "names": None}
    for col, dtype in SCALAR_LABEL_COLUMNS.items():
        features[col] = {"dtype": dtype, "shape": [1], "names": None}
    for col in STRING_COLUMNS:
        features[col] = {"dtype": "string", "shape": [1], "names": None}
    for cam in common.CAMERA_MAP.values():
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [3, 240, 320],
            "names": ["channels", "height", "width"],
            "video_info": {
                "video.fps": common.TARGET_FPS,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }

    info = {
        "codebase_version": "v2.1",
        "robot_type": "roboreal_robopro",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_episodes * n_cams,
        "total_chunks": total_chunks,
        "chunks_size": common.CHUNK_SIZE,
        "fps": common.TARGET_FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    path.write_text(json.dumps(info, indent=2))


def write_modality(path: Path) -> None:
    modality = {
        "action": {
            "left_joints": {"start": 0, "end": 6, "original_key": "action"},
            "left_gripper": {"start": 6, "end": 7, "original_key": "action"},
            "right_joints": {"start": 7, "end": 13, "original_key": "action"},
            "right_gripper": {"start": 13, "end": 14, "original_key": "action"},
        },
        "state": {
            "left_joints": {"start": 0, "end": 6, "original_key": "observation.state"},
            "left_gripper": {"start": 6, "end": 7, "original_key": "observation.state"},
            "right_joints": {"start": 7, "end": 13, "original_key": "observation.state"},
            "right_gripper": {"start": 13, "end": 14, "original_key": "observation.state"},
        },
        "video": {cam: {"original_key": f"observation.images.{cam}"} for cam in common.CAMERA_MAP.values()},
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
        "label": {
            "collision": {"original_key": "collision", "note": "per-frame binary collision flag (RoboTwin sim)"},
            "contact": {"original_key": "contact", "note": "per-frame binary contact flag (RoboTwin sim)"},
            "success": {"original_key": "success", "note": "episode-level success outcome, broadcast per-frame"},
            "scene_id": {"original_key": "scene_id", "note": "tier/seedN folder this rollout came from; each scene mixes success and fail draws"},
            "scene_index": {"original_key": "scene_index", "note": "0-based scene id in canonical (tier, seed) order"},
            "data_source": {"original_key": "data_source", "note": DATA_SOURCE},
        },
    }
    path.write_text(json.dumps(modality, indent=2))


def write_readme(path: Path, *, src: Path, records: list[dict], scenes: list[dict],
                 total_frames: int, fail_max_length: int | None,
                 task_text: str = "", task_name: str = "") -> None:
    n_succ = sum(1 for r in records if r["success"])
    cams = ", ".join(common.CAMERA_MAP.values())
    by_tier = {}
    for s in scenes:
        by_tier.setdefault(s["tier"], 0)
        by_tier[s["tier"]] += 1
    tier_line = ", ".join(f"{t}={n}" for t, n in sorted(by_tier.items(), key=lambda kv: _tier_sort_key(kv[0])))
    cap = "kept full length (no truncation)" if fail_max_length is None else f"truncated to the first {fail_max_length} frames"
    title = task_name or src.name
    path.write_text(f"""# {title} — scene-preserving LeRobot v2.1

Converted from RoboTwin HDF5 at `{src.name}` **1:1 at {common.TARGET_FPS}fps**
(one frame = one raw row = one policy command).
Single task: **{task_text}**. 14-dim dual-arm joint+gripper
state; `action` is the next-step state. Cameras: {cams}.

- **{len(records)} episodes** across **{len(scenes)} scenes** ({n_succ} success /
  {len(records) - n_succ} fail).
- A *scene* is a `tier/seedN` folder ({tier_line} scenes per tier). Every scene
  mixes success and fail draws; the grouping is preserved via the
  `scene_index / scene_id / tier / seed / draw` columns (in every parquet and in
  `meta/episodes.jsonl`), and `meta/scenes.jsonl` maps each scene to its episode
  indices and success/fail split.
- Unsuccessful episodes: {cap}.

## Layout
- `data/chunk-*/episode_*.parquet`
- `videos/chunk-*/observation.images.{{{",".join(common.CAMERA_MAP.values())}}}/episode_*.mp4`
- `meta/` — info.json, episodes.jsonl, episodes_stats.jsonl, tasks.jsonl, modality.json, **scenes.jsonl**
- `manifests/` — the original `manifest*.json` copied verbatim

## Notes
- `success` is read from the HDF5 root attr (authoritative). The copied
  `manifest*.json` join by their `hdf5` path field; their own tier/seed labels
  are an inconsistent clean/high/mid labelling and are NOT used for scene ids.
- Per-frame `collision`/`contact` (+ impulses/pairs) come from the simulator.
- No `norm_stats.json` is emitted — normalization stats are model-specific and
  out of scope for this conversion.
""")


_REPO_ROOT = Path(__file__).resolve().parents[2]  # RoboPRO
PLAIN_INSTRUCTIONS = (
    _REPO_ROOT / "benchmark" / "bench_description" / "plain_instructions.json"
)


def detect_task(src: Path) -> tuple[str, str]:
    """Read ``task_name`` from the first episode; look its prompt up in RoboPRO's
    instruction bank. Returns ``(task_name, task_text)``; text is "" if unknown."""
    import h5py

    eps = sorted(src.glob("*/seed*/data/episode*.hdf5"))
    if not eps:
        raise SystemExit(f"no episodes under {src}")
    with h5py.File(eps[0], "r") as f:
        task_name = str(f.attrs.get("task_name", ""))
    text = ""
    if PLAIN_INSTRUCTIONS.exists():
        bank = json.loads(PLAIN_INSTRUCTIONS.read_text())
        for scene_map in bank.values():
            if task_name in scene_map:
                text = scene_map[task_name]
                break
    return task_name, text


def run(*, src: Path, out: Path, task_text: str, limit: int | None,
        fail_max_length: int | None, overwrite: bool, task_name: str = "") -> dict:
    if out.exists():
        if not overwrite:
            raise SystemExit(f"output {out} exists; pass --overwrite to replace it")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    scenes = discover_scenes(src)
    for i, s in enumerate(scenes):
        s["scene_index"] = i

    # Flatten episodes in canonical (scene, draw) order.
    items: list[tuple[dict, int, Path]] = []  # (scene, draw, hdf5_path)
    for s in scenes:
        for draw, fp in s["episodes"]:
            items.append((s, draw, fp))
    if limit is not None:
        items = items[:limit]

    lengths = [peek_length(fp, fail_max_length) for _, _, fp in items]
    offsets, running = [], 0
    for length in lengths:
        offsets.append(running)
        running += length
    total_frames = running

    print(f"[scenes] {len(scenes)} scenes, {len(items)} episodes -> {out} "
          f"(fail_max_length={fail_max_length}, target_fps={common.TARGET_FPS})")
    t0 = time.time()
    records: list[dict] = []
    for i, ((scene, draw, fp), length) in enumerate(zip(items, lengths)):
        r = convert_episode(
            hdf5_path=fp,
            scene=scene,
            scene_index=scene["scene_index"],
            draw=draw,
            episode_index=i,
            task_index=0,
            global_index_start=offsets[i],
            out_root=out,
            fail_max_length=fail_max_length,
        )
        if r["length"] != length:
            raise RuntimeError(f"length mismatch {fp}: peek={length} convert={r['length']}")
        records.append(r)
        if (i + 1) % 20 == 0 or i + 1 == len(items):
            print(f"  [{i + 1}/{len(items)}] {time.time() - t0:.0f}s  {scene['scene_id']}/ep{draw}")

    # Only keep scenes that actually produced episodes (relevant under --limit).
    kept_scene_idx = {r["scene_index"] for r in records}
    scenes_kept = [s for s in scenes if s["scene_index"] in kept_scene_idx]

    write_meta(out, task_text=task_text, records=records, scenes=scenes_kept,
               total_frames=total_frames)
    write_readme(out / "README.md", src=src, records=records, scenes=scenes_kept,
                 total_frames=total_frames, fail_max_length=fail_max_length,
                 task_text=task_text, task_name=task_name)

    # Copy all manifest JSONs verbatim.
    man_dir = out / "manifests"
    man_dir.mkdir(exist_ok=True)
    for fp in sorted(src.glob("manifest*.json")):
        shutil.copyfile(fp, man_dir / fp.name)

    summary = {
        "episodes": len(records),
        "scenes": len(scenes_kept),
        "frames": total_frames,
        "success": sum(1 for r in records if r["success"]),
        "fail": sum(1 for r in records if not r["success"]),
        "seconds": round(time.time() - t0, 1),
        "out": str(out),
    }
    print(f"[scenes] done: {summary}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="Scene-organised RoboTwin HDF5 root.")
    ap.add_argument("--out", type=Path, required=True, help="Output LeRobot dataset dir.")
    ap.add_argument("--task-text", default=None,
                    help="Single task prompt. Default: looked up from the episode's "
                         "task_name in RoboPRO's plain_instructions.json.")
    ap.add_argument("--limit", type=int, default=None, help="Only convert the first N episodes (smoke).")
    ap.add_argument("--fail-max-length", type=int, default=0,
                    help="Cap unsuccessful episodes to N frames, one frame = one policy "
                         "command (keep prefix). "
                         "0 (default) keeps full length.")
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing --out.")
    args = ap.parse_args()
    fail_max = None if args.fail_max_length == 0 else args.fail_max_length
    task_name, detected = detect_task(args.src)
    task_text = args.task_text or detected
    if not task_text:
        raise SystemExit(f"no instruction found for task_name={task_name!r}; pass --task-text")
    print(f"[scenes] task_name={task_name}  task_text={task_text!r}")
    run(src=args.src, out=args.out, task_text=task_text, limit=args.limit,
        fail_max_length=fail_max, overwrite=args.overwrite, task_name=task_name)


if __name__ == "__main__":
    main()
