"""Run ONE seed of the grounding-data collector (single-episode subprocess).

The per-seed unit behind collect/collect_parallel.py's dynamic GPU dispatch
(workers pull seeds from a shared stream, all writing into ONE run dir).
Mirrors exactly one search-iteration + one collect-iteration
of collect_data.run(); concurrent-safe via an flock'd slot file:

  phase 1  qualify the seed (setup_demo + play_once, need_plan=True)
  claim    atomically take the next episode index from <run_dir>/slots.json
           (also appends the seed to seed.txt — claim order == episode order)
  phase 2  re-run the seed with data saving -> data/episode{i}.hdf5 + video

Exit codes (read by the launcher):
  0  episode saved          3  seed failed qualification (incl. UnStable/crash)
  4  surplus: target already met (nothing written)
  5  slot claimed but collection failed (slot burned — same gap semantics as
     the stock collector, which deletes the episode and moves on)

    python collect/collect_one_episode.py <task_name> <task_config> --seed N
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _env  # noqa: F401

import fcntl
import json
import os
import shutil
import traceback
from argparse import ArgumentParser

import collect_data as cd
from export_scene import export_scene


def _locked(run_dir: Path):
    lock = open(run_dir / ".slots.lock", "a+", encoding="utf-8")
    fcntl.flock(lock, fcntl.LOCK_EX)
    return lock


def _unlock(lock):
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()


def claim_slot(run_dir: Path, seed: int):
    """Atomically take the next episode index, or None if the target is met."""
    lock = _locked(run_dir)
    try:
        sf = run_dir / "slots.json"
        state = json.loads(sf.read_text(encoding="utf-8"))
        if state["count"] >= state["target"]:
            return None
        idx = state["count"]
        state["count"] += 1
        sf.write_text(json.dumps(state), encoding="utf-8")
        # stock seed.txt format: space-separated seeds, position == episode index
        with open(run_dir / "seed.txt", "a", encoding="utf-8") as f:
            f.write(f"{seed} ")
        return idx
    finally:
        _unlock(lock)


def update_scene_info(run_dir: Path, episode_idx: int, info: dict):
    lock = _locked(run_dir)
    try:
        p = run_dir / "scene_info.json"
        db = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        db[f"episode_{episode_idx}"] = info
        p.write_text(json.dumps(db, ensure_ascii=False, indent=4, default=str),
                     encoding="utf-8")
    finally:
        _unlock(lock)


def _safe_close(task):
    try:
        task.close_env()
    except Exception:
        pass


def main():
    ap = ArgumentParser()
    ap.add_argument("task_name")
    ap.add_argument("task_config")
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()

    task, args = cd.build_task_and_args(a.task_name, a.task_config)
    run_dir = Path(args["save_path"])  # already <save_path>/<task>/<config>
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = a.seed
    # Private per-seed scratch index (far above any real episode index) so
    # concurrent workers never collide on caches/traj/hdf5 while staging.
    TEMP = 900000 + seed

    # ── phase 1: qualification (mirrors the run() seed-search iteration) ────
    args["need_plan"] = True
    args["save_data"] = False  # qualification only — no frames written
    init_state = None
    try:
        task.setup_demo(now_ep_num=TEMP, seed=seed, **args)
        if hasattr(task, "_maybe_apply_language_perturbation"):
            task._maybe_apply_language_perturbation()
        # capture the t=0 scene state (before play_once mutates it) so this
        # episode can be replayed later to collect any forgotten modality.
        if hasattr(task, "capture_init_state"):
            try:
                init_state = task.capture_init_state()
            except Exception as _e:
                print(f"[seed {seed}] capture_init_state failed: {_e}")
        task.play_once()
        ok = bool(task.plan_success and task.check_success())
    except cd.UnStableError as e:  # exported into collect_data via `from envs import *`
        print(f"[seed {seed}] unstable scene: {e}")
        _safe_close(task)
        sys.exit(3)
    except Exception as e:  # noqa: BLE001
        print(f"[seed {seed}] qualification crashed: {e}")
        print(traceback.format_exc())
        _safe_close(task)
        sys.exit(3)

    if not ok:
        print(f"[seed {seed}] failed qualification "
              f"(plan_success={task.plan_success})")
        _safe_close(task)
        sys.exit(3)

    # Collect to the private TEMP index first; only claim a real (contiguous)
    # episode index AFTER a fully successful save (claim-on-success). This way a
    # rare phase-2 failure never consumes a slot, so "--episodes N" always yields
    # exactly N saved, gap-free episodes.
    print(f"[seed {seed}] qualified -> staging as temp {TEMP}")
    task.save_traj_data(TEMP)
    if init_state is not None and hasattr(task, "save_init_state"):
        try:
            task.save_init_state(TEMP, state=init_state)
        except Exception as _e:
            print(f"[seed {seed}] save_init_state failed: {_e}")
    _safe_close(task)

    # ── phase 2: collection (mirrors the run() collect iteration) ───────────
    args["need_plan"] = False
    args["render_freq"] = 0
    args["save_data"] = True

    def _rm(*paths):
        for p in paths:
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p)
                elif os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

    tmp_hdf5 = os.path.join(args["save_path"], "data", f"episode{TEMP}.hdf5")
    tmp_video = os.path.join(args["save_path"], "video", f"episode{TEMP}.mp4")
    tmp_scene = os.path.join(args["save_path"], "scene", f"episode{TEMP}")
    try:
        task.setup_demo(now_ep_num=TEMP, seed=seed, **args)
        if hasattr(task, "_maybe_apply_language_perturbation"):
            task._maybe_apply_language_perturbation()

        # Export the (post-settle) static scene geometry so link/sphere proximity
        # can be computed offline from per-frame qpos + the robot FK model.
        try:
            export_scene(task, tmp_scene)
        except Exception as e:  # noqa: BLE001
            print(f"[seed {seed}] export_scene failed: {e}")

        traj_data = task.load_tran_data(TEMP)
        args["left_joint_path"] = traj_data["left_joint_path"]
        args["right_joint_path"] = traj_data["right_joint_path"]
        task.set_path_lst(args)

        info = task.play_once()
        if info is None:
            info = getattr(task, "info", None) or {}
        if hasattr(task, "get_actor_id_map"):
            info["actor_id_map"] = task.get_actor_id_map()
        if hasattr(task, "get_role_names"):
            info["role_names"] = task.get_role_names()
        if hasattr(task, "get_object_poses"):
            info["object_pose_ids"] = task.get_object_poses()[0]  # column order of /object_poses
        if getattr(task, "enable_collision_metrics", False) and hasattr(task, "get_collision_metrics"):
            info["collision_metrics"] = task.get_collision_metrics()

        success = bool(task.check_success())
        cache_dir = os.path.join(args["save_path"], ".cache", f"episode{TEMP}")
        has_frames = os.path.isdir(cache_dir) and any(
            fn.endswith(".pkl") for fn in os.listdir(cache_dir))
        print(f"[seed {seed}] temp {TEMP}: success={success} (frames={has_frames})")

        task.close_env(clear_cache=True)
        if has_frames:
            task.merge_pkl_to_hdf5_video()
        task.remove_data_cache()

        if not (success and has_frames):
            _rm(tmp_hdf5, tmp_video, tmp_scene)
            print(f"[seed {seed}] collection failed — temp discarded (no slot used)")
            sys.exit(5)
    except Exception as e:  # noqa: BLE001
        print(f"[seed {seed}] crashed during collection: {e}")
        print(traceback.format_exc())
        for cleanup in (task.remove_data_cache, task.close_env):
            try:
                cleanup()
            except Exception:
                pass
        _rm(tmp_hdf5, tmp_video, tmp_scene)
        sys.exit(5)

    # success — now atomically claim the next contiguous index and promote temp.
    episode_idx = claim_slot(run_dir, seed)
    if episode_idx is None:
        _rm(tmp_hdf5, tmp_video, tmp_scene)
        print(f"[seed {seed}] surplus — target met while collecting; temp discarded")
        sys.exit(4)
    hdf5_path = os.path.join(args["save_path"], "data", f"episode{episode_idx}.hdf5")
    video_path = os.path.join(args["save_path"], "video", f"episode{episode_idx}.mp4")
    os.replace(tmp_hdf5, hdf5_path)
    if os.path.exists(tmp_video):
        os.replace(tmp_video, video_path)
    scene_path = os.path.join(args["save_path"], "scene", f"episode{episode_idx}")
    if os.path.isdir(tmp_scene):
        _rm(scene_path)
        os.replace(tmp_scene, scene_path)
    # promote the trajectory + init-state (needed for replay) from temp to idx
    _td = os.path.join(args["save_path"], "_traj_data")
    for _suf in (".pkl", "_init.json"):
        _src = os.path.join(_td, f"episode{TEMP}{_suf}")
        if os.path.exists(_src):
            os.replace(_src, os.path.join(_td, f"episode{episode_idx}{_suf}"))
    cd._stamp_provenance_attrs(hdf5_path, args, seed=seed, success=success,
                               timestep=getattr(task, 'timestep', None))
    info["seed"] = seed
    update_scene_info(run_dir, episode_idx, info)
    # Grounding needs the per-frame actor_bbox group (only written when
    # data_type.actor_bbox is on). Without it masking_resolve raises SystemExit,
    # which escapes except-Exception and would mark a saved episode as error_rc1.
    if (args.get("data_type") or {}).get("actor_bbox", False) and getattr(cd, "finalize_grounding", None) is not None:
        try:
            cd.finalize_grounding(args["save_path"], episode_idx, obj_pad=args.get("table_obj_pad"))
        except Exception as _me:  # noqa: BLE001
            print(f"[seed {seed}] grounding failed (episode {episode_idx}): {_me}")
    print(f"[seed {seed}] episode {episode_idx} saved")
    sys.exit(0)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    main()
