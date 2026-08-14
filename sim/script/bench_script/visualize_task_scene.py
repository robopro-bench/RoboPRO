"""
Simple visualization script for benchmark task scenes.
Uses environments from bench_envs and the same config layout as collect_data.

USAGE:
    Run this script from the benchmark folder:

    cd benchmark
    source set_env.sh  # or set SIM_ROOT and BENCH_ROOT manually
    python bench_script/visualize_task_scene.py <task_name> <task_config> [options]

EXAMPLES:
    # Basic usage with default seed
    python bench_script/visualize_task_scene.py grab_roller_thing bench_demo_clean

    # With custom seed
    python bench_script/visualize_task_scene.py grab_roller_thing bench_demo_clean --seed 42

    # With custom render frequency
    python bench_script/visualize_task_scene.py grab_roller_thing bench_demo_clean --render-freq 5

    # Viewer aligned to a scene camera (see embodiment static_camera_list + bench office_config)
    python bench_script/visualize_task_scene.py put_bottle_in_fridge bench_demo_clean --viewer-camera demo_camera

    # Roll out the task (run play_once) then view
    python bench_script/visualize_task_scene.py put_away_stapler bench_demo_clean --rollout

    # With explicit bench subdir (for tasks under office/, study/, etc.)
    python bench_script/visualize_task_scene.py mouse_on_pad bench_demo_clean --bench-subdir office --rollout --seed 0

    # Run headless (no GUI)
    python bench_script/visualize_task_scene.py grab_roller_thing bench_demo_clean --no-render
    python bench_script/visualize_task_scene.py grab_roller_thing bench_demo_clean --no-render --rollout

    # Save RGB on first planning failure (move_step = Nth top-level env.move in play_once)
    python bench_script/visualize_task_scene.py move_books_onto_table bench_demo_clean --bench-subdir study \\
        --rollout --save-plan-fail-dir ./plan_fail_debug --plan-fail-camera head_camera
    
    python script/bench_script/visualize_task_scene.py close_fridge bench_demo_randomized --bench-subdir kitchenl --rollout --seed 0 --save-plan-fail-dir ./plan_fail_debug --plan-fail-camera head_camera --no-render --save_data

ARGUMENTS:
    task_name      Task module name from bench_envs (e.g. grab_roller_thing)
    task_config    Task config name without .yml extension (e.g. bench_demo_clean)

OPTIONS:
    --seed N                   Random seed for scene initialization (default: 0)
    --render-freq N            Render every N simulation steps (default: 1)
    --viewer-camera N          Match viewer to this scene camera name (default: demo_camera; use default to keep built-in pose)
    --rollout                  Run play_once() to roll out the task; if not set, only view initial setup
    --bench-subdir S           Subdirectory under bench_envs (e.g. office, study). If set, imports bench_envs.<S>.<task_name>
    --no-render                Disable simulator rendering (run headless). Default: render on.
    --save-plan-fail-dir PATH  With --rollout, save camera RGB when a top-level env.move() fails planning
    --plan-fail-camera NAME    Camera key for save_camera_rgb (default: head_camera)

NOTES:
    - The script automatically changes directory to sim for proper path resolution
    - Requires SIM_ROOT and BENCH_ROOT environment variables (set via set_env.sh)
    - Close the viewer window to exit the visualization
"""
import sys
import os
import argparse
import importlib
import time
import yaml
import json
from pathlib import Path
import numpy as np

from setup_paths import setup_paths
setup_paths()
# Paths: script is run from benchmark folder, but changes to sim
current_file_path = os.path.abspath(__file__)
script_dir = os.path.dirname(current_file_path)
# bench_root is the parent of bench_script directory
bench_root = Path(os.environ["BENCH_ROOT"])
sim_root = Path(os.environ["SIM_ROOT"])

os.chdir(sim_root)  # Change to sim for proper path resolution

from envs import CONFIGS_PATH  # from sim


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _build_object_hypotheses(scene_info):
    hypotheses = []

    for key, value in (scene_info.get("info") or {}).items():
        if len(key) == 3 and key.startswith("{") and key.endswith("}") and key[1].islower():
            hypotheses.append({
                "source": "task_info",
                "placeholder": key,
                "entity_type": "arm",
                "value": value,
            })
        else:
            hypotheses.append({
                "source": "task_info",
                "placeholder": key,
                "entity_type": "object",
                "asset_ref": value,
            })

    for idx, clutter_obj in enumerate(scene_info.get("cluttered_table_info") or []):
        hypotheses.append({
            "source": "cluttered_table_info",
            "entity_type": "distractor",
            "hypothesis_index": idx,
            "object_type": clutter_obj.get("object_type"),
            "object_index": clutter_obj.get("object_index"),
        })

    return hypotheses


def _write_rollout_scene_info(env, task_name, task_config, seed, success):
    save_dir = Path(env.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    scene_info_path = save_dir / "scene_info.json"

    episode_id = f"{env.__class__.__name__}_{env.ep_num}"
    scene_info = _to_jsonable(getattr(env, "info", {}) or {})
    collision_info = {}
    if hasattr(env, "get_collision_metrics"):
        collision_info = _to_jsonable(env.get_collision_metrics())

    scene_info["collision_info"] = collision_info
    scene_info["object_hypotheses"] = _build_object_hypotheses(scene_info)
    scene_info["metadata"] = {
        "task_name": task_name,
        "task_config": task_config,
        "seed": seed,
        "success": bool(success),
        "episode_id": episode_id,
        "data_file": f"data/episode_{episode_id}.hdf5",
        "video_file": f"video/episode_{episode_id}.mp4",
    }

    if scene_info_path.exists():
        with open(scene_info_path, "r", encoding="utf-8") as f:
            scene_db = json.load(f)
    else:
        scene_db = {}

    scene_db[f"episode_{episode_id}"] = scene_info
    with open(scene_info_path, "w", encoding="utf-8") as f:
        json.dump(scene_db, f, ensure_ascii=False, indent=2)

    print(f"Saved scene info to {scene_info_path} [episode_{episode_id}]")


def sync_viewer_to_scene_camera(env, camera_name: str) -> bool:
    """
    Match the interactive viewer frustum to an existing scene camera (same pose as data-collection cams).
    ``camera_name`` must appear in ``env.cameras.static_camera_name`` (e.g. head_camera, demo_camera).
    """
    if not camera_name or str(camera_name).lower() in ("default", "none", ""):
        return False
    viewer = getattr(env, "viewer", None)
    cams = getattr(env, "cameras", None)
    if viewer is None or cams is None:
        print("Warning: cannot sync viewer (missing viewer or cameras).")
        return False
    names = getattr(cams, "static_camera_name", None) or []
    if camera_name not in names:
        print(
            f"Warning: viewer-camera '{camera_name}' not in env static cameras {names}. "
            "Using default viewer pose; pass a valid name or --viewer-camera default."
        )
        return False
    idx = names.index(camera_name)
    cam = cams.static_camera_list[idx]
    try:
        pose = cam.entity.get_pose()
        viewer.set_camera_pose(pose)
    except Exception as e:
        print(f"Warning: failed to sync viewer to '{camera_name}': {e}")
        return False
    print(f"Viewer pose aligned with scene camera '{camera_name}'.")
    return True


def _extract_task_class(envs_module, task_name):
    """Extract task class from module, handling class names that differ from module name."""
    try:
        return getattr(envs_module, task_name)
    except AttributeError:
        from envs._base_task import Base_Task
        for name in dir(envs_module):
            obj = getattr(envs_module, name)
            if isinstance(obj, type) and issubclass(obj, Base_Task) and obj is not Base_Task:
                return obj
        raise SystemExit(f"No task class found in {envs_module.__name__}")


def get_env_class(task_name, bench_subdir=None):
    """Load task env class from bench_envs."""
    from script.task_loader import load_task_module
    envs_module = load_task_module(task_name, bench_subdir=bench_subdir)
    return _extract_task_class(envs_module, task_name)


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def _install_plan_fail_camera_saver(env, out_dir: Path, task_name: str, seed: int, camera_name: str):
    """
    Wrap env.move: on the first planning failure in a rollout, save RGB. move_step counts each
    top-level move() call in play_once (not inner simulation frames).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    move_step = [0]
    orig_move = env.move

    def wrapped_move(*args, **kwargs):
        move_step[0] += 1
        was_ok = env.plan_success
        ret = orig_move(*args, **kwargs)
        if was_ok and ret is False:
            stem = f"{task_name}_seed{seed}_move_step_{move_step[0]}"
            path = out_dir / f"{stem}.png"
            try:
                env.save_camera_rgb(str(path), camera_name=camera_name)
                print(f"[plan-fail] saved {path} (move step {move_step[0]})")
            except Exception as e:
                print(f"[plan-fail] could not save camera ({camera_name}): {e}")
        return ret

    env.move = wrapped_move


def main():
    parser = argparse.ArgumentParser(description="Visualize a benchmark task scene")
    parser.add_argument("task_name", type=str, help="Task module name (e.g. grab_roller_thing)")
    parser.add_argument("task_config", type=str, help="Task config name (e.g. bench_demo_clean)")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed for scene")
    parser.add_argument("--render-freq", type=int, default=3, help="Render every N steps (default 1)")
    parser.add_argument("--rollout", action="store_true", help="Run play_once() to roll out the task")
    parser.add_argument("--bench-subdir", type=str, default=None,
                       help="Subdirectory under bench_envs (e.g. office, study)")
    parser.add_argument("--save_data", action="store_true", help="Save the visualization")
    parser.add_argument(
        "--viewer-camera",
        type=str,
        default="demo_camera",
        help=(
            "Scene camera name to copy the viewer pose from (see embodiment static_camera_list), "
            "e.g. demo_camera, head_camera, front_camera. Use 'default' to keep setup_scene viewer xyz/rpy."
        ),
    )
    parser.add_argument("--no-render", action="store_true",
                       help="Disable simulator rendering (run headless). Default: render on.")
    parser.add_argument(
        "--save-plan-fail-dir",
        type=str,
        default=None,
        help="If set with --rollout, save camera RGB when a top-level env.move() fails planning",
    )
    parser.add_argument(
        "--plan-fail-camera",
        type=str,
        default="demo_camera",
        help="Camera name passed to save_camera_rgb (default: demo_camera)",
    )

    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    seed = args.seed
    render_freq = args.render_freq
    rollout = args.rollout
    bench_subdir = args.bench_subdir
    enable_render = not args.no_render

    # Load env class from bench_envs
    env_class = get_env_class(task_name, bench_subdir=bench_subdir)

    from script.task_loader import bench_config_path
    config_path = bench_config_path(task_config)
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    cfg["task_name"] = task_name
    cfg["render_freq"] = render_freq if enable_render else 0
    cfg["now_ep_num"] = 0
    cfg["seed"] = seed if seed != -1 else np.random.randint(100)
    cfg["need_plan"] = True
    cfg["save_data"] = bool(args.save_data)

    # Embodiment setup (same as collect_data)
    embodiment_type = cfg.get("embodiment", ["aloha-agilex"])
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type_name):
        robot_file = _embodiment_types[embodiment_type_name]["file_path"]
        if robot_file is None:
            raise SystemExit("missing embodiment files")
        return robot_file

    if len(embodiment_type) == 1:
        cfg["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        cfg["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        cfg["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        cfg["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        cfg["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        cfg["embodiment_dis"] = embodiment_type[2]
        cfg["dual_arm_embodied"] = False
    else:
        raise SystemExit("embodiment config should have 1 or 3 entries")

    cfg["left_embodiment_config"] = get_embodiment_config(cfg["left_robot_file"])
    cfg["right_embodiment_config"] = get_embodiment_config(cfg["right_robot_file"])

    # Build env and setup scene with viewer
    env = env_class()
    env.setup_demo(**cfg)

    if enable_render and (not getattr(env, "render_freq", 0) or getattr(env, "viewer", None) is None):
        print("Warning: viewer not created (render_freq was 0?). Exiting.")
        env.close_env()
        return

    if enable_render:
        sync_viewer_to_scene_camera(env, args.viewer_camera)

    if rollout:
        print("Rolling out task (play_once)...")
        if args.save_plan_fail_dir:
            _install_plan_fail_camera_saver(
                env,
                Path(args.save_plan_fail_dir),
                task_name,
                cfg["seed"],
                args.plan_fail_camera,
            )
        env.play_once()
        if env.save_data:
            rollout_success = env.check_success()
            _write_rollout_scene_info(env, task_name, task_config, cfg["seed"], rollout_success)
            env.ep_num = f"_{env.__class__.__name__}_{env.ep_num}"
            env.close_env(clear_cache=True)
            env.merge_pkl_to_hdf5_video()
            env.remove_data_cache()
        print("Rollout done.")

    if enable_render:
        viewer = env.viewer
        if not rollout:
            print("Scene ready. Close the viewer window to exit.")
        else:
            print("Close the viewer window to exit.")
        while not viewer.closed:
            env.scene.step()
            env.scene.update_render()
            viewer.render()
    else:
        if not rollout:
            print("Scene ready (headless). Press Ctrl+C to exit.")
        else:
            print("Rollout done (headless). Exiting.")
            print(f"Success: {env.check_success()}")
            if not env.save_data:
                env.close_env()
            print("Done.")
            return
        try:
            while True:
                env.scene.step()
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass

    print(f"Success: {env.check_success()}")
    env.close_env()
    print("Done.")


if __name__ == "__main__":
    main()
