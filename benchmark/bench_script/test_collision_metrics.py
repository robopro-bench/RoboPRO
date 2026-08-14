"""
Test script for collision metrics: load a bench env (office or study), run play_once(),
save video + collision log.

USAGE:
    source set_env.sh  # repo root
    cd sim
    python $BENCH_ROOT/bench_script/test_collision_metrics.py <task_name> <task_config> [options]

    (Ensure your conda/venv with PyYAML, numpy, imageio/cv2 is activated.)

EXAMPLES:
    # Office tasks
    python $BENCH_ROOT/bench_script/test_collision_metrics.py place_phone_shelf bench_demo_clean
    python $BENCH_ROOT/bench_script/test_collision_metrics.py put_mouse_on_pad bench_demo_clean --seed 42 --output-dir ./test_output
    python $BENCH_ROOT/bench_script/test_collision_metrics.py put_mouse_on_pad bench_demo_clean --bench-subdir office

    # Study tasks
    python $BENCH_ROOT/bench_script/test_collision_metrics.py move_cup bench_demo_clean --bench-subdir study
    python $BENCH_ROOT/bench_script/test_collision_metrics.py put_seal_in_box bench_demo_clean --bench-subdir study --scene-id 1

OUTPUTS (default dir: benchmark/collision_test/):
    - <task>_<config>.mp4                       : Video of demo_camera view
    - <task>_<config>_collision_log.json        : Per-step raw contact details for debug
    - <task>_<config>_collision_metrics.json    : Summary metrics (robot_to_furniture, robot_to_static_object, etc.)
    - <task>_<config>_contact_step_*.png        : Debug images for first contact every N steps (default: 50)
"""
import sys
import os
import argparse
import importlib
import yaml
import json
import subprocess
from pathlib import Path
import numpy as np

def _resolve_repo_paths():
    """Resolve benchmark/sim roots, tolerating a wrong SIM_ROOT."""
    script_path = Path(__file__).resolve()
    default_bench_root = script_path.parent.parent
    workspace_root = default_bench_root.parent

    def _valid_bench_root(path):
        return (path / "bench_envs").is_dir() and (path / "bench_task_config").is_dir()

    def _valid_sim_root(path):
        return (path / "envs").is_dir() and (path / "task_config" / "_embodiment_config.yml").exists()

    bench_candidates = [
        os.environ.get("BENCH_ROOT"),
        default_bench_root,
    ]
    robotwin_candidates = [
        os.environ.get("SIM_ROOT"),
        workspace_root / "sim",
    ]

    bench_root = None
    for candidate in bench_candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if _valid_bench_root(path):
            bench_root = path
            break

    sim_root = None
    for candidate in robotwin_candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if _valid_sim_root(path):
            sim_root = path
            break

    if bench_root is None or sim_root is None:
        raise SystemExit(
            "Could not resolve BENCH_ROOT/sim. "
            "Expected benchmark/ and sim/ to be sibling folders."
        )

    os.environ["BENCH_ROOT"] = str(bench_root)
    os.environ["SIM_ROOT"] = str(sim_root)
    return bench_root, sim_root


bench_root, sim_root = _resolve_repo_paths()

for p in [sim_root, bench_root]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

os.chdir(sim_root)

from envs import CONFIGS_PATH, resolve_embodiment_path


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
    """Load task env class from bench_envs, searching office/study subdirs as needed."""
    BENCH_SUBDIRS = ["office", "study", "kitchenl"]

    if bench_subdir:
        try:
            envs_module = importlib.import_module(f"bench_envs.{bench_subdir}.{task_name}")
            return _extract_task_class(envs_module, task_name)
        except ModuleNotFoundError:
            raise SystemExit(f"Task '{task_name}' not found in bench_envs.{bench_subdir}")

    # Try bench_envs.{task_name} first (flat structure)
    try:
        envs_module = importlib.import_module(f"bench_envs.{task_name}")
        return _extract_task_class(envs_module, task_name)
    except ModuleNotFoundError:
        pass

    # Try bench_envs.{subdir}.{task_name} for each known subdir
    for subdir in BENCH_SUBDIRS:
        try:
            envs_module = importlib.import_module(f"bench_envs.{subdir}.{task_name}")
            return _extract_task_class(envs_module, task_name)
        except ModuleNotFoundError:
            continue

    # Fallback to envs
    try:
        envs_module = importlib.import_module(f"envs.{task_name}")
        return _extract_task_class(envs_module, task_name)
    except ModuleNotFoundError:
        raise SystemExit(f"No task class found for '{task_name}' in bench_envs or envs")


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def collect_raw_contacts(env):
    """Return filtered contacts from env (populated by check_collisions)."""
    return getattr(env, "filtered_contacts_for_log", [])


def make_patched_take_dense_action(env, collision_log, frame_list, capture_every_n=5, output_dir=None, contact_image_every_n=50, contact_images_saved=None, run_prefix=""):
    """Return a patched take_dense_action that logs collisions and captures frames."""
    original = env.take_dense_action
    global_step = [0]  # use list to allow mutation in closure
    last_contact_image_at_step = [-(contact_image_every_n + 1)]  # so first contact triggers save
    if contact_images_saved is None:
        contact_images_saved = [0]

    def patched(control_seq, save_freq=-1):
        left_arm = control_seq["left_arm"]
        left_gripper = control_seq["left_gripper"]
        right_arm = control_seq["right_arm"]
        right_gripper = control_seq["right_gripper"]

        save_freq_val = env.save_freq if save_freq == -1 else save_freq
        if save_freq_val is not None:
            env._take_picture()

        max_control_len = 0
        if left_arm is not None:
            max_control_len = max(max_control_len, left_arm["position"].shape[0])
        if left_gripper is not None:
            max_control_len = max(max_control_len, left_gripper["num_step"])
        if right_arm is not None:
            max_control_len = max(max_control_len, right_arm["position"].shape[0])
        if right_gripper is not None:
            max_control_len = max(max_control_len, right_gripper["num_step"])

        for control_idx in range(max_control_len):
            if left_arm is not None and control_idx < left_arm["position"].shape[0]:
                env.robot.set_arm_joints(
                    left_arm["position"][control_idx],
                    left_arm["velocity"][control_idx],
                    "left",
                )
            if left_gripper is not None and control_idx < left_gripper["num_step"]:
                env.robot.set_gripper(
                    left_gripper["result"][control_idx],
                    "left",
                    left_gripper["per_step"],
                )
            if right_arm is not None and control_idx < right_arm["position"].shape[0]:
                env.robot.set_arm_joints(
                    right_arm["position"][control_idx],
                    right_arm["velocity"][control_idx],
                    "right",
                )
            if right_gripper is not None and control_idx < right_gripper["num_step"]:
                env.robot.set_gripper(
                    right_gripper["result"][control_idx],
                    "right",
                    right_gripper["per_step"],
                )

            env.scene.step()

            # --- Inject: collision metrics ---
            if hasattr(env, "robot_link_names"):
                env.check_collisions()

            # --- Inject: raw contact log ---
            step_contacts = collect_raw_contacts(env)
            if step_contacts:
                collision_log.append({
                    "global_step": global_step[0],
                    "control_idx": control_idx,
                    "contacts": step_contacts,
                })

                # --- Save first contact image every N steps ---
                if output_dir is not None and (global_step[0] - last_contact_image_at_step[0]) >= contact_image_every_n:
                    env._update_render()
                    env.cameras.update_picture()
                    rgb_dict = env.cameras.get_rgb()
                    cam_name = next((c for c in ("countertop_camera", "demo_camera", "head_camera") if c in rgb_dict), None)
                    if cam_name is not None:
                        img = rgb_dict[cam_name]["rgb"].copy()
                        if img.dtype != np.uint8:
                            img = (img * 255).clip(0, 255).astype(np.uint8)
                        prefix = f"{run_prefix}_" if run_prefix else ""
                        contact_img_path = output_dir / f"{prefix}contact_step_{global_step[0]:06d}.png"
                        try:
                            import imageio
                            imageio.imwrite(str(contact_img_path), img)
                        except ImportError:
                            import cv2
                            cv2.imwrite(str(contact_img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        last_contact_image_at_step[0] = global_step[0]
                        contact_images_saved[0] += 1

            # --- Inject: frame capture ---
            if global_step[0] % capture_every_n == 0:
                env._update_render()
                env.cameras.update_picture()
                rgb_dict = env.cameras.get_rgb()
                cam_name = next((c for c in ("countertop_camera", "demo_camera", "head_camera") if c in rgb_dict), None)
                if cam_name is not None:
                    img = rgb_dict[cam_name]["rgb"].copy()
                    if img.dtype != np.uint8:
                        img = (img * 255).clip(0, 255).astype(np.uint8)
                    frame_list.append(img)

            global_step[0] += 1

            # --- Original viewer render ---
            if env.render_freq and control_idx % env.render_freq == 0:
                env._update_render()
                if env.viewer is not None:
                    env.viewer.render()

            if save_freq_val is not None and control_idx % save_freq_val == 0:
                env._update_render()
                env._take_picture()

        if save_freq_val is not None:
            env._take_picture()

        return True

    return patched


def main():
    parser = argparse.ArgumentParser(description="Test collision metrics: run task, save video + collision log")
    parser.add_argument("task_name", type=str, help="Task module (e.g. place_phone_shelf, move_cup)")
    parser.add_argument("task_config", type=str, help="Task config (e.g. bench_demo_clean)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: <bench_root>/collision_test)")
    parser.add_argument("--capture-every", type=int, default=5, help="Capture frame every N physics steps")
    parser.add_argument("--contact-image-every", type=int, default=50, help="Save contact debug image every N steps (first contact only)")
    parser.add_argument("--render-freq", type=int, default=0, help="Render freq (0=headless)")
    parser.add_argument("--bench-subdir", type=str, default=None,
                        help="Subdirectory under bench_envs (e.g. office, study)")
    parser.add_argument("--scene-id", type=int, default=None,
                        help="Scene ID for study tasks (0-2). If not set, chosen randomly.")
    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    run_prefix = f"{task_name}_{task_config}"
    output_dir = Path(args.output_dir) if args.output_dir else bench_root / "collision_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config_path = bench_root / "bench_task_config" / f"{task_config}.yml"
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    cfg["task_name"] = task_name
    cfg["render_freq"] = args.render_freq
    cfg["now_ep_num"] = 0
    cfg["seed"] = args.seed
    cfg["need_plan"] = True
    cfg["save_data"] = False
    if args.scene_id is not None:
        cfg["scene_id"] = args.scene_id

    # Embodiment setup
    embodiment_type = cfg.get("embodiment", ["aloha-agilex"])
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(name):
        robot_file = resolve_embodiment_path(_embodiment_types[name]["file_path"])
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
    cfg["enable_collision_metrics"] = True

    # Build env
    print(f"Loading {task_name} with {task_config} (seed={args.seed})...")
    env_class = get_env_class(task_name, bench_subdir=args.bench_subdir)
    env = env_class()
    env.setup_demo(**cfg)

    # Debug: print collision name sets
    if hasattr(env, "robot_link_names"):
        print("\n--- Collision name sets (for debug) ---")
        all_names = {e.get_name() for e in env.scene.get_all_actors() if e.get_name()}
        unclassified = all_names - env.robot_link_names - env.furniture_names - env.target_object_names - env.static_object_names
        print(f"  Furniture: {env.furniture_names}")
        print(f"  Target objects: {env.target_object_names}")
        print(f"  Static objects: {env.static_object_names}")
        print(f"  Unclassified actors: {unclassified}")

    # Prepare recording
    collision_log = []
    frame_list = []
    contact_images_saved = [0]
    env.take_dense_action = make_patched_take_dense_action(
        env, collision_log, frame_list,
        capture_every_n=args.capture_every,
        output_dir=output_dir,
        contact_image_every_n=args.contact_image_every,
        contact_images_saved=contact_images_saved,
        run_prefix=run_prefix,
    )

    # Run play_once
    print("\nRunning play_once()...")
    env.play_once()
    print("Done.")

    # Save collision log
    log_path = output_dir / f"{run_prefix}_collision_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(collision_log, f, indent=2)
    print(f"Saved collision log: {log_path} ({len(collision_log)} steps with contacts)")
    if contact_images_saved[0] > 0:
        print(f"Saved {contact_images_saved[0]} contact debug images (every {args.contact_image_every} steps)")

    # Save metrics summary
    metrics = env.get_collision_metrics()
    metrics["task_success"] = bool(env.check_success())
    metrics_path = output_dir / f"{run_prefix}_collision_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")
    print("\n=== Collision Metrics Summary ===")
    max_key = max(len(k) for k in metrics)
    for k, v in metrics.items():
        if isinstance(v, list):
            names_str = ", ".join(v) if v else "(none)"
            print(f"  {k:<{max_key}} : {names_str}")
        else:
            print(f"  {k:<{max_key}} : {v}")

    # Save video
    video_path = output_dir / f"{run_prefix}.mp4"
    if frame_list:
        try:
            import imageio
            fps = 30
            writer = imageio.get_writer(str(video_path), fps=fps)
            for frame in frame_list:
                writer.append_data(frame)
            writer.close()
            print(f"Saved video: {video_path} ({len(frame_list)} frames, {fps} fps)")
        except ImportError:
            # Fallback: use ffmpeg if imageio not available
            import tempfile
            tmp_dir = tempfile.mkdtemp()
            for i, frame in enumerate(frame_list):
                img_path = Path(tmp_dir) / f"frame_{i:06d}.png"
                import cv2
                cv2.imwrite(str(img_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            subprocess.run([
                "ffmpeg", "-y", "-framerate", "30", "-i", f"{tmp_dir}/frame_%06d.png",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video_path)
            ], check=True, capture_output=True)
            import shutil
            shutil.rmtree(tmp_dir)
            print(f"Saved video: {video_path} ({len(frame_list)} frames)")
    else:
        print("No frames captured (no demo_camera/head_camera or no motion).")

    env.close_env()
    print("Done.")


if __name__ == "__main__":
    main()
