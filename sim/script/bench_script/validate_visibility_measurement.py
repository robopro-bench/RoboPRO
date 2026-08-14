"""
Validation for the Phase 0 visibility-measurement primitive (issue #28).

Builds a known office scene (default: put_mouse_on_pad) headlessly, then exercises
``Base_Task.measure_target_visibility`` on the countertop camera and checks:

  1. Label correctness  - overlay the resolved target mask on countertop RGB
                           (saved as PNG) so it can be confirmed by eye to cover
                           the target object and nothing else.
  2. Sanity of counts    - a normally-placed target yields a plausible pixel
                           count and bucket; the same target teleported far
                           off-frame yields visible_pixel_count == 0 -> not_visible.
  3. No truncation       - report the max raw actor id present; if it exceeds 255
                           the old uint8 path would have aliased it.

USAGE (from the repo root):
    source set_env.sh
    cd sim
    python script/bench_script/validate_visibility_measurement.py \
        put_mouse_on_pad bench_demo_office_clean --bench-subdir office --seed 0 \
        --out-dir ./visibility_validation
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import imageio

from setup_paths import setup_paths
setup_paths()

bench_root = Path(os.environ["BENCH_ROOT"])
sim_root = Path(os.environ["SIM_ROOT"])
os.chdir(sim_root)  # match path resolution used by collect_data / visualize

import yaml
from envs import CONFIGS_PATH
from visualize_task_scene import get_env_class, get_embodiment_config


def build_env(task_name, task_config, seed, bench_subdir):
    env_class = get_env_class(task_name, bench_subdir=bench_subdir)

    from script.task_loader import bench_config_path
    config_path = bench_config_path(task_config)
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    cfg["task_name"] = task_name
    cfg["render_freq"] = 0          # headless
    cfg["now_ep_num"] = 0
    cfg["seed"] = seed if seed != -1 else int(np.random.randint(100))
    cfg["need_plan"] = True
    cfg["save_data"] = False

    embodiment_type = cfg.get("embodiment", ["aloha-agilex"])
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def emb_file(name):
        robot_file = _embodiment_types[name]["file_path"]
        if robot_file is None:
            raise SystemExit("missing embodiment files")
        return robot_file

    if len(embodiment_type) == 1:
        cfg["left_robot_file"] = emb_file(embodiment_type[0])
        cfg["right_robot_file"] = emb_file(embodiment_type[0])
        cfg["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        cfg["left_robot_file"] = emb_file(embodiment_type[0])
        cfg["right_robot_file"] = emb_file(embodiment_type[1])
        cfg["embodiment_dis"] = embodiment_type[2]
        cfg["dual_arm_embodied"] = False
    else:
        raise SystemExit("embodiment config should have 1 or 3 entries")

    cfg["left_embodiment_config"] = get_embodiment_config(cfg["left_robot_file"])
    cfg["right_embodiment_config"] = get_embodiment_config(cfg["right_robot_file"])

    env = env_class()
    env.setup_demo(**cfg)
    return env, cfg["seed"]


def overlay_mask(rgb, mask, color=(255, 0, 0), alpha=0.5):
    out = rgb.copy()
    out[mask] = (alpha * np.array(color) + (1 - alpha) * out[mask]).astype(np.uint8)
    return out


def get_target(env):
    """Locate the task's primary target actor wrapper."""
    for attr in ("target_obj", "target", "mouse"):
        if hasattr(env, attr):
            return getattr(env, attr)
    raise SystemExit("Could not find a target actor attribute on the env")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_name", nargs="?", default="put_mouse_on_pad")
    ap.add_argument("task_config", nargs="?", default="bench_demo_office_clean")
    ap.add_argument("--bench-subdir", default="office")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--camera", default="countertop_camera")
    ap.add_argument("--out-dir", default="./visibility_validation")
    ap.add_argument("--demo-occluder", action="store_true",
                    help="Capture the full count, then grow a box on the camera->target "
                         "line and re-measure to show visible_fraction fall below 1.0 "
                         "(mini-preview of Phase 2).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env, seed = build_env(args.task_name, args.task_config, args.seed, args.bench_subdir)
    target = get_target(env)

    print("\n================ Visibility measurement validation ================")
    print(f"task={args.task_name}  config={args.task_config}  seed={seed}  camera={args.camera}")
    print(f"target attr resolved -> name={target.get_name()}")

    # --- denominator capture on the clean scene (pre-occlusion, same camera) ---
    denom = env.capture_target_pixel_count(target, camera_name=args.camera)
    print(f"\n[denominator] full_target_pixel_count = {denom}")

    # --- main measurement ---
    res = env.measure_target_visibility(target, camera_name=args.camera, denominator=denom)
    print("\n[measure] normally-placed target:")
    print(f"  target_ids          = {res['target_ids']}")
    print(f"  visible_pixel_count = {res['visible_pixel_count']}")
    print(f"  in_fov              = {res['in_fov']}")
    print(f"  visible_fraction    = {res['visible_fraction']:.4f}")
    print(f"  bucket              = {res['bucket']}")

    # --- truncation check: report the raw id range present ---
    seg = env.cameras.get_segmentation_raw(level="actor", camera_name=args.camera)
    seg_img = seg[args.camera]["actor_segmentation_raw"]
    uniq = np.unique(seg_img)
    print("\n[truncation] raw actor-seg stats on this camera:")
    print(f"  dtype={seg_img.dtype}  num_unique_ids={uniq.size}  max_id={int(uniq.max())}")
    print(f"  -> ids > 255 present? {'YES (uint8 path would alias these)' if uniq.max() > 255 else 'no'}")

    # --- label-correctness overlay ---
    rgb = env.cameras.get_rgb()[args.camera]["rgb"]
    overlay = overlay_mask(rgb, res["mask"])
    side = np.concatenate([rgb, overlay], axis=1)
    overlay_path = out_dir / f"{args.task_name}_seed{seed}_overlay.png"
    imageio.imwrite(overlay_path, side)
    print(f"\n[label-correctness] saved RGB|overlay to {overlay_path}")
    print("  -> confirm by eye: red mask covers ONLY the target object.")

    # --- off-frame sanity: teleport the target far away, expect 0 visible px ---
    import sapien
    orig_pose = target.actor.get_pose()
    target.actor.set_pose(sapien.Pose([50.0, 50.0, 50.0], orig_pose.q))
    off = env.measure_target_visibility(target, camera_name=args.camera, denominator=denom)
    print("\n[off-frame] target teleported to (50,50,50):")
    print(f"  visible_pixel_count = {off['visible_pixel_count']}  in_fov = {off['in_fov']}  bucket = {off['bucket']}")
    off_ok = off["visible_pixel_count"] == 0 and off["bucket"] == "not_visible"
    target.actor.set_pose(orig_pose)  # restore

    # --- optional: grow an occluder to demonstrate visible_fraction < 1.0 ---
    if args.demo_occluder:
        from envs.utils import create_box
        tp = target.actor.get_pose().p
        print("\n[demo-occluder] denominator captured on clean scene; "
              "growing a box on the camera->target line:")
        print(f"  {'box_height(m)':>14} | {'visible_px':>10} | {'fraction':>8} | bucket")
        for h in (0.03, 0.06, 0.10, 0.16, 0.24):
            # box bottom on the table (create_box adds table_z_bias to z),
            # placed just robot-side (-y) of the target on the countertop sightline.
            box = create_box(
                scene=env,
                pose=sapien.Pose([float(tp[0]), float(tp[1]) - 0.035, 0.74 + h]),
                half_size=[0.06, 0.012, h],
                color=(0.2, 0.2, 0.2),
                name=f"occluder_demo_{h}",
                is_static=True,
            )
            r = env.measure_target_visibility(target, camera_name=args.camera, denominator=denom)
            print(f"  {h:>14.2f} | {r['visible_pixel_count']:>10} | {r['visible_fraction']:>8.3f} | {r['bucket']}")
            if h == 0.06:  # save one partial-occlusion overlay for inspection
                rgb_d = env.cameras.get_rgb()[args.camera]["rgb"]
                imageio.imwrite(out_dir / f"{args.task_name}_seed{seed}_occluder_h{h}.png",
                                np.concatenate([rgb_d, overlay_mask(rgb_d, r["mask"])], axis=1))
            box.actor.set_pose(sapien.Pose([100.0, 100.0, 100.0]))  # remove before next height
        print("  -> fraction should decrease monotonically toward 0 as the box grows.")

    # --- verdict ---
    print("\n================ Verdict ================")
    checks = {
        "target id resolved (>=1 id)": len(res["target_ids"]) >= 1,
        "visible normally (count>0, in_fov)": res["visible_pixel_count"] > 0 and res["in_fov"],
        "off-frame -> 0 px / not_visible": off_ok,
        "fraction in [0,1]": 0.0 <= (res["visible_fraction"] or 0) <= 1.0,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    all_ok = all(checks.values())
    print(f"\nOverall: {'PASS' if all_ok else 'FAIL'}  (inspect the overlay PNG for label correctness)")

    env.close_env()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
