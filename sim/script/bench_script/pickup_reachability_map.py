#!/usr/bin/env python3
"""
Post-pickup reachability maps for BACKWARD (placement) subgoal search (issue #35).

For each seed: build the occluder scene, run the rollout ONLY up to the bottle being
picked up (side waypoint -> grasp -> lift), then stop. If the pickup succeeded, compute
a collision-free gripper IK reachability map (box + table world; the held bottle is NOT
modelled -- we only care where the gripper can go) for the arm that did the pickup, and
plot it with the occluder / target / pad overlaid, labelled SUCCESS + seed. If the
pickup FAILED, plot only the scene (occluder / target / pad), no IK, labelled FAILED.

The reachable cells are candidate SUBGOAL positions for carrying the bottle back to the
pad -- a chain of nearby reachable cells is what trajopt can actually connect.

Reuses the IK solver / frame-transform / grid machinery from reachability_map.py.

USAGE (from sim, env sourced):
    python pickup_reachability_map.py --seeds 1,2,3,4,5 --offset 0.2
    python pickup_reachability_map.py --seeds 10-20 --z 0.90 --chunk 128
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from setup_paths import setup_paths
setup_paths()

import torch  # noqa: E402
from envs.utils import ArmTag  # noqa: E402
from analyze_occluder_visibility import make_occluder_task, PAD_XY, OCC_HALF_FOOTPRINT  # noqa: E402
from analyze_natural_visibility import build_cfg, DR_CLEAN  # noqa: E402
# IK machinery (fresh cuda-graph-off solver, frame transform, chunked batch solve)
from reachability_map import _build_ik_solver, _solve_grid  # noqa: E402


def _parse_seeds(s):
    if "-" in s and "," not in s:
        a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def _pickup_only(env):
    """Run OccluderTask.play_once truncated to the pickup (env.PICKUP_ONLY=True) so this
    uses the EXACT rollout path -- the first (around-box) subgoal, the forced side grasp,
    grasp_actor_from_table and lift -- then report whether the bottle actually rose."""
    z0 = float(env.target_obj.get_pose().p[2])
    env.play_once()
    z1 = float(env.target_obj.get_pose().p[2])
    return bool(env.plan_success) and (z1 - z0 > 0.05)


def _reach_grid(env, arm_tag, grasp_q, args):
    """Gripper IK reachability grid for `arm_tag` at fixed z + the grasp orientation."""
    name = str(arm_tag)
    planner = env.robot.left_planner if name == "left" else env.robot.right_planner

    xs = np.arange(args.xmin, args.xmax + 1e-9, args.res)
    ys = np.arange(args.ymin, args.ymax + 1e-9, args.res)
    XX, YY = np.meshgrid(xs, ys)
    gp = np.zeros((XX.size, 7))
    gp[:, 0] = XX.ravel(); gp[:, 1] = YY.ravel(); gp[:, 2] = args.z
    gp[:, 3:] = grasp_q

    ik = _build_ik_solver(planner)
    succ = _solve_grid(env.robot, planner, ik, arm_tag, gp, chunk=args.chunk).reshape(XX.shape)
    del ik
    torch.cuda.empty_cache()
    return XX, YY, succ


def _plot_seed(seed, ok, XX, YY, succ, box_p, tgt_p, pad_xy, arm_name, args):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(args.xmin, args.xmax); ax.set_ylim(args.ymin, args.ymax)
    if ok and succ is not None:
        ax.imshow(succ, origin="lower", extent=[args.xmin, args.xmax, args.ymin, args.ymax],
                  cmap="Greens", alpha=0.8, aspect="equal", vmin=0, vmax=1)
        status = f"SUCCESS (pickup, {arm_name} arm) -- green = reachable+collision-free"
    else:
        status = "FAILED (no pickup) -- no IK"
    h = OCC_HALF_FOOTPRINT
    ax.add_patch(plt.Rectangle((box_p[0] - h, box_p[1] - h), 2 * h, 2 * h,
                               fill=False, edgecolor="red", lw=2, label="occluder"))
    ax.plot(tgt_p[0], tgt_p[1], "b*", ms=16, label="target")
    ax.plot(pad_xy[0], pad_xy[1], "ms", ms=12, label="pad")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"seed {seed} @ z={args.z:.2f}m\n{status}")
    ax.legend(loc="upper right"); ax.set_aspect("equal")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tag = "ok" if ok else "fail"
    p = out / f"pickup_reach_seed{seed:04d}_off{args.offset}_z{args.z:.2f}_{tag}.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    print(f"  saved {p}")


def run(args):
    seeds = _parse_seeds(args.seeds)
    env = make_occluder_task()()
    env.spawn_occluder = True
    env.occluder_offset = args.offset
    env.PICKUP_ONLY = True             # play_once() stops right after the lift
    print(f"seeds={seeds}  offset={args.offset}  z={args.z}")

    for seed in seeds:
        try:
            cfg = build_cfg("put_mouse_on_pad", args.base_config, seed, DR_CLEAN, rollout=True)
            cfg["save_data"] = False   # need_plan=True to execute, but no video capture
            env.setup_demo(**cfg)
        except Exception as e:
            print(f"[seed {seed}] scene build failed ({type(e).__name__}: {e}); skipping")
            continue

        box_p = np.array(env.occluder.get_pose().p)
        tgt_p = np.array(env.target_obj.get_pose().p)
        arm_tag = ArmTag("right" if tgt_p[0] > 0 else "left")

        # grasp orientation from the PRE-pickup target (for the reachability-grid quat);
        # play_once recomputes the same cp_id internally for the actual pickup.
        cp_id = env._pick_side_grasp_id(env.target_obj, arm_tag)
        gpose = env._geometric_grasp_pose(env.target_obj, cp_id, pre_dis=0.0) if cp_id is not None else None
        grasp_q = np.array(gpose[-4:]) if gpose is not None else np.array([0, 1, 0, 0], float)

        ok = _pickup_only(env)
        print(f"[seed {seed}] pickup {'OK' if ok else 'FAILED'} (arm={arm_tag})")

        XX = YY = succ = None
        if ok:
            XX, YY, succ = _reach_grid(env, arm_tag, grasp_q, args)
            print(f"[seed {seed}] reachable cells: {int(succ.sum())}/{succ.size}")
        _plot_seed(seed, ok, XX, YY, succ, box_p, tgt_p, np.array(PAD_XY), str(arm_tag), args)

        try:
            env.close_env()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="bench_demo_office_clean")
    ap.add_argument("--seeds", default="1,2,3,4,5", help="comma list (1,2,3) or range (10-20)")
    ap.add_argument("--offset", type=float, default=0.2, help="occluder offset in front of target (m)")
    ap.add_argument("--z", type=float, default=0.90, help="EE height for the grid (m); table top ~0.74")
    ap.add_argument("--xmin", type=float, default=-0.6)
    ap.add_argument("--xmax", type=float, default=0.6)
    ap.add_argument("--ymin", type=float, default=-0.35)
    ap.add_argument("--ymax", type=float, default=0.35)
    ap.add_argument("--res", type=float, default=0.02, help="grid resolution (m)")
    ap.add_argument("--chunk", type=int, default=256, help="IK poses per batch; lower on CUDA OOM")
    ap.add_argument("--out-dir", default="../scripts/validation/results/pickup_reachability",
                    help="repo-root results location (same convention as other bench scripts)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
