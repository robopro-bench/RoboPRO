#!/usr/bin/env python3
"""
Per-subgoal reachability maps along the FULL occluder rollout (issue #35).

For each seed we run the REAL expert rollout (OccluderTask.play_once -- around-box
waypoint -> grasp -> lift -> backward subgoals -> place). A hook fires just before every
subgoal move and logs the ACTUAL executed pose plus the EE pose right before it. Then,
for each logged subgoal, we compute a collision-free gripper IK reachability map (box +
table world; the held bottle is NOT modelled) at:

    * the ORIENTATION of that subgoal, and
    * the HEIGHT (z) of that subgoal,

and plot it with the occluder / target / pad, the "from" position (where the arm is right
before this step) and the subgoal target overlaid. Green = reachable+collision-free. This
is a step-by-step heuristic for "from where I am now, with the pose the next subgoal
needs, is that subgoal actually reachable?" -- one PNG per subgoal per seed.

Because we run the real rollout, the subgoal orientations/heights are the ones the plan
truly uses (not a static guess). Many rollouts fail partway; we plot every subgoal that
was reached ("as far as there are successes"). The overall SUCCESS/FAILED of the rollout
is written into every frame's title.

Reuses the IK solver / frame-transform / grid machinery from reachability_map.py.

NOTE ON THE COLLISION WORLD: grids are computed AFTER the rollout with the table enabled
(a single consistent world). During the real grasp/lift the table is momentarily disabled,
so those two low frames are slightly conservative near table height; the mid/high subgoals
(box_mid, pad_high, bottle, waypoint) are unaffected.

USAGE (from sim, env sourced):
    python subgoal_reachability_map.py --seeds 1,2,3 --offset 0.2
    python subgoal_reachability_map.py --seeds 10-20 --res 0.03 --chunk 256
"""

import argparse
from pathlib import Path

import os
# Reduce CUDA fragmentation OOM: the curobo planners already hold ~9GB, and building the
# IK solver on top of the finished rollout fragments the pool. expandable_segments lets the
# allocator grow/reuse instead of failing on a fragmented arena. Set BEFORE torch/cuda init.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from setup_paths import setup_paths
setup_paths()

import torch  # noqa: E402
from analyze_occluder_visibility import (make_occluder_task, PAD_XY, OCC_HALF_FOOTPRINT,  # noqa: E402
                                         save_rollout_video)
from analyze_natural_visibility import build_cfg, DR_CLEAN  # noqa: E402
# IK machinery (fresh cuda-graph-off solver, frame transform, chunked batch solve)
from reachability_map import _build_ik_solver, _solve_grid  # noqa: E402


def _parse_seeds(s):
    if "-" in s and "," not in s:
        a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def _is_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower())


def _reach_grid(env, arm_tag, ik, quat, z, args):
    """Gripper IK reachability grid for `arm_tag` at a fixed height z and orientation quat
    (the next subgoal's). Reuses a pre-built solver `ik`. On CUDA OOM the batch chunk is
    halved and the grid re-solved (peak memory scales with chunk, so this recovers instead
    of crashing). Returns (XX, YY, succ)."""
    name = str(arm_tag)
    planner = env.robot.left_planner if name == "left" else env.robot.right_planner
    xs = np.arange(args.xmin, args.xmax + 1e-9, args.res)
    ys = np.arange(args.ymin, args.ymax + 1e-9, args.res)
    XX, YY = np.meshgrid(xs, ys)                       # (ny, nx)
    gp = np.zeros((XX.size, 7))
    gp[:, 0] = XX.ravel(); gp[:, 1] = YY.ravel(); gp[:, 2] = z
    gp[:, 3:] = quat
    chunk = args.chunk
    while True:
        try:
            torch.cuda.empty_cache()
            succ = _solve_grid(env.robot, planner, ik, arm_tag, gp, chunk=chunk).reshape(XX.shape)
            return XX, YY, succ
        except Exception as e:
            if not _is_oom(e) or chunk <= 8:
                raise
            torch.cuda.empty_cache()
            chunk = max(8, chunk // 2)
            print(f"    CUDA OOM -> retrying this grid at chunk={chunk}")


def _cell_reachable(succ, XX, YY, x, y, res):
    """Nearest-grid-cell reachability for an arbitrary (x, y) -- used to report whether the
    subgoal target itself lands on a reachable cell. None if it falls outside the grid."""
    xs, ys = XX[0, :], YY[:, 0]
    if not (xs.min() - res <= x <= xs.max() + res and ys.min() - res <= y <= ys.max() + res):
        return None
    ix = int(np.argmin(np.abs(xs - x)))
    iy = int(np.argmin(np.abs(ys - y)))
    return bool(succ[iy, ix])


def _plot_subgoal(seed, step, name, ok_overall, XX, YY, succ, target_pose, from_ee,
                  box_p, tgt_p, pad_xy, arm_name, args):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(args.xmin, args.xmax); ax.set_ylim(args.ymin, args.ymax)
    ax.imshow(succ, origin="lower", extent=[args.xmin, args.xmax, args.ymin, args.ymax],
              cmap="Greens", alpha=0.8, aspect="equal", vmin=0, vmax=1)
    # scene
    h = OCC_HALF_FOOTPRINT
    ax.add_patch(plt.Rectangle((box_p[0] - h, box_p[1] - h), 2 * h, 2 * h,
                               fill=False, edgecolor="red", lw=2, label="occluder"))
    ax.plot(tgt_p[0], tgt_p[1], "b*", ms=15, label="target obj")
    ax.plot(pad_xy[0], pad_xy[1], "ms", ms=11, label="pad")
    # from (position right before this step) -> to (this subgoal target)
    if from_ee is not None:
        ax.plot(from_ee[0], from_ee[1], "kX", ms=13, mew=2, label="from (prev EE)")
        ax.annotate("", xy=(target_pose[0], target_pose[1]), xytext=(from_ee[0], from_ee[1]),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
    tgt_cell = _cell_reachable(succ, XX, YY, target_pose[0], target_pose[1], args.res)
    to_color = "blue" if tgt_cell else "red"
    ax.plot(target_pose[0], target_pose[1], "o", color=to_color, ms=13, mec="black",
            mew=1.5, label=f"-> {name} (target)", zorder=6)
    reach_txt = {True: "target REACHABLE", False: "target NOT reachable",
                 None: "target off-grid"}[tgt_cell]
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"seed {seed}  step {step}: '{name}' ({arm_name} arm)  "
                 f"rollout {'SUCCESS' if ok_overall else 'FAILED'}\n"
                 f"IK map @ z={target_pose[2]:.2f}m, subgoal orientation  |  "
                 f"green=reachable  ({int(succ.sum())}/{succ.size})  |  {reach_txt}")
    ax.legend(loc="upper right", fontsize=8); ax.set_aspect("equal")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tag = "ok" if ok_overall else "fail"
    p = out / f"subgoal_seed{seed:04d}_step{step:02d}_{name}_z{target_pose[2]:.2f}_{tag}.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    print(f"  saved {p}")


def _group_overlaps(subs, tol=0.03):
    """Group subgoals [(name, pose7), ...] that coincide in (x, y) within `tol` m (many do --
    forward/backward share the same x_side and pad/box/bottle y-lines, and grasp/lift share
    the bottle x,y). Returns [(indices, (x, y)), ...] in first-seen order."""
    groups = []
    for i, entry in enumerate(subs):
        x, y = entry[1][0], entry[1][1]
        for grp in groups:
            if abs(x - grp[1]) <= tol and abs(y - grp[2]) <= tol:
                grp[0].append(i)
                break
        else:
            groups.append([[i], x, y])
    return [(g[0], (g[1], g[2])) for g in groups]


def _plot_all_subgoals(seed, subs, box_p, tgt_p, pad_xy, args):
    """One top-down overview of ALL planned subgoals [(name, pose7), ...] for a seed (static,
    no rollout). Subgoals sharing an (x, y) are merged into one marker (orange) and their
    label lists every one -- each entry is '#idx name z=height'. Non-overlapping = blue. A
    faint grey line traces execution order."""
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    ax.set_xlim(args.xmin, args.xmax); ax.set_ylim(args.ymin, args.ymax)
    h = OCC_HALF_FOOTPRINT
    ax.add_patch(plt.Rectangle((box_p[0] - h, box_p[1] - h), 2 * h, 2 * h,
                               fill=False, edgecolor="red", lw=2, label="occluder"))
    ax.plot(tgt_p[0], tgt_p[1], "b*", ms=15, label="target obj")
    ax.plot(pad_xy[0], pad_xy[1], "ms", ms=11, label="pad")
    # execution-order path through the subgoal targets
    ax.plot([e[1][0] for e in subs], [e[1][1] for e in subs],
            "-", color="gray", lw=1, alpha=0.5, zorder=2)
    for idxs, (cx, cy) in _group_overlaps(subs):
        overlap = len(idxs) > 1
        color = "darkorange" if overlap else "tab:blue"
        ax.plot(cx, cy, "o", color=color, ms=11, mec="black", mew=1.2, zorder=5)
        lines = [f"#{i} {subs[i][0]} z={subs[i][1][2]:.2f}" for i in idxs]
        label = ("OVERLAP x%d\n" % len(idxs) if overlap else "") + "\n".join(lines)
        ax.annotate(label, (cx, cy), textcoords="offset points", xytext=(8, 6), fontsize=7,
                    color=color, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.85))
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"seed {seed}: all {len(subs)} planned subgoals (top-down, static)\n"
                 f"orange = subgoals overlapping in (x,y); label = #idx name z=height")
    ax.legend(loc="upper right", fontsize=8); ax.set_aspect("equal")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    p = out / f"subgoals_all_seed{seed:04d}.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    print(f"  saved {p}")


def run(args):
    requested = _parse_seeds(args.seeds)
    if not requested:
        print("no seeds requested"); return
    env = make_occluder_task()()
    env.spawn_occluder = True
    env.occluder_offset = args.offset
    print(f"seeds={requested} (want {len(requested)} STABLE)  offset={args.offset}  res={args.res}")

    # Produce len(requested) STABLE rollouts. A seed whose scene build is rejected (e.g. a
    # toppled bottle -> UnStableError from check_stable) is replaced by a fresh seed past the
    # requested set, so we always get that many rollouts on upright scenes. A rollout that
    # BUILDS fine but fails to plan still counts (we plot its failing subgoal) -- only build
    # rejections draw a replacement.
    pending = list(requested)
    draw = max(requested) + 1
    produced = 0
    attempts = 0
    max_attempts = len(requested) * 10 + 50   # safety cap: don't loop forever if builds keep failing
    while produced < len(requested) and attempts < max_attempts:
        attempts += 1
        if pending:
            seed = pending.pop(0)
        else:
            seed = draw
            draw += 1
        try:
            # rollout build; when --save-video, capture frames + set save_path so the
            # rollout mp4 lands in <out-dir>/video/episode{seed}.mp4 (ep_num = seed)
            cfg = build_cfg("put_mouse_on_pad", args.base_config, seed, DR_CLEAN,
                            rollout=True, ep_num=seed,
                            save_path=(args.out_dir if args.save_video else None))
            if not args.save_video:
                cfg["save_data"] = False   # execute the plan but skip frame capture (faster)
            env.setup_demo(**cfg)
        except Exception as e:
            print(f"[seed {seed}] scene build failed/rejected ({type(e).__name__}: {e}); "
                  f"drawing another seed")
            continue
        produced += 1

        box_p = np.array(env.occluder.get_pose().p)
        tgt_p = np.array(env.target_obj.get_pose().p)

        # overview: ALL planned subgoals, computed statically from the scene (no rollout).
        # Kept around so a FAILED rollout can also plot the subgoal it failed ON (the next
        # planned subgoal after the last one the hook reached -- see end of the loop below).
        planned = env._planned_subgoals()
        _plot_all_subgoals(seed, planned, box_p, tgt_p, np.array(PAD_XY), args)

        # log every subgoal the rollout actually reaches: (name, target_pose, from_ee, arm)
        log = []
        env.subgoal_hook = lambda name, target_pose, cur, arm_tag: log.append(
            (name, list(target_pose), (list(cur) if cur is not None else None), str(arm_tag)))

        try:
            env.play_once()
            ok_overall = bool(getattr(env, "plan_success", False))
        except Exception as e:
            print(f"[seed {seed}] rollout raised ({type(e).__name__}: {e}); "
                  f"plotting {len(log)} reached subgoal(s)")
            ok_overall = False
        env.subgoal_hook = None
        print(f"[seed {seed}] rollout {'SUCCESS' if ok_overall else 'FAILED'}; "
              f"{len(log)} subgoal(s) reached")
        if not log:
            _safe_close(env)
            if args.save_video:
                save_rollout_video(env, args.out_dir, seed)   # merge after close_env()
            continue

        # one consistent (table-on) world for all grids; one solver per arm for this seed
        try:
            env.enable_table(enable=True)
        except Exception:
            pass
        # release the rollout's cached buffers before allocating the IK solver
        torch.cuda.empty_cache()
        arm_name = log[0][3]
        planner = env.robot.left_planner if arm_name == "left" else env.robot.right_planner
        ik = _build_ik_solver(planner)
        for step, (name, target_pose, from_ee, arm_tag) in enumerate(log):
            quat = np.array(target_pose[3:])
            try:
                XX, YY, succ = _reach_grid(env, arm_tag, ik, quat, target_pose[2], args)
            except Exception as e:
                # unrecoverable even at the min chunk -> skip this frame, keep the run going
                lvl = "CUDA OOM" if _is_oom(e) else type(e).__name__
                print(f"  step {step} '{name}' grid failed ({lvl}); skipping frame")
                torch.cuda.empty_cache()
                continue
            print(f"  step {step} '{name}' z={target_pose[2]:.2f} "
                  f"reachable {int(succ.sum())}/{succ.size}")
            _plot_subgoal(seed, step, name, ok_overall, XX, YY, succ, target_pose, from_ee,
                          box_p, tgt_p, np.array(PAD_XY), arm_tag, args)

        # ONE extra IK map for the subgoal the rollout FAILED on. move() short-circuits after
        # the first failed plan, so the hook logged exactly the subgoals that were reached and
        # the failing one is the NEXT planned subgoal (planned[len(log)]). A SUCCESS reaches
        # every subgoal (len(log) == len(planned)) and needs no extra frame. The name guard
        # bails if the reached prefix doesn't line up with the planned order (e.g. a no-grasp
        # path), so we never emit a mislabelled frame.
        reached_names = [e[0] for e in log]
        if (not ok_overall and len(log) < len(planned)
                and reached_names == [p[0] for p in planned[:len(log)]]):
            fail_name, fail_pose = planned[len(log)]
            fail_from = log[-1][1] if log else None      # arm sits at the last reached target
            quat = np.array(fail_pose[3:])
            try:
                XX, YY, succ = _reach_grid(env, arm_name, ik, quat, fail_pose[2], args)
            except Exception as e:
                lvl = "CUDA OOM" if _is_oom(e) else type(e).__name__
                print(f"  FAILED subgoal '{fail_name}' grid failed ({lvl}); skipping frame")
            else:
                print(f"  FAILED subgoal '{fail_name}' z={fail_pose[2]:.2f} "
                      f"reachable {int(succ.sum())}/{succ.size}")
                _plot_subgoal(seed, len(log), f"FAILED_{fail_name}", ok_overall, XX, YY, succ,
                              list(fail_pose), fail_from, box_p, tgt_p, np.array(PAD_XY),
                              arm_name, args)
        del ik
        torch.cuda.empty_cache()
        _safe_close(env)
        if args.save_video:
            save_rollout_video(env, args.out_dir, seed)   # merge after close_env()

    if produced < len(requested):
        print(f"WARNING: only produced {produced}/{len(requested)} stable rollouts "
              f"after {attempts} attempts (hit safety cap)")


def _safe_close(env):
    try:
        env.close_env()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="bench_demo_office_clean")
    ap.add_argument("--seeds", default="1,2,3", help="comma list (1,2,3) or range (10-20)")
    ap.add_argument("--offset", type=float, default=0.2, help="occluder offset in front of target (m)")
    ap.add_argument("--xmin", type=float, default=-0.6)
    ap.add_argument("--xmax", type=float, default=0.6)
    ap.add_argument("--ymin", type=float, default=-0.35)
    ap.add_argument("--ymax", type=float, default=0.35)
    ap.add_argument("--res", type=float, default=0.03, help="grid resolution (m)")
    ap.add_argument("--chunk", type=int, default=64,
                    help="IK poses per batch; auto-halves on CUDA OOM. Lower this (e.g. 32/16) "
                         "if you still OOM, or coarsen --res to shrink the grid.")
    ap.add_argument("--out-dir", default="../scripts/validation/results/subgoal_reachability",
                    help="repo-root results location (same convention as other bench scripts)")
    ap.add_argument("--no-video", dest="save_video", action="store_false",
                    help="skip saving the rollout mp4 (faster: no frame capture). By default a "
                         "video per seed is written to <out-dir>/video/episode{seed}.mp4")
    ap.set_defaults(save_video=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
