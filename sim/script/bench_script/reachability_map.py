#!/usr/bin/env python3
"""
Collision-free reachability map via curobo batched IK (issue #35, subgoal search).

Builds the occluder scene (milk-box in the collision world), then for a grid of
(x, y) GRIPPER positions at a fixed height and orientation, batch-solves
collision-free IK per arm and plots which cells are reachable. Reachable ==
curobo found a joint solution within limits, self-collision-free AND world-
collision-free (box + table). Overlays the box footprint, target and pad.

This tells you where a SUBGOAL may legally sit (a valid collision-free EE pose),
NOT whether a path exists between two cells -- but a chain of nearby reachable
cells from behind the box to the pad is what trajopt can actually connect.

The milk-box occluder is added to the collision world by default; pass --no-occluder
to map reachability on the bare (table-only) world instead.

USAGE (from the benchmark folder, env sourced):
    python reachability_map.py --seed 1 --offset 0.2 --res 0.02
    python reachability_map.py --arms right --z 0.90
    python reachability_map.py --no-occluder            # bare table, no milk box
    python reachability_map.py --volume                 # 3D sweep -> band/area/slices/isosurface

--volume sweeps z (--zmin/--zmax/--zres) into a boolean volume R(x,y,z), caches it to .npz,
and derives figures: (A) a ceiling z_max(x,y) 3D surface, (A2) a flat 2D z_max heatmap with a
contrast-stretched turbo colour map, (B) reachable-area-vs-z curve, (C) a per-z slice montage,
(D) a marching-cubes isosurface of the reachable region. For a rotatable version of the 3D
views, run reachability_view.py on the cached .npz (needs only numpy+matplotlib, no GPU).

NOTE: reachability is orientation-specific. By default we use the target's own
horizontal side-grasp orientation (per arm). Pass --topdown for a top-down quat.
"""

import argparse
from pathlib import Path

import numpy as np
import transforms3d as t3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from setup_paths import setup_paths
setup_paths()

# repo-root results dir, anchored to THIS file so it resolves the same from any cwd
# (bench_script -> script -> sim -> RoboPRO)
RESULTS_DIR = Path(__file__).resolve().parents[3] / "scripts" / "validation" / "results" / "reachability"

import torch  # noqa: E402  (after setup_paths so curobo's torch is on the path)
from analyze_occluder_visibility import make_occluder_task, PAD_XY, OCC_HALF_FOOTPRINT  # noqa: E402
from analyze_natural_visibility import build_cfg, DR_CLEAN  # noqa: E402


# ----------------------------------------------------------------------------- frames
def _world_gripper_to_curobo(robot, planner, arm_tag, gripper_pose):
    """Replicate robot.left/right_plan_path + planner.plan_path frame chain for a single
    WORLD gripper pose [x,y,z,qw,qx,qy,qz] -> (pos3, quat4) in curobo's IK frame.
    (aloha-agilex branch, since the bench uses that embodiment.)"""
    # 1) gripper -> endlink (robot frame convention)
    endlink = robot._trans_from_gripper_to_endlink(gripper_pose, arm_tag=str(arm_tag))
    # 2) world -> arm base
    world_base = np.concatenate([np.array(planner.robot_origion_pose.p),
                                 np.array(planner.robot_origion_pose.q)])
    world_target = np.concatenate([np.array(endlink.p), np.array(endlink.q)])
    p, q = planner._trans_from_world_to_base(world_base, world_target)
    # 3) frame_bias + small per-arm yaw (aloha-agilex patch, see planner.plan_path)
    T_target = t3d.affines.compose(p, t3d.quaternions.quat2mat(q), [1, 1, 1])
    T_bias = t3d.affines.compose(planner.frame_bias, np.eye(3), [1, 1, 1])
    rot = t3d.axangles.axangle2mat([0, 0, 1], -0.02 if str(arm_tag) == "left" else -0.01)
    T_rot = t3d.affines.compose([0, 0, 0], rot, [1, 1, 1])
    T_new = T_rot @ T_bias @ T_target
    return T_new[:3, 3], t3d.quaternions.mat2quat(T_new[:3, :3])


def _build_ik_solver(planner):
    """Fresh IKSolver that REUSES the planner's already-loaded collision world, with
    use_cuda_graph=False so we can pass an arbitrary-size batch (the motion_gen ik_solver
    locks its batch size via cuda-graph, so we can't reuse it directly)."""
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    mg = planner.motion_gen
    cfg = IKSolverConfig.load_from_robot_config(
        planner.yml_path,
        None,                                   # world comes from the shared checker below
        tensor_args=mg.tensor_args,
        world_coll_checker=mg.world_coll_checker,   # <- already has the box + table
        use_cuda_graph=False,
        self_collision_check=True,
        self_collision_opt=False,
    )
    return IKSolver(cfg)


def _solve_grid(robot, planner, ik, arm_tag, gp_world, chunk=256):
    """gp_world: (N,7) world gripper poses -> (N,) bool reachable+collision-free.
    Solved in chunks: batched IK spawns many seeds per pose, so the whole grid at once
    is a multi-GB allocation -> chunk to cap peak memory (safe since use_cuda_graph=False)."""
    from curobo.types.math import Pose as CuroboPose
    ta = planner.motion_gen.tensor_args
    N = len(gp_world)
    pos = np.empty((N, 3), dtype=np.float32)
    quat = np.empty((N, 4), dtype=np.float32)
    for i, gp in enumerate(gp_world):
        p, q = _world_gripper_to_curobo(robot, planner, arm_tag, gp)
        pos[i], quat[i] = p, q
    out = np.zeros(N, dtype=bool)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        goal = CuroboPose(position=ta.to_device(pos[s:e]), quaternion=ta.to_device(quat[s:e]))
        result = ik.solve_batch(goal)
        out[s:e] = result.success.detach().cpu().numpy().reshape(e - s, -1)[:, 0].astype(bool)
        del result, goal
        torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------- run
def run(args):
    env = make_occluder_task()()
    env.spawn_occluder = args.occluder     # --no-occluder -> empty (table-only) collision world
    env.occluder_offset = args.offset
    env.setup_demo(**build_cfg("put_mouse_on_pad", args.base_config, args.seed, DR_CLEAN))

    # box_p is None when the occluder is disabled; every occluder-specific step below guards on it
    box_p = np.array(env.occluder.get_pose().p) if getattr(env, "occluder", None) is not None else None
    tgt_p = np.array(env.target_obj.get_pose().p)
    pad_xy = np.array(PAD_XY)
    box_txt = f"box=({box_p[0]:.3f},{box_p[1]:.3f})  " if box_p is not None else "box=OFF  "
    print(f"{box_txt}target=({tgt_p[0]:.3f},{tgt_p[1]:.3f})  pad={pad_xy}")

    arms = {"left": "left", "right": "right"} if args.arms == "both" else {args.arms: args.arms}

    # grid
    xs = np.arange(args.xmin, args.xmax + 1e-9, args.res)
    ys = np.arange(args.ymin, args.ymax + 1e-9, args.res)
    XX, YY = np.meshgrid(xs, ys)                      # (ny, nx)

    # single slice (default) is just a 1-level "volume"; --volume sweeps z
    if args.volume:
        zs = np.arange(args.zmin, args.zmax + 1e-9, args.zres)
    else:
        zs = np.array([args.z], dtype=float)
    z_check = float(zs[len(zs) // 2])                 # representative height for self-checks

    reach_any = np.zeros((len(zs),) + XX.shape, dtype=bool)   # (nz, ny, nx)
    per_arm = {}
    for name in arms:
        planner = env.robot.left_planner if name == "left" else env.robot.right_planner

        # orientation: the target's horizontal side grasp for this arm (or top-down). Fixed across
        # the whole grid/volume -> the map answers "reachable WITH THIS grasp", not "in any pose".
        if args.topdown:
            grasp_q = np.array([0, 1, 0, 0], dtype=float)      # gripper pointing down
            grasp_pose = None
        else:
            cp_id = env._pick_side_grasp_id(env.target_obj, name)
            grasp_pose = env._geometric_grasp_pose(env.target_obj, cp_id, pre_dis=0.0) if cp_id is not None else None
            grasp_q = np.array(grasp_pose[-4:]) if grasp_pose is not None else np.array([0, 1, 0, 0], dtype=float)

        ik = _build_ik_solver(planner)

        # --- self-checks: real grasp pose should be reachable; box centre should not ---
        if grasp_pose is not None:
            chk = _solve_grid(env.robot, planner, ik, name, np.array([grasp_pose]), chunk=args.chunk)
            print(f"[{name}] self-check grasp pose reachable = {bool(chk[0])}  (expect True)")
        if box_p is not None:      # box-centre should be blocked; meaningless with no occluder
            inbox = np.array([[box_p[0], box_p[1], z_check, *grasp_q]])
            chk2 = _solve_grid(env.robot, planner, ik, name, inbox, chunk=args.chunk)
            print(f"[{name}] self-check box-centre reachable = {bool(chk2[0])}  (expect False)")

        vol = np.zeros((len(zs),) + XX.shape, dtype=bool)
        for zi, z in enumerate(zs):
            gp = np.zeros((XX.size, 7))
            gp[:, 0] = XX.ravel(); gp[:, 1] = YY.ravel(); gp[:, 2] = z
            gp[:, 3:] = grasp_q
            vol[zi] = _solve_grid(env.robot, planner, ik, name, gp, chunk=args.chunk).reshape(XX.shape)
            if args.volume:
                print(f"[{name}] z={z:.3f}: {int(vol[zi].sum())}/{vol[zi].size}")
        per_arm[name] = vol
        reach_any |= vol
        print(f"[{name}] reachable cells (all z): {int(vol.sum())}/{vol.size}")
        del ik
        torch.cuda.empty_cache()

    if args.volume:
        _save_and_plot_volume(XX, YY, xs, ys, zs, reach_any, per_arm, box_p, tgt_p, pad_xy, args)
    else:
        _plot(XX, YY, reach_any[0], {n: v[0] for n, v in per_arm.items()},
              box_p, tgt_p, pad_xy, float(zs[0]), args)
    try:
        env.close_env()
    except Exception:
        pass


def _plot(XX, YY, reach_any, per_arm, box_p, tgt_p, pad_xy, z, args):
    fig, ax = plt.subplots(figsize=(8, 7))
    extent = [XX.min(), XX.max(), YY.min(), YY.max()]
    ax.imshow(reach_any, origin="lower", extent=extent, cmap="Greens", alpha=0.8, aspect="equal")
    # box footprint (approx square of half-diagonal OCC_HALF_FOOTPRINT); skipped when off
    if box_p is not None:
        h = OCC_HALF_FOOTPRINT
        ax.add_patch(plt.Rectangle((box_p[0] - h, box_p[1] - h), 2 * h, 2 * h,
                                   fill=False, edgecolor="red", lw=2, label="occluder"))
    ax.plot(tgt_p[0], tgt_p[1], "b*", ms=16, label="target")
    ax.plot(pad_xy[0], pad_xy[1], "ms", ms=12, label="pad")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Collision-free reachability @ z={z:.2f}m, arms={args.arms}"
                 f"{' (top-down)' if args.topdown else ' (side-grasp quat)'}"
                 f"  |  occluder {'ON' if box_p is not None else 'OFF'}\n"
                 f"green = reachable+collision-free (curobo IK)")
    ax.legend(loc="upper right")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tag = "topdown" if args.topdown else "sidegrasp"
    occ = "occ" if box_p is not None else "noocc"
    p = out / f"reach_seed{args.seed}_off{args.offset}_z{z:.2f}_{args.arms}_{tag}_{occ}.png"
    fig.tight_layout(); fig.savefig(p, dpi=130)
    print(f"saved {p}")


# ------------------------------------------------------------------ 3D volume plots
def _save_and_plot_volume(XX, YY, xs, ys, zs, reach_any, per_arm, box_p, tgt_p, pad_xy, args):
    """Cache the boolean volume once, then derive every figure from it (all cheap)."""
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tag = "topdown" if args.topdown else "sidegrasp"
    occ = "occ" if box_p is not None else "noocc"
    stem = f"vol_seed{args.seed}_off{args.offset}_{args.arms}_{tag}_{occ}_z{zs[0]:.2f}-{zs[-1]:.2f}"

    npz = out / f"{stem}.npz"
    np.savez_compressed(npz,
                        reach_any=reach_any, xs=xs, ys=ys, zs=zs,
                        box_p=(box_p if box_p is not None else np.array([])),
                        tgt_p=tgt_p, pad_xy=pad_xy, occ_half=np.array(OCC_HALF_FOOTPRINT),
                        **{f"arm_{n}": v for n, v in per_arm.items()})
    print(f"saved {npz}")

    _plot_ceiling3d(XX, YY, zs, reach_any, out / f"{stem}_ceiling.png", box_p, tgt_p, pad_xy, args)
    _plot_ceiling_heatmap(XX, YY, zs, reach_any, out / f"{stem}_ceiling_heat.png", box_p, tgt_p, pad_xy, args)
    _plot_area_curve(zs, reach_any, per_arm, out / f"{stem}_area.png", args.res, box_p, args)
    _plot_slice_montage(XX, YY, zs, reach_any, out / f"{stem}_slices.png", box_p, tgt_p, pad_xy, args)
    _plot_isosurface(xs, ys, zs, reach_any, out / f"{stem}_iso.png", box_p, tgt_p, pad_xy, args)
    print(f"[interactive] rotate the 3D view with:  python reachability_view.py {npz}")


def _column_bounds(reach_any, zs):
    """Per (x,y): lowest and highest reachable z (NaN where the column is never reachable).
    A vertical line through the arm's shell-shaped workspace enters once and exits once, so
    [z_floor, z_ceil] is an (almost always) lossless summary of the column."""
    any_reach = reach_any.any(axis=0)                                   # (ny,nx)
    floor_idx = np.argmax(reach_any, axis=0)                            # first True from bottom
    ceil_idx = reach_any.shape[0] - 1 - np.argmax(reach_any[::-1], axis=0)  # first True from top
    z_floor = zs[floor_idx].astype(float); z_floor[~any_reach] = np.nan
    z_ceil = zs[ceil_idx].astype(float);   z_ceil[~any_reach] = np.nan
    return z_floor, z_ceil


def _plot_ceiling3d(XX, YY, zs, reach_any, path, box_p, tgt_p, pad_xy, args):
    """(A) ceiling z_max(x,y) as a single height-coloured 3D surface (floor dropped).
    Static PNG; for a rotatable version run:  python reachability_view.py <cache.npz> --kind ceiling
    Colour is stretched to the actual ceiling min..max (turbo) to emphasise height differences."""
    _, z_ceil = _column_bounds(reach_any, zs)
    if not np.isfinite(z_ceil).any():
        print("ceiling3d skipped: nothing reachable"); return
    vmin, vmax = float(np.nanmin(z_ceil)), float(np.nanmax(z_ceil))
    norm = plt.Normalize(vmin, vmax)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(XX, YY, z_ceil, cmap="turbo", norm=norm, linewidth=0,
                    antialiased=True, rcount=80, ccount=80)
    ax.plot([tgt_p[0]] * 2, [tgt_p[1]] * 2, [vmin, vmax], color="k", lw=2, label="target")
    ax.scatter([pad_xy[0]], [pad_xy[1]], [vmin], color="magenta", s=40, marker="s", label="pad")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("max reachable z (m)")
    ax.set_zlim(vmin, vmax); ax.view_init(elev=28, azim=-60); ax.legend(loc="upper left")
    sm = plt.cm.ScalarMappable(norm=norm, cmap="turbo"); sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.6, label="max reachable z (m)")
    ax.set_title(f"Ceiling z_max(x,y)  |  occluder {'ON' if box_p is not None else 'OFF'}, arms={args.arms}")
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"saved {path}")


def _plot_ceiling_heatmap(XX, YY, zs, reach_any, path, box_p, tgt_p, pad_xy, args):
    """(A2) flat 2D z_max(x,y) heatmap with the 'clears the milk-box top' divider. The actual
    drawing lives in reachability_view.ceiling_heatmap (single source, so the standalone viewer
    regenerates the identical figure from the cache with no re-sweep)."""
    import reachability_view as rv   # light module (numpy/argparse only at import); safe under Agg
    fig, _ = rv.ceiling_heatmap(XX[0, :], YY[:, 0], zs, reach_any, box_p, tgt_p, pad_xy,
                                OCC_HALF_FOOTPRINT, args.arms)
    if fig is None:
        print("ceiling heatmap skipped: nothing reachable"); return
    fig.savefig(path, dpi=140); plt.close(fig)
    print(f"saved {path}")


def _plot_area_curve(zs, reach_any, per_arm, path, res, box_p, args):
    """(B) reachable area A(z) vs height -- the 'best working height' curve."""
    cell = res * res
    A_any = reach_any.sum(axis=(1, 2)) * cell
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(zs, A_any, "-o", color="tab:green", lw=2, label="either arm")
    for n, v in per_arm.items():
        ax.plot(zs, v.sum(axis=(1, 2)) * cell, "-", lw=1.5, alpha=0.8, label=f"{n} arm")
    zstar = float(zs[int(np.argmax(A_any))])
    ax.axvline(zstar, ls="--", color="gray", lw=1)
    ax.annotate(f"peak z={zstar:.2f}", (zstar, A_any.max()),
                textcoords="offset points", xytext=(6, -4), fontsize=9)
    ax.set_xlabel("EE height z (m)"); ax.set_ylabel("reachable area (m$^2$)")
    ax.set_title(f"Reachable area vs height  |  occluder {'ON' if box_p is not None else 'OFF'}, "
                 f"arms={args.arms}")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"saved {path}")


def _plot_slice_montage(XX, YY, zs, reach_any, path, box_p, tgt_p, pad_xy, args):
    """(C) honest, lossless flip-book: one 2D reachability slice per z level."""
    nz = len(zs)
    ncols = min(5, nz); nrows = int(np.ceil(nz / ncols))
    extent = [XX.min(), XX.max(), YY.min(), YY.max()]
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.7 * nrows), squeeze=False)
    for i in range(nrows * ncols):
        ax = axes[i // ncols][i % ncols]
        if i >= nz:
            ax.axis("off"); continue
        ax.imshow(reach_any[i], origin="lower", extent=extent, cmap="Greens",
                  vmin=0, vmax=1, aspect="equal")
        if box_p is not None:
            h = OCC_HALF_FOOTPRINT
            ax.add_patch(plt.Rectangle((box_p[0] - h, box_p[1] - h), 2 * h, 2 * h,
                                       fill=False, edgecolor="red", lw=1.2))
        ax.plot(tgt_p[0], tgt_p[1], "b*", ms=9)
        ax.plot(pad_xy[0], pad_xy[1], "ms", ms=7)
        ax.set_title(f"z={zs[i]:.2f}", fontsize=9); ax.tick_params(labelsize=6)
    fig.suptitle(f"Reachability slices  |  occluder {'ON' if box_p is not None else 'OFF'}, "
                 f"arms={args.arms}   (green = reachable+collision-free)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"saved {path}")


def _plot_isosurface(xs, ys, zs, reach_any, path, box_p, tgt_p, pad_xy, args):
    """(D) the reachable region as one solid blob (marching cubes) -- best 'show a newcomer' view."""
    if reach_any.sum() == 0:
        print("isosurface skipped: nothing reachable"); return
    if len(zs) < 2:
        print("isosurface skipped: need >= 2 z levels"); return
    from skimage import measure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    vol = reach_any.astype(np.float32)                        # (nz, ny, nx)
    try:                                                      # light smoothing -> a readable blob
        from scipy.ndimage import gaussian_filter
        vol = gaussian_filter(vol, sigma=0.6)
    except Exception:
        pass
    spacing = (float(zs[1] - zs[0]), float(ys[1] - ys[0]), float(xs[1] - xs[0]))
    try:
        verts, faces, _, _ = measure.marching_cubes(vol, level=0.5, spacing=spacing)
    except (ValueError, RuntimeError) as e:
        print(f"isosurface skipped: {e}"); return
    # verts columns are (z,y,x) in index*spacing -> shift onto world axes, reorder to (x,y,z)
    vz = verts[:, 0] + float(zs.min()); vy = verts[:, 1] + float(ys.min()); vx = verts[:, 2] + float(xs.min())
    tri = np.stack([vx, vy, vz], axis=1)[faces]               # (nfaces, 3, 3)
    zlo, zhi = float(tri[..., 2].min()), float(tri[..., 2].max())   # fit z to the blob, not the sweep
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(tri, alpha=0.35, facecolor="tab:green", edgecolor="none"))
    ax.plot([tgt_p[0]] * 2, [tgt_p[1]] * 2, [zlo, zhi], color="k", lw=2, label="target")
    ax.scatter([pad_xy[0]], [pad_xy[1]], [zlo], color="magenta", s=45, marker="s", label="pad")
    if box_p is not None:
        h = OCC_HALF_FOOTPRINT
        sq_x = [box_p[0] - h, box_p[0] + h, box_p[0] + h, box_p[0] - h, box_p[0] - h]
        sq_y = [box_p[1] - h, box_p[1] - h, box_p[1] + h, box_p[1] + h, box_p[1] - h]
        for zlvl in (zlo, zhi):
            ax.plot(sq_x, sq_y, [zlvl] * 5, color="red", lw=1)
    ax.set_xlim(float(xs.min()), float(xs.max())); ax.set_ylim(float(ys.min()), float(ys.max()))
    ax.set_zlim(zlo, zhi)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title(f"Reachable region (isosurface)  |  occluder {'ON' if box_p is not None else 'OFF'}, "
                 f"arms={args.arms}")
    ax.view_init(elev=24, azim=-58); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="bench_demo_office_clean",
                    help="bench_task_config/<name>.yml (same scene config as the occluder experiment)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--offset", type=float, default=0.2, help="occluder offset in front of target (m)")
    ap.add_argument("--no-occluder", dest="occluder", action="store_false",
                    help="do NOT add the milk-box occluder -> reachability on the bare (table-only) "
                         "collision world. By default the occluder IS spawned.")
    ap.set_defaults(occluder=True)
    ap.add_argument("--arms", choices=["both", "left", "right"], default="both")
    ap.add_argument("--z", type=float, default=0.90, help="EE height for the (single-slice) grid (m)")
    ap.add_argument("--volume", action="store_true",
                    help="sweep z into a 3D reachability volume and emit band/area/slices/isosurface "
                         "figures (+ .npz cache) instead of the single 2D slice")
    ap.add_argument("--zmin", type=float, default=0.78, help="--volume: lowest EE height (m)")
    ap.add_argument("--zmax", type=float, default=1.6, help="--volume: highest EE height (m)")
    ap.add_argument("--zres", type=float, default=0.03, help="--volume: z step (m)")
    ap.add_argument("--topdown", action="store_true", help="use a top-down quat instead of the side grasp")
    ap.add_argument("--xmin", type=float, default=-0.6)
    ap.add_argument("--xmax", type=float, default=0.6)
    ap.add_argument("--ymin", type=float, default=-0.35)
    ap.add_argument("--ymax", type=float, default=0.35)
    ap.add_argument("--res", type=float, default=0.02, help="grid resolution (m)")
    ap.add_argument("--chunk", type=int, default=256,
                    help="IK poses per batch; lower if you hit CUDA OOM (planners already use ~9GB)")
    ap.add_argument("--out-dir", default=str(RESULTS_DIR),
                    help="results location (default: repo-root scripts/validation/results/reachability, "
                         "resolved from the script path so it's the same from any cwd)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
