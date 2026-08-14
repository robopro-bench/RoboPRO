#!/usr/bin/env python3
"""Interactive (rotatable) matplotlib viewer for a reachability volume cache.

Loads a .npz written by `reachability_map.py --volume` and opens a rotatable 3D window --
drag to orbit, scroll to zoom. This needs NO curobo / GPU / env, so you can copy the .npz
to any machine with a display and just run it. (Static PNGs are still written by the main
script; this is only for exploring the shape by hand.)

USAGE (from bench_script, on a machine with a display):
    python reachability_view.py                      # newest vol_*.npz in the results dir
    python reachability_view.py path/to/vol_*.npz    # a specific cache
    python reachability_view.py --kind ceiling       # z_max(x,y) surface instead of the solid blob
    python reachability_view.py --save shot.png      # also dump a PNG of the opening view

--kind iso     -> marching-cubes isosurface of the whole reachable region (default)
--kind ceiling -> the z_max(x,y) surface, turbo-coloured by height
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np

# repo-root results dir, anchored to THIS file so it resolves the same from any cwd
# (bench_script -> script -> sim -> RoboPRO)
RESULTS_DIR = Path(__file__).resolve().parents[3] / "scripts" / "validation" / "results" / "reachability"

# Milk-box 038 id2 at scale 1.0 is stood upright, so its VERTICAL height is the long axis of the
# model extents [0.110, 0.254, 0.122] -> 0.254 m (the other two dims are the 0.11x0.122 footprint).
# The box top height in world = box_p[2] (base, = table top ~0.742) + OCC_HEIGHT.
OCC_HEIGHT = 0.2542


def _iso_mesh(xs, ys, zs, reach_any, smooth=0.6):
    """Reachable-region isosurface as triangle verts (nfaces,3,3) in world (x,y,z); None if empty."""
    from skimage import measure
    if reach_any.sum() == 0 or len(zs) < 2:
        return None
    vol = reach_any.astype(np.float32)
    try:                                     # light smoothing -> a readable blob (optional)
        from scipy.ndimage import gaussian_filter
        if smooth:
            vol = gaussian_filter(vol, sigma=smooth)
    except Exception:
        pass
    spacing = (float(zs[1] - zs[0]), float(ys[1] - ys[0]), float(xs[1] - xs[0]))
    try:
        verts, faces, _, _ = measure.marching_cubes(vol, level=0.5, spacing=spacing)
    except (ValueError, RuntimeError):
        return None
    vz = verts[:, 0] + float(zs.min()); vy = verts[:, 1] + float(ys.min()); vx = verts[:, 2] + float(xs.min())
    return np.stack([vx, vy, vz], axis=1)[faces]


def _ceiling(reach_any, zs):
    """z_max(x,y): highest reachable height per column (NaN where the column is never reachable)."""
    any_reach = reach_any.any(axis=0)
    ceil_idx = reach_any.shape[0] - 1 - np.argmax(reach_any[::-1], axis=0)
    zc = zs[ceil_idx].astype(float); zc[~any_reach] = np.nan
    return zc


def ceiling_heatmap(xs, ys, zs, reach_any, box_p, tgt_p, pad_xy, occ_half, arms="both"):
    """2D top-down z_max(x,y) heatmap (turbo, contrast-stretched so small height differences pop),
    with iso-height contours and -- when the occluder is on -- a BOLD divider at z_max = box top:
    on the high side the gripper clears the milk box, on the low side it can't.
    Returns (fig, box_top) or (None, None) if nothing is reachable. Shared by reachability_map.py."""
    import matplotlib.pyplot as plt
    XX, YY = np.meshgrid(xs, ys)
    zc = _ceiling(reach_any, zs)
    masked = np.ma.masked_invalid(zc)
    if masked.count() == 0:
        return None, None
    vmin, vmax = float(masked.min()), float(masked.max())
    box_top = (float(box_p[2]) + OCC_HEIGHT) if box_p is not None else None

    cmap = plt.get_cmap("turbo").copy(); cmap.set_bad("0.85")
    extent = [float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())]
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    im = ax.imshow(masked, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="equal", interpolation="nearest")
    filled = np.ma.filled(masked, np.nan)
    if vmax - vmin > 1e-6:                       # faint iso-height contours for the gradient
        ax.contour(XX, YY, filled, levels=np.linspace(vmin, vmax, 8),
                   colors="k", linewidths=0.5, alpha=0.3)
    fig.colorbar(im, ax=ax, shrink=0.85, label="max reachable z (m)")

    # BOLD divider: where the reachable ceiling exactly equals the box top
    divider_note = ""
    if box_top is not None:
        if vmin < box_top < vmax:
            cs = ax.contour(XX, YY, filled, levels=[box_top], colors="k", linewidths=2.6)
            import matplotlib.patheffects as pe
            halo = [pe.withStroke(linewidth=4.5, foreground="w")]
            try:                                  # mpl>=3.8: ContourSet is itself a Collection
                cs.set_path_effects(halo)
            except AttributeError:                # older mpl: iterate member collections
                for c in cs.collections:
                    c.set_path_effects(halo)
            divider_note = "black line = gripper just clears box top"
        elif box_top <= vmin:
            divider_note = "whole reachable region clears the box top"
        else:
            divider_note = "NO reachable cell clears the box top"

    if box_p is not None:
        h = occ_half
        ax.add_patch(plt.Rectangle((box_p[0] - h, box_p[1] - h), 2 * h, 2 * h,
                                   fill=False, edgecolor="w", lw=2, label="occluder"))
    ax.plot(tgt_p[0], tgt_p[1], "w*", ms=15, mec="k", label="target")
    ax.plot(pad_xy[0], pad_xy[1], "ws", ms=11, mec="k", label="pad")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.legend(loc="upper right", fontsize=8)

    ttl = (f"Max reachable height z_max(x,y)  |  occluder {'ON' if box_p is not None else 'OFF'}, "
           f"arms={arms}\ncolour stretched to {vmin:.2f}-{vmax:.2f} m   (grey = unreachable)")
    ax.set_title(ttl)
    if box_top is not None:
        # write the milk-box height + top into the figure so the divider is self-explanatory
        txt = (f"milk box: height {OCC_HEIGHT:.3f} m, top at z = {box_top:.3f} m\n{divider_note}")
        ax.text(0.5, -0.13, txt, transform=ax.transAxes, ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="0.95", ec="0.6"))
    fig.tight_layout()
    return fig, box_top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?", default=None, help="cache file; default = newest vol_*.npz in --dir")
    ap.add_argument("--dir", default=str(RESULTS_DIR),
                    help="where to look for vol_*.npz when no path is given (default: repo-root "
                         "scripts/validation/results/reachability, resolved from the script path)")
    ap.add_argument("--kind", choices=["iso", "ceiling", "heat"], default="iso",
                    help="iso = 3D reachable blob, ceiling = 3D z_max surface, "
                         "heat = 2D z_max heatmap with the 'clears the box top' divider")
    ap.add_argument("--save", default=None, help="also save a PNG of the view to this path")
    args = ap.parse_args()

    path = args.npz
    if path is None:
        cands = sorted(glob.glob(os.path.join(args.dir, "vol_*.npz")), key=os.path.getmtime)
        if not cands:
            raise SystemExit(f"no vol_*.npz found in {args.dir} -- run reachability_map.py --volume first")
        path = cands[-1]
        print(f"using newest cache: {path}")

    d = np.load(path, allow_pickle=True)
    xs, ys, zs = d["xs"], d["ys"], d["zs"]
    reach_any = d["reach_any"]
    box_p = d["box_p"]; box_p = None if box_p.size == 0 else box_p
    tgt_p = d["tgt_p"]; pad_xy = d["pad_xy"]
    occ_half = float(d["occ_half"]) if "occ_half" in d.files else 0.04

    import matplotlib.pyplot as plt          # interactive backend (tkagg here); no Agg forced
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # 2D heatmap with the box-clearance divider -- regenerated straight from the cache (no sweep)
    if args.kind == "heat":
        fig, box_top = ceiling_heatmap(xs, ys, zs, reach_any, box_p, tgt_p, pad_xy, occ_half)
        if fig is None:
            raise SystemExit("nothing reachable to show in this cache")
        out = args.save or path.replace(".npz", "_ceiling_heat.png")
        fig.savefig(out, dpi=140); print(f"saved {out}")
        if box_top is not None:
            print(f"milk-box height = {OCC_HEIGHT:.3f} m, box top z = {box_top:.3f} m")
        plt.show()
        return

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    if args.kind == "iso":
        tri = _iso_mesh(xs, ys, zs, reach_any)
        if tri is None:
            raise SystemExit("nothing reachable to show in this cache")
        ax.add_collection3d(Poly3DCollection(tri, alpha=0.35, facecolor="tab:green", edgecolor="none"))
        zlo, zhi = float(tri[..., 2].min()), float(tri[..., 2].max())
        title = "Reachable region (isosurface)"
    else:
        XX, YY = np.meshgrid(xs, ys)
        zc = _ceiling(reach_any, zs)
        if not np.isfinite(zc).any():
            raise SystemExit("nothing reachable to show in this cache")
        zlo, zhi = float(np.nanmin(zc)), float(np.nanmax(zc))
        norm = plt.Normalize(zlo, zhi)
        ax.plot_surface(XX, YY, zc, cmap="turbo", norm=norm, linewidth=0, antialiased=True,
                        rcount=80, ccount=80)
        sm = plt.cm.ScalarMappable(norm=norm, cmap="turbo"); sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.6, label="max reachable z (m)")
        title = "Max reachable height z_max(x,y)"

    ax.plot([tgt_p[0]] * 2, [tgt_p[1]] * 2, [zlo, zhi], color="k", lw=2, label="target")
    ax.scatter([pad_xy[0]], [pad_xy[1]], [zlo], color="magenta", s=45, marker="s", label="pad")
    if box_p is not None:
        h = occ_half
        sq_x = [box_p[0] - h, box_p[0] + h, box_p[0] + h, box_p[0] - h, box_p[0] - h]
        sq_y = [box_p[1] - h, box_p[1] - h, box_p[1] + h, box_p[1] + h, box_p[1] - h]
        for zlvl in (zlo, zhi):
            ax.plot(sq_x, sq_y, [zlvl] * 5, color="red", lw=1)
    ax.set_xlim(float(xs.min()), float(xs.max())); ax.set_ylim(float(ys.min()), float(ys.max()))
    ax.set_zlim(zlo, zhi)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title(f"{title}   (drag to rotate)")
    ax.legend(loc="upper left")

    if args.save:
        fig.savefig(args.save, dpi=140)
        print(f"saved {args.save}")
    plt.show()


if __name__ == "__main__":
    main()
