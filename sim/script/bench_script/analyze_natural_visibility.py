"""
Phase 1 (#30): natural visibility distribution / rejection-sampling baseline.

Sweeps seeds x obstacle_density for an office task and measures the t=0
countertop `visible_fraction` of the target (no rollout), reusing the Phase 0
primitive (#28). Produces:

  - a JSONL of per-(seed, density) records,
  - per-density histograms of visible_fraction with the bucket boundaries marked,
  - a per-bucket readout (natural mass / acceptance rate) classifying each bucket
    as selectable-by-rejection vs rare/empty -> needs-occluder.

DENOMINATOR CHOICE (documented):
  visible_fraction = visible_target_px(scene at density D) / full_target_px(clean scene),
  both on the countertop camera, for the SAME seed. The denominator is captured
  from a clean build (`cluttered_table=False`, density 0). This is valid because
  the target pose is set in load_actors() *before* any clutter is generated and
  np.random is seeded first, so the target sits at the same pose for every density
  of a given seed (verified by asserting pose equality). The clean count still
  includes any static-furniture / gripper occlusion present at t=0, so the
  fraction isolates the marginal effect of table clutter.

USAGE (run from the benchmark folder):
    cd benchmark
    source set_env.sh
    python script/bench_script/analyze_natural_visibility.py \
        put_mouse_on_pad --bench-subdir office \
        --base-config bench_demo_office_d8 \
        --seed-start 0 --num-seeds 50 \
        --densities 8,12,15 --bins 20 \
        --out-dir ./visibility_phase1

    # Re-plot from an existing sweep without rebuilding scenes:
    python script/bench_script/analyze_natural_visibility.py --plot-only \
        --out-dir ./visibility_phase1
"""
import os
import sys
import json
import argparse
import shutil
from pathlib import Path

import numpy as np

from setup_paths import setup_paths
setup_paths()
bench_root = Path(os.environ["BENCH_ROOT"])
sim_root = Path(os.environ["SIM_ROOT"])
os.chdir(sim_root)

import yaml
from envs import CONFIGS_PATH
from visualize_task_scene import get_env_class, get_embodiment_config

CAMERA = "countertop_camera"
TARGET_ATTRS = ("target_obj", "target", "mouse")

# Bucket taxonomy (must match DEFAULT_VISIBILITY_BUCKETS in envs/_base_task.py):
#   not_visible        frac == 0
#   heavily_occluded   0    < frac < 0.20
#   mostly_occluded    0.20 <= frac < 0.5
#   partially_occluded 0.5  <= frac < 0.9
#   fully_visible      0.9  <= frac
BUCKET_ORDER = ["not_visible", "heavily_occluded", "mostly_occluded",
                "partially_occluded", "fully_visible"]
BUCKET_BOUNDARIES = [0.20, 0.5, 0.9]  # interior guide lines (plus 0 for not_visible)
BUCKET_COLORS = {
    "not_visible": "#C44E52",        # red
    "heavily_occluded": "#DD8452",   # orange
    "mostly_occluded": "#CCB974",    # yellow
    "partially_occluded": "#55A868", # green
    "fully_visible": "#4C72B0",      # blue
}


def _resolve_target(env):
    for attr in TARGET_ATTRS:
        if hasattr(env, attr):
            return getattr(env, attr)
    raise SystemExit("Could not find a target actor attribute on the env")


def build_cfg(task_name, base_config, seed, dr_overrides, rollout=False, ep_num=0, save_path=None):
    """Build a setup_demo cfg.

    rollout=False (default): a fast t=0 measurement build -- no planning, no saved
    data, single-camera render (measurement_only).
    rollout=True: an expert curobo rollout build -- need_plan=True + save_data=True
    so play_once() plans/executes and frames are captured for merge_pkl_to_hdf5_video()
    (writes <save_path>/video/episode{ep_num}.mp4). Full render (not measurement_only).
    """
    from script.task_loader import bench_config_path
    config_path = bench_config_path(base_config)
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    cfg["task_name"] = task_name
    cfg["render_freq"] = 0
    cfg["now_ep_num"] = int(ep_num)
    cfg["seed"] = int(seed)
    if rollout:
        cfg["need_plan"] = True        # expert curobo plans + executes the task
        cfg["save_data"] = True        # capture frames so a video can be written
        cfg["measurement_only"] = False
        cfg.setdefault("save_freq", 15)
        if save_path is not None:
            cfg["save_path"] = str(save_path)
    else:
        cfg["need_plan"] = False       # no planning/rollout for a t=0 measurement
        cfg["save_data"] = False
        cfg["measurement_only"] = True  # t=0 measurement: render only the measured camera

    cfg.setdefault("domain_randomization", {})
    cfg["domain_randomization"].update(dr_overrides)

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
    return cfg


# Denominator: clean (no table clutter), density 0.
DR_CLEAN = {"cluttered_table": False, "obstacle_density": 0, "clean_background_rate": 0}


def dr_dense(density, tall_only=False, handcrafted=False):
    dr = {"cluttered_table": True, "obstacle_density": int(density), "clean_background_rate": 0}
    if tall_only:
        dr["obstacle_tall_only"] = True
    if handcrafted:
        dr["obstacle_handcrafted"] = True
    return dr


def _bucket_rollout_artifacts(save_path, ep_num, success):
    base_dir = Path(save_path)
    bucket = "success" if success else "fail"
    bucket_dir = base_dir / bucket
    data_dir = bucket_dir / "data"
    video_dir = bucket_dir / "video"
    data_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    src_hdf5 = base_dir / "data" / f"episode{ep_num}.hdf5"
    src_mp4 = base_dir / "video" / f"episode{ep_num}.mp4"
    dst_hdf5 = data_dir / f"episode{ep_num}.hdf5"
    dst_mp4 = video_dir / f"episode{ep_num}.mp4"

    if src_hdf5.exists():
        shutil.move(str(src_hdf5), str(dst_hdf5))
    if src_mp4.exists():
        shutil.move(str(src_mp4), str(dst_mp4))

    return {
        "bucket": bucket,
        "hdf5_relpath": f"{bucket}/data/episode{ep_num}.hdf5",
        "video_relpath": f"{bucket}/video/episode{ep_num}.mp4",
    }


def run_rollout(env, task_name, base_config, seed, dr_overrides, save_path, ep_num):
    """Run one expert curobo rollout on the SAME scene (same seed -> same poses)
    and report whether it solved the task. A video of the attempt is written to
    <save_path>/video/episode{ep_num}.mp4 via the save_data + merge machinery.

    Returns a dict containing the success flag and relative artifact paths. A
    planning/execution failure or any exception counts as a failed rollout,
    which is exactly what we want to measure.
    """
    success = False
    artifact_info = None
    try:
        env.setup_demo(**build_cfg(task_name, base_config, seed, dr_overrides,
                                   rollout=True, ep_num=ep_num, save_path=str(save_path)))
        # Capture the initial state so early planning failures still leave behind
        # a minimal HDF5/MP4 artifact in success/ or fail/.
        if getattr(env, "save_data", False) and getattr(env, "FRAME_IDX", 0) == 0:
            env._take_picture()
        env.play_once()
        plan_ok = bool(getattr(env, "plan_success", False))
        check_ok = bool(env.check_success())
        if os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1":
            print(f"    [rollout seed {seed} ep{ep_num}] plan_success={plan_ok} check_success={check_ok}")
        success = plan_ok and check_ok
    except Exception as e:
        print(f"    [rollout seed {seed} ep{ep_num}] failed ({type(e).__name__}: {e})")
        success = False
    try:
        env.close_env()
    except Exception:
        pass
    # write the video only if this episode actually captured frames
    try:
        if getattr(env, "FRAME_IDX", 0) > 0:
            env.merge_pkl_to_hdf5_video()
            env.remove_data_cache()
            artifact_info = _bucket_rollout_artifacts(save_path, ep_num, success)
    except Exception as e:
        print(f"    [rollout seed {seed} ep{ep_num}] video merge failed ({type(e).__name__}: {e})")
    return {
        "success": bool(success),
        "artifact_info": artifact_info,
    }


def save_overlay(env, mask, out_path, header):
    """Save [countertop RGB | same view with the target's visible pixels in red],
    with a one-line header. Must be called while the render buffers are current
    (i.e. after measure_target_visibility, before close_env)."""
    import imageio
    import cv2
    # measurement_only renders only CAMERA, so request just it (avoids reading
    # the never-rendered wrist/other static cameras)
    rgb = env.cameras.get_rgb(camera_names=[CAMERA])[CAMERA]["rgb"]
    overlay = rgb.copy()
    overlay[mask] = (0.5 * np.array([255, 0, 0]) + 0.5 * overlay[mask]).astype(np.uint8)
    side = np.ascontiguousarray(np.concatenate([rgb, overlay], axis=1))
    cv2.putText(side, header, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 0, 0), 2, cv2.LINE_AA)
    imageio.imwrite(out_path, side)


def run_sweep(args):
    env_class = get_env_class(args.task_name, bench_subdir=args.bench_subdir)
    env = env_class()  # reuse one instance across builds (collect_data pattern)

    out_dir = effective_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "records.jsonl"
    img_dir = out_dir / "images"
    if args.save_images:
        img_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    densities = [int(d) for d in args.densities.split(",")]
    rollout = getattr(args, "rollout", False)
    ep_counter = 0   # unique episode id per rollout (-> video/episode{N}.mp4)

    print(f"task={args.task_name} base={args.base_config} camera={CAMERA}")
    print(f"seeds={seeds[0]}..{seeds[-1]} ({len(seeds)})  densities={densities}  rollout={rollout}")
    print(f"writing -> {jsonl_path}\n")

    def safe_close():
        try:
            env.close_env()
        except Exception:
            pass

    with open(jsonl_path, "w") as fout:
        for si, seed in enumerate(seeds):
            # --- denominator: clean build (skip seed if the build is unstable) ---
            try:
                env.setup_demo(**build_cfg(args.task_name, args.base_config, seed, DR_CLEAN))
                target = _resolve_target(env)
                clean_pose = np.array(target.actor.get_pose().p)
                res_clean = env.measure_target_visibility(target, camera_name=CAMERA)
                full_px = res_clean["visible_pixel_count"]
                if args.save_images:
                    save_overlay(env, res_clean["mask"], img_dir / f"seed{seed:04d}_clean.png",
                                 f"seed {seed} CLEAN (denominator)  full_px={full_px}")
            except Exception as e:
                print(f"[seed {seed}] clean build failed ({type(e).__name__}: {e}); skipping seed")
                safe_close()
                continue
            safe_close()

            if full_px <= 0:
                print(f"[seed {seed}] WARNING: full_target_px=0 on clean scene; skipping seed.")
                continue

            for density in densities:
                try:
                    env.setup_demo(**build_cfg(args.task_name, args.base_config, seed,
                                               dr_dense(density, args.tall_only, args.handcrafted)))
                    target = _resolve_target(env)
                    # guard: target pose must match the clean build for the denominator to be valid
                    pose_ok = bool(np.allclose(np.array(target.actor.get_pose().p), clean_pose, atol=1e-4))
                    res = env.measure_target_visibility(
                        target, camera_name=CAMERA, denominator=full_px,
                    )  # default DEFAULT_VISIBILITY_BUCKETS taxonomy
                    if args.save_images:
                        tag = "hand" if args.handcrafted else "tall" if args.tall_only else "mix"
                        save_overlay(
                            env, res["mask"],
                            img_dir / f"seed{seed:04d}_d{density:02d}_{tag}.png",
                            f"seed {seed} d={density} {tag}  vis_px={res['visible_pixel_count']} "
                            f"full={full_px} frac={res['visible_fraction']:.3f} "
                            f"{res['bucket']} pose_match={pose_ok}",
                        )
                except Exception as e:
                    print(f"[seed {seed} d{density}] build failed ({type(e).__name__}: {e}); skipping")
                    safe_close()
                    continue
                safe_close()

                rec = {
                    "seed": int(seed),
                    "density": int(density),
                    "full_px": int(full_px),
                    "visible_px": int(res["visible_pixel_count"]),
                    "visible_fraction": float(res["visible_fraction"]),
                    "bucket": res["bucket"],
                    "in_fov": bool(res["in_fov"]),
                    "pose_match": pose_ok,
                }
                # --- expert curobo rollout on the same scene (video + success) ---
                if rollout:
                    success = run_rollout(env, args.task_name, args.base_config, seed,
                                          dr_dense(density, args.tall_only, args.handcrafted),
                                          out_dir, ep_counter)
                    rec["rollout_success"] = bool(success)
                    rec["rollout_ep"] = ep_counter
                    rec["rollout_video"] = f"video/episode{ep_counter}.mp4"
                    print(f"    seed {seed} d={density} {res['bucket']}: "
                          f"rollout {'SUCCESS' if success else 'FAIL'} -> episode{ep_counter}.mp4")
                    ep_counter += 1
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
            print(f"[{si+1}/{len(seeds)}] seed {seed}: full_px={full_px} done")

    env.close_env()
    print(f"\nsweep complete -> {jsonl_path}")
    return jsonl_path


def load_records(out_dir):
    jsonl_path = Path(out_dir) / "records.jsonl"
    if not jsonl_path.exists():
        raise SystemExit(f"No records at {jsonl_path}; run the sweep first.")
    return [json.loads(l) for l in open(jsonl_path) if l.strip()]


def analyze(out_dir, bins, selectable_threshold, group_key="density",
            group_label="obstacle_density",
            suptitle="Natural visibility distribution on countertop camera",
            bar_title="Bucket proportions vs clutter density"):
    import csv
    import math as _m
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = load_records(out_dir)
    if not recs:
        print(f"no records found in {out_dir}; skipping analysis plots")
        return
    # Keep ALL records. Pose drift (clutter nudging the target during settling) can
    # push visible_fraction above 1 against the clean-build denominator; we clamp to
    # [0, 1] rather than dropping, so genuinely occluded/hidden scenes (the whole
    # point) are still counted. A clamped frac>1 just means "fully visible".
    n_drift = sum(1 for r in recs if not r.get("pose_match", True))
    n_over = sum(1 for r in recs if r["visible_fraction"] > 1.0)
    if n_drift or n_over:
        print(f"NOTE: {n_drift}/{len(recs)} records had target pose drift; "
              f"{n_over} had frac>1 -> clamped to 1.0 (kept, not dropped).")

    densities = sorted({r[group_key] for r in recs})
    buckets = BUCKET_ORDER
    edges = np.linspace(0.0, 1.0, bins + 1)   # equal-width bins over [0, 1]

    plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13})

    def clamped(r):
        return min(max(r["visible_fraction"], 0.0), 1.0)

    # Re-derive the bucket from the clamped visible_fraction (not the stored label),
    # so the current taxonomy applies even when re-plotting older runs.
    def classify(frac):
        if frac <= 0.0:
            return "not_visible"
        for name, hi in zip(buckets[1:], BUCKET_BOUNDARIES + [float("inf")]):
            if frac < hi:
                return name
        return buckets[-1]

    def fracs_for(d):
        return [clamped(r) for r in recs if r[group_key] == d]

    def bucket_counts(d):
        sub = [r for r in recs if r[group_key] == d]
        n = len(sub)
        counts = {b: 0 for b in buckets}
        for r in sub:
            counts[classify(clamped(r))] += 1
        return n, counts

    # ---- numeric equal-bin histogram table (stdout + CSV) ----
    print(f"\n================ Numeric histogram ({bins} equal bins over [0,1]) ================")
    csv_path = Path(out_dir) / "bin_counts.csv"
    with open(csv_path, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow([group_key, "bin_lo", "bin_hi", "count"])
        for d in densities:
            counts, _ = np.histogram(fracs_for(d), bins=edges)
            n = int(counts.sum())
            print(f"\n{group_label} = {d}  (n={n})")
            for i, c in enumerate(counts):
                lo, hi = edges[i], edges[i + 1]
                bar = "#" * int(c)
                print(f"  [{lo:4.2f}, {hi:4.2f}) {c:>4}  {bar}")
                writer.writerow([d, f"{lo:.4f}", f"{hi:.4f}", int(c)])
    print(f"\nsaved {csv_path}")

    # ---- histograms: equal bins, max 3 panels per row, annotated with bucket % ----
    ncols = min(3, len(densities))
    nrows = _m.ceil(len(densities) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 4.8 * nrows), squeeze=False)
    flat = axes.flatten()
    for ax, d in zip(flat, densities):
        n, counts = bucket_counts(d)
        ax.hist(fracs_for(d), bins=edges, color="#4C72B0", edgecolor="white")
        for x in BUCKET_BOUNDARIES:
            ax.axvline(x, color="0.4", ls="--", lw=1.5)
        ax.set_title(f"{group_label} = {d}   (n={n})")
        ax.set_xlabel("visible_fraction")
        ax.set_ylabel("seed count")
        ax.set_xlim(-0.03, 1.05)
        txt = "\n".join(f"{b}: {(counts[b] / n if n else 0):.0%}" for b in buckets)
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=11, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    for ax in flat[len(densities):]:
        ax.axis("off")
    fig.suptitle(suptitle + "\ndashed guides at bucket boundaries 0.25 / 0.5 / 0.9", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    hist_path = Path(out_dir) / "histograms.png"
    fig.savefig(hist_path, dpi=130)
    plt.close(fig)
    print(f"saved {hist_path}")

    # ---- dedicated pooled histogram of visible_fraction (all densities), bar counts labeled ----
    all_fr = [r["visible_fraction"] for r in recs]
    counts, _ = np.histogram(all_fr, bins=edges)
    fig3, ax3 = plt.subplots(figsize=(11, 5.5))
    centers = (edges[:-1] + edges[1:]) / 2
    ax3.bar(centers, counts, width=(edges[1] - edges[0]) * 0.95,
            color="#4C72B0", edgecolor="white")
    for x in BUCKET_BOUNDARIES:
        ax3.axvline(x, color="0.4", ls="--", lw=1.5)
    for c, cx in zip(counts, centers):
        if c > 0:
            ax3.text(cx, c, str(int(c)), ha="center", va="bottom", fontsize=10)
    ax3.set_xlabel("visible_fraction")
    ax3.set_ylabel("count (all seeds × densities)")
    ax3.set_xlim(-0.03, 1.05)
    ax3.set_xticks(np.round(edges, 2))
    ax3.tick_params(axis="x", labelrotation=90, labelsize=9)
    dens_txt = ",".join(str(d) for d in densities)
    ax3.set_title(f"Pooled visible_fraction histogram — {bins} equal bins  "
                  f"({group_label} {dens_txt}, n={len(all_fr)})\n"
                  "dashed guides at bucket boundaries 0.25 / 0.5 / 0.9")
    fig3.tight_layout()
    pooled_path = Path(out_dir) / "fraction_histogram.png"
    fig3.savefig(pooled_path, dpi=130)
    plt.close(fig3)
    print(f"saved {pooled_path}")

    # ---- summary: stacked bucket proportions vs density (headline figure) ----
    fig2, ax2 = plt.subplots(figsize=(max(8, 1.6 * len(densities) + 4), 5.8))
    x = np.arange(len(densities))
    bottom = np.zeros(len(densities))
    for b in buckets:
        vals = np.array([bucket_counts(d)[1][b] / (bucket_counts(d)[0] or 1) for d in densities])
        ax2.bar(x, vals, bottom=bottom, label=b, color=BUCKET_COLORS[b], edgecolor="white")
        for xi, (v, bo) in enumerate(zip(vals, bottom)):
            if v > 0.03:
                ax2.text(xi, bo + v / 2, f"{v:.0%}", ha="center", va="center",
                         color="white", fontsize=12, fontweight="bold")
        bottom += vals
    # uniform target: each of the 5 buckets would occupy 1/5; dotted lines mark the
    # cumulative boundaries (0.2/0.4/0.6/0.8). KL(observed || uniform) in base 5 lies
    # in [0,1] (0 = perfectly uniform, 1 = all mass in one bucket); annotate per bar.
    U = 1.0 / len(buckets)
    for yv in np.arange(U, 1.0, U):
        ax2.axhline(yv, color="0.25", ls=":", lw=1.2, zorder=5)
    for xi, d in enumerate(densities):
        n, counts = bucket_counts(d)
        props = [counts[b] / n if n else 0.0 for b in buckets]
        kl5 = sum(p * _m.log(p / U, 5) for p in props if p > 0)
        ax2.text(xi, 1.02, f"KL₅={kl5:.3f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(d) for d in densities])
    ax2.set_xlabel(group_label)
    ax2.set_ylabel("fraction of seeds")
    ax2.set_ylim(0, 1.12)
    ax2.set_title(bar_title + "   (dotted = uniform target, 0.2 each)", fontsize=14)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.26), ncol=3, frameon=False)
    fig2.tight_layout()
    summary_fig = Path(out_dir) / "bucket_proportions.png"
    fig2.savefig(summary_fig, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"saved {summary_fig}")

    # ---- per-bucket readout (selectable vs needs-occluder) ----
    summary = {}
    print("\n================ Per-bucket natural mass (acceptance rate) ================")
    header = f"{group_label[:8]:>8} | " + " | ".join(f"{b:>18}" for b in buckets) + " |  needs-occluder"
    print(header)
    print("-" * len(header))
    for d in densities:
        n, counts = bucket_counts(d)
        row = {"n": n, "buckets": {}}
        cells, needs = [], []
        for b in buckets:
            c = counts[b]
            frac = c / n if n else 0.0
            row["buckets"][b] = {"count": c, "fraction": frac,
                                 "status": ("selectable" if frac >= selectable_threshold
                                            else "rare" if frac > 0 else "empty")}
            cells.append(f"{c:>4} ({frac:>5.1%})")
            # any non-fully_visible bucket the natural distribution can't reach needs the occluder
            if b != "fully_visible" and frac < selectable_threshold:
                needs.append(b)
        summary[str(d)] = row
        print(f"{d:>8} | " + " | ".join(f"{c:>18}" for c in cells) + f" |  {', '.join(needs) or '-'}")

    print(f"\nselectable threshold = {selectable_threshold:.0%} of seeds; "
          f"non-fully_visible buckets below it need the occluder.")
    summary_path = Path(out_dir) / "bucket_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"buckets": buckets, "bins": bins,
                   "selectable_threshold": selectable_threshold,
                   "per_density": summary}, f, indent=2)
    print(f"saved {summary_path}")


def effective_out_dir(args):
    """When --rollout is on, write everything to <out-dir>_rollout so the rollout
    run never collides with a measurement-only run of the same out-dir."""
    suffix = "_rollout" if getattr(args, "rollout", False) else ""
    return Path(str(args.out_dir).rstrip("/") + suffix)


def _bucket_of(frac):
    """Re-derive the visibility bucket from a (clamped) visible_fraction, matching
    the taxonomy used in analyze()."""
    frac = min(max(frac, 0.0), 1.0)
    if frac <= 0.0:
        return "not_visible"
    for name, hi in zip(BUCKET_ORDER[1:], BUCKET_BOUNDARIES + [float("inf")]):
        if frac < hi:
            return name
    return BUCKET_ORDER[-1]


def analyze_success_by_bucket(out_dir):
    """Pooled observed P(rollout success | visibility bucket) over all records."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = [r for r in load_records(out_dir) if "rollout_success" in r]
    if not recs:
        print("no rollout records found; skipping success-by-bucket")
        return
    buckets = BUCKET_ORDER
    tot = {b: 0 for b in buckets}
    suc = {b: 0 for b in buckets}
    for r in recs:
        b = _bucket_of(r["visible_fraction"])
        tot[b] += 1
        suc[b] += 1 if r["rollout_success"] else 0
    rates = {b: (suc[b] / tot[b] if tot[b] else 0.0) for b in buckets}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(buckets))
    ax.bar(x, [rates[b] for b in buckets],
           color=[BUCKET_COLORS[b] for b in buckets], edgecolor="white")
    for xi, b in zip(x, buckets):
        label = f"{suc[b]}/{tot[b]}\n{rates[b]:.0%}" if tot[b] else "0/0"
        ax.text(xi, rates[b] + 0.02, label, ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets, rotation=20, ha="right")
    ax.set_ylabel("P(successful rollout)")
    ax.set_ylim(0, 1.12)
    ax.set_title(f"Observed rollout success rate per visibility bucket  (n={len(recs)})")
    fig.tight_layout()
    fig_path = Path(out_dir) / "rollout_success_by_bucket.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"saved {fig_path}")

    out = {b: {"total": tot[b], "success": suc[b], "success_rate": rates[b]} for b in buckets}
    json_path = Path(out_dir) / "rollout_success_by_bucket.json"
    with open(json_path, "w") as f:
        json.dump({"n": len(recs), "per_bucket": out}, f, indent=2)
    print(f"saved {json_path}")


def analyze_rollout(out_dir, bins, selectable_threshold, **analyze_kwargs):
    """Three rollout outputs: (1) the standard distribution over ALL records,
    (2) the same distribution over successful rollouts only (success_only/), and
    (3) the pooled P(success | bucket)."""
    out_dir = Path(out_dir)
    recs = load_records(out_dir)
    if not recs:
        print(f"no rollout records found in {out_dir}; skipping rollout analysis")
        return
    # (1) standard distribution, all records (the "same data")
    analyze(out_dir, bins, selectable_threshold, **analyze_kwargs)

    # (2) distribution over successful rollouts only
    succ = [r for r in recs if r.get("rollout_success")]
    succ_dir = out_dir / "success_only"
    succ_dir.mkdir(parents=True, exist_ok=True)
    with open(succ_dir / "records.jsonl", "w") as f:
        for r in succ:
            f.write(json.dumps(r) + "\n")
    print(f"\n[success-only] {len(succ)}/{len(recs)} rollouts succeeded; "
          f"rebuilding distribution on the success subset -> {succ_dir}")
    if succ:
        kw = dict(analyze_kwargs)
        if "suptitle" in kw:
            kw["suptitle"] += "  [successful rollouts only]"
        if "bar_title" in kw:
            kw["bar_title"] += "  [successful only]"
        analyze(succ_dir, bins, selectable_threshold, **kw)

    # (3) pooled P(success | bucket)
    analyze_success_by_bucket(out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_name", nargs="?", default="put_mouse_on_pad")
    ap.add_argument("--bench-subdir", default="office")
    ap.add_argument("--base-config", default="bench_demo_office_d8")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--num-seeds", type=int, default=50)
    ap.add_argument("--densities", default="30")
    ap.add_argument("--bins", type=int, default=20, help="equal-width histogram bins over [0,1]")
    ap.add_argument("--tall-only", action="store_true",
                    help="place table clutter from the tall obstacle pool only (diagnostic)")
    ap.add_argument("--handcrafted", action="store_true",
                    help="restrict table clutter to the handcrafted tallest/widest set (diagnostic)")
    ap.add_argument("--save-images", action="store_true",
                    help="save per-run countertop overlays (RGB | visible mouse pixels in red) to <out-dir>/images/")
    ap.add_argument("--selectable-threshold", type=float, default=0.05)
    ap.add_argument("--out-dir", default="../scripts/validation/results/visibility_phase1")
    ap.add_argument("--rollout", action="store_true",
                    help="run an expert curobo rollout per scene (writes to <out-dir>_rollout, "
                         "saves videos, and adds success-only distribution + P(success) per bucket)")
    ap.add_argument("--plot-only", action="store_true",
                    help="Skip the sweep; just (re)build figures from records.jsonl.")
    args = ap.parse_args()

    if not args.plot_only:
        run_sweep(args)
    out_dir = effective_out_dir(args)
    if args.rollout:
        analyze_rollout(out_dir, args.bins, args.selectable_threshold)
    else:
        analyze(out_dir, args.bins, args.selectable_threshold)


if __name__ == "__main__":
    main()
