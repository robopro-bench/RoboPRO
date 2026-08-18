#!/usr/bin/env python3
"""
Generate 4 comparison videos for the same task using 4 perturbation configs:
  1. randomized (original)
  2. vision
  3. object
  4. language

All runs use the same seed and clutter settings for fair comparison.

USAGE:
    source set_env.sh  # repo root
    cd sim
    python $BENCH_ROOT/bench_script/generate_comparison_videos.py \
        --task put_cup_on_coaster --bench-subdir study --seed 42
"""
import argparse
import copy
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


CONFIGS = {
    "randomized": "bench_demo_randomized",
    "vision": "bench_demo_vision",
    "object": "bench_demo_object",
    "language": "bench_demo_language",
}

CLUTTER_OVERRIDES = {
    "cluttered_table": True,
    "obstacle_density": 15,
}


def load_config(bench_root, config_name):
    path = Path(bench_root) / "bench_task_config" / f"{config_name}.yml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_temp_config(cfg, dest_path):
    with open(dest_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description="Generate 4 perturbation comparison videos")
    parser.add_argument("--task", type=str, default="put_cup_on_coaster",
                        help="Task name (default: put_cup_on_coaster)")
    parser.add_argument("--bench-subdir", type=str, default="study",
                        help="bench_envs subdirectory (default: study)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for videos (default: ./comparison_videos/<task>)")
    args = parser.parse_args()

    bench_root = os.environ.get("BENCH_ROOT")
    sim_root = os.environ.get("SIM_ROOT")
    if not bench_root or not sim_root:
        print("Error: BENCH_ROOT and SIM_ROOT must be set. Run: source set_env.sh")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else Path(f"./comparison_videos/{args.task}")
    output_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(sim_root) / "script" / "bench_script" / "visualize_task_scene.py"
    if not script_path.exists():
        print(f"Error: {script_path} not found")
        sys.exit(1)

    temp_config_dir = Path(bench_root) / "bench_task_config"

    for label, config_name in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  Generating video: {label}")
        print(f"{'='*60}")

        cfg = load_config(bench_root, config_name)

        dr = cfg.get("domain_randomization", {})
        for k, v in CLUTTER_OVERRIDES.items():
            dr[k] = v
        cfg["domain_randomization"] = dr

        cfg["save_path"] = str(output_dir / label)

        temp_name = f"_tmp_comparison_{label}"
        temp_path = temp_config_dir / f"{temp_name}.yml"
        write_temp_config(cfg, temp_path)

        cmd = [
            sys.executable,
            str(script_path),
            args.task,
            temp_name,
            "--seed", str(args.seed),
            "--rollout",
            "--save_data",
            "--no-render",
        ]
        if args.bench_subdir:
            cmd += ["--bench-subdir", args.bench_subdir]

        env = os.environ.copy()

        print(f"  Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, env=env, cwd=str(bench_root),
                                    timeout=600, capture_output=False)
            if result.returncode != 0:
                print(f"  WARNING: {label} exited with code {result.returncode}")
        except subprocess.TimeoutExpired:
            print(f"  WARNING: {label} timed out after 600s")
        finally:
            temp_path.unlink(missing_ok=True)

    print(f"\n{'='*60}")
    print(f"  Collecting videos")
    print(f"{'='*60}")

    for label in CONFIGS:
        video_dir = output_dir / label / "video"
        if video_dir.exists():
            for mp4 in sorted(video_dir.glob("*.mp4")):
                dest = output_dir / f"{label}.mp4"
                shutil.copy2(mp4, dest)
                print(f"  {dest}")
                break
        else:
            print(f"  WARNING: no video found for {label} at {video_dir}")

    print(f"\nDone. Videos saved to: {output_dir}")


if __name__ == "__main__":
    main()
