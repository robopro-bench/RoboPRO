"""Pre-collect evaluation seeds for the RoboTwin benchmark.

Mirrors the first-pass loop of collect/collect_data.py but writes ONLY a seed
file — no pkl, no hdf5, no mp4. Output:

    {BENCH_ROOT}/eval_seeds/{task}/{config}.txt

Seeds start at 40000 to guarantee no collision with training seeds (0..~30).
Per (task, config) target: 20 seeds for *_clean configs, 2 seeds otherwise.
"""

import sys
import os
import time
import traceback
import importlib
import tempfile
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _env  # noqa: F401

import yaml
import sapien.core as sapien  # noqa: F401  -- import order matters for envs
from envs import *  # noqa: F401,F403

bench_root = Path(os.environ["BENCH_ROOT"])

EVAL_SEED_ROOT = bench_root / "eval_seeds"
SEED_BASE = 40000
MAX_TRIES_CLEAN = int(os.environ.get("MAX_TRIES_CLEAN", "500"))
MAX_TRIES_CLUTTER = int(os.environ.get("MAX_TRIES_CLUTTER", "200"))


def class_decorator(task_name):
    from script.task_loader import load_task
    return load_task(task_name)


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def load_args(task_name, task_config):
    from script.task_loader import bench_config_path
    task_config_path = bench_config_path(task_config)

    with open(task_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")  # noqa: F405
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(et):
        rf = _embodiment_types[et]["file_path"]
        if rf is None:
            raise RuntimeError("missing embodiment files")
        return rf

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("number of embodiment config parameters should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])
    args["embodiment_name"] = embodiment_name

    # Force seed-only behavior: never collect data, never render, never save traj.
    args["collect_data"] = False
    args["render_freq"] = 0
    args["save_data"] = False
    args["use_seed"] = False
    args["need_plan"] = True
    args["save_seed"] = False  # seed file is owned by us, not the env

    # Scratch save_path so any incidental directory creation lands in /tmp.
    scratch = tempfile.mkdtemp(prefix=f"precollect_{task_name}_{task_config}_")
    args["save_path"] = scratch
    args["__scratch_path"] = scratch
    return args


def atomic_write_seeds(seed_path: Path, seeds):
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = seed_path.with_suffix(seed_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(" ".join(str(s) for s in seeds))
    os.replace(tmp, seed_path)


def main(task_name, task_config):
    target = 20 if task_config.endswith("_clean") else 2
    max_tries = MAX_TRIES_CLEAN if task_config.endswith("_clean") else MAX_TRIES_CLUTTER

    seed_path = EVAL_SEED_ROOT / task_name / f"{task_config}.txt"

    existing = []
    if seed_path.exists():
        existing = [int(s) for s in seed_path.read_text().split() if s]

    if len(existing) >= target:
        print(f"[skip] {task_name}/{task_config}: already have {len(existing)} >= {target}")
        return

    seeds = list(existing)
    epid = max([SEED_BASE] + [s + 1 for s in existing])
    tries = 0

    print(f"[{task_name}/{task_config}] target={target} have={len(seeds)} start_epid={epid}")

    args = load_args(task_name, task_config)
    TASK_ENV = class_decorator(task_name)

    fail_num = 0
    try:
        while len(seeds) < target and tries < max_tries:
            try:
                TASK_ENV.setup_demo(now_ep_num=len(seeds), seed=epid, is_test=True, **args)
                if hasattr(TASK_ENV, "_maybe_apply_language_perturbation"):
                    TASK_ENV._maybe_apply_language_perturbation()
                TASK_ENV.play_once()

                if TASK_ENV.plan_success and TASK_ENV.check_success():
                    print(f"  success! (seed={epid}) -> {len(seeds)+1}/{target}")
                    seeds.append(epid)
                    atomic_write_seeds(seed_path, seeds)
                else:
                    fail_num += 1
                TASK_ENV.close_env()
            except Exception as e:
                fail_num += 1
                print(f"  fail (seed={epid}): {e}")
                traceback.print_exc()
                try:
                    TASK_ENV.close_env()
                except Exception:
                    pass
                time.sleep(0.3)

            epid += 1
            tries += 1
    finally:
        # Best-effort scratch cleanup.
        try:
            import shutil
            shutil.rmtree(args["__scratch_path"], ignore_errors=True)
        except Exception:
            pass

    if len(seeds) < target:
        print(f"[WARN] {task_name}/{task_config}: only {len(seeds)}/{target} after {tries} tries (fails={fail_num})")
    else:
        print(f"[done] {task_name}/{task_config}: {len(seeds)}/{target} (tries={tries}, fails={fail_num})")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    pa = parser.parse_args()
    main(task_name=pa.task_name, task_config=pa.task_config)
