"""Pre-collect frozen evaluation environment packs.

Mirrors script/precollect_eval_seeds.py but writes JSON env packs instead of
seed integers. Search still walks candidate seeds internally; the artifact is:

    {BENCH_ROOT}/eval_envs/{task}/{config}/episode_XXXX.json
    {BENCH_ROOT}/eval_envs/{task}/{config}/manifest.json

Requires the task to set supports_eval_env and implement capture_task_spec /
load_actors_from_spec. Does not write eval_seeds/.
"""

import sys
import os
import time
import traceback
from argparse import ArgumentParser
from pathlib import Path

sys.path.append("./")

from script.bench_script.setup_paths import setup_paths
setup_paths()

import sapien.core as sapien  # noqa: F401  -- import order matters for envs
from envs import *  # noqa: F401,F403

from script.precollect_eval_seeds import (
    class_decorator,
    load_args,
    SEED_BASE,
    MAX_TRIES_CLEAN,
    MAX_TRIES_CLUTTER,
)
from script.eval_envs import (
    eval_env_dir,
    episode_filename,
    episode_path,
    list_eval_env_paths,
    atomic_write_pack,
    write_manifest,
)

bench_root = Path(os.environ["BENCH_ROOT"])


def main(task_name, task_config, target=None, start_seed=None):
    if target is None:
        env_target = os.environ.get("EVAL_ENV_TARGET")
        target = int(env_target) if env_target else None
    if target is None:
        target = 20 if task_config.endswith("_clean") else 2
    max_tries = MAX_TRIES_CLEAN if task_config.endswith("_clean") else MAX_TRIES_CLUTTER
    if start_seed is None:
        start_seed = int(os.environ.get("EVAL_ENV_START_SEED", str(SEED_BASE)))

    TASK_ENV = class_decorator(task_name)
    if not getattr(TASK_ENV, "supports_eval_env", False):
        raise SystemExit(
            f"{task_name} does not set supports_eval_env; implement capture_task_spec "
            f"and load_actors_from_spec before precollecting eval envs"
        )

    env_dir = eval_env_dir(task_name, task_config)
    existing_paths = list_eval_env_paths(env_dir)
    if len(existing_paths) >= target:
        print(f"[skip] {task_name}/{task_config}: already have {len(existing_paths)} >= {target}")
        return

    episode_files = [p.name for p in existing_paths]
    next_idx = len(episode_files)
    epid = int(start_seed)
    if existing_paths:
        import json
        used_seeds = []
        for p in existing_paths:
            try:
                meta = json.loads(p.read_text(encoding="utf-8")).get("meta") or {}
                s = meta.get("search_seed")
                if s is not None:
                    used_seeds.append(int(s))
            except Exception:
                pass
        if used_seeds:
            epid = max(max(used_seeds) + 1, epid)
    tries = 0

    print(f"[{task_name}/{task_config}] target={target} have={len(episode_files)} start_epid={epid}")

    args = load_args(task_name, task_config)
    args["eval_mode"] = True
    fail_num = 0
    try:
        while len(episode_files) < target and tries < max_tries:
            try:
                TASK_ENV.setup_demo(now_ep_num=next_idx, seed=epid, is_test=True, **args)
                if hasattr(TASK_ENV, "_maybe_apply_language_perturbation"):
                    TASK_ENV._maybe_apply_language_perturbation()
                pack = TASK_ENV.build_eval_env_pack(episode_index=next_idx, search_seed=epid)
                TASK_ENV.play_once()

                if TASK_ENV.plan_success and TASK_ENV.check_success():
                    fname = episode_filename(next_idx)
                    atomic_write_pack(episode_path(env_dir, next_idx), pack)
                    episode_files.append(fname)
                    write_manifest(
                        env_dir,
                        task_name=task_name,
                        task_config=task_config,
                        episodes=episode_files,
                    )
                    print(f"  success! (search_seed={epid}) -> {fname} ({len(episode_files)}/{target})")
                    next_idx += 1
                else:
                    fail_num += 1
                TASK_ENV.close_env()
            except Exception as e:
                fail_num += 1
                print(f"  fail (search_seed={epid}): {e}")
                traceback.print_exc()
                try:
                    TASK_ENV.close_env()
                except Exception:
                    pass
                time.sleep(0.3)

            epid += 1
            tries += 1
    finally:
        try:
            import shutil
            shutil.rmtree(args["__scratch_path"], ignore_errors=True)
        except Exception:
            pass

    if len(episode_files) < target:
        print(f"[WARN] {task_name}/{task_config}: only {len(episode_files)}/{target} after {tries} tries (fails={fail_num})")
    else:
        print(f"[done] {task_name}/{task_config}: {len(episode_files)}/{target} (tries={tries}, fails={fail_num})")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("--target", type=int, default=None,
                        help="Number of expert-passing env packs to collect (default: 20 clean / 2 otherwise)")
    parser.add_argument("--start-seed", type=int, default=None,
                        help=f"First search seed (default: {SEED_BASE})")
    pa = parser.parse_args()
    main(task_name=pa.task_name, task_config=pa.task_config,
         target=pa.target, start_seed=pa.start_seed)
