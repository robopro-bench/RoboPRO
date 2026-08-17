#!/usr/bin/env python3
"""Multi-GPU dispatcher for the data collector — invoked by collect/collect_data.sh when
its GPU argument is a comma list (e.g. `bash collect/collect_data.sh <task> <config> 0,1`).

Dynamic per-seed dispatch (targetted-failures style): workers pull candidate
seeds from ONE shared stream and run each as its own subprocess
(collect_one_episode.py) pinned to a free GPU from a pool — no static per-GPU
split, no disjoint seed bands. All workers write into ONE run dir; the episode
index is claimed atomically (flock'd slots.json) the moment a seed qualifies, so
episodes come out contiguous (episode0..N-1) exactly like a sequential run.
`episode_num` is read from the task config, so single- and multi-GPU runs collect
the same amount. Reruns resume from the existing seed.txt / slot count.

Usually you don't call this directly — use:  bash collect/collect_data.sh <task> <config> 0,1
Direct form:  python collect/collect_parallel.py <task> <config> --gpus 0,1
"""
import argparse
import fcntl
import itertools
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _env  # noqa: F401
from _env import resolve_save_path, run_gen_instructions

SCRIPT_DIR = Path(__file__).resolve().parent
SIM_ROOT = Path(os.environ["SIM_ROOT"])
REPO_ROOT = Path(os.environ["WORKSPACE_ROOT"])
BENCH_ROOT = Path(os.environ["BENCH_ROOT"])
BENCH_CFG_DIR = BENCH_ROOT / "bench_task_config"
RUN_EPISODE = SCRIPT_DIR / "collect_one_episode.py"

_INDEX_LOCK = threading.Lock()


class GpuOccupiedError(RuntimeError):
    """A visible GPU was occupied by a foreign process — GPUs are user-managed."""


def gpu_free_gib(gpu_id: str):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"], text=True, timeout=30)
    except Exception:  # noqa: BLE001
        return None
    for row in out.strip().splitlines():
        idx, _, free = row.partition(",")
        if idx.strip() == str(gpu_id).strip():
            try:
                return float(free) / 1024.0
            except ValueError:
                return None
    return None


def cfg_path(config: str) -> Path:
    """Resolve a config name to its YAML under benchmark/bench_task_config."""
    return BENCH_CFG_DIR / f"{config}.yml"


def read_cfg_scalar(config: str, key: str, default=None):
    text = cfg_path(config).read_text(encoding="utf-8")
    m = re.search(rf"^{re.escape(key)}\s*:\s*([^\n#]+)", text, re.MULTILINE)
    return m.group(1).strip() if m else default


def read_slots(run_dir: Path, target: int | None = None) -> dict:
    # Defensive: recreate the dir/lock so a run dir removed out from under us
    # (e.g. an external `rm -rf`) self-heals from the surviving seed.txt instead
    # of crashing the whole batch with FileNotFoundError.
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = open(run_dir / ".slots.lock", "a+", encoding="utf-8")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        sf = run_dir / "slots.json"
        if sf.exists():
            return json.loads(sf.read_text(encoding="utf-8"))
        seed_file = run_dir / "seed.txt"
        n = len(seed_file.read_text().split()) if seed_file.exists() else 0
        state = {"target": target if target is not None else n, "count": n}
        sf.write_text(json.dumps(state), encoding="utf-8")
        return state
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def init_slots(run_dir: Path, target: int):
    run_dir.mkdir(parents=True, exist_ok=True)
    seeds = []
    seed_file = run_dir / "seed.txt"
    if seed_file.exists():
        seeds = [int(s) for s in seed_file.read_text().split()]
    lock = open(run_dir / ".slots.lock", "a+", encoding="utf-8")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        (run_dir / "slots.json").write_text(
            json.dumps({"target": target, "count": len(seeds)}), encoding="utf-8")
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    return len(seeds), (max(seeds) + 1 if seeds else None)


def append_index(index_path: Path, row: dict):
    with _INDEX_LOCK:
        with open(index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def sweep_stale_temps(run_dir: Path):
    """Remove staging artifacts (episode index >= 900000) left by a hard-killed
    worker, so they never pollute the episode* glob."""
    import glob as _glob
    for sub, pat in (("data", "episode9[0-9][0-9][0-9][0-9][0-9].hdf5"),
                     ("video", "episode9[0-9][0-9][0-9][0-9][0-9].mp4"),
                     ("scene", "episode9[0-9][0-9][0-9][0-9][0-9]"),
                     ("_traj_data", "episode9[0-9][0-9][0-9][0-9][0-9].pkl"),
                     ("_traj_data", "episode9[0-9][0-9][0-9][0-9][0-9]_init.json")):
        for p in _glob.glob(str(run_dir / sub / pat)):
            try:
                if os.path.isdir(p):
                    import shutil as _sh
                    _sh.rmtree(p)
                else:
                    os.remove(p)
            except Exception:
                pass


def collect(args, task, config, gpu_ids, run_dir, index_path):
    sweep_stale_temps(run_dir)
    claimed0, resume_seed = init_slots(run_dir, args.episodes)
    next_seed = itertools.count(resume_seed if resume_seed is not None
                                else args.start_seed)
    if claimed0:
        print(f"[parallel] resuming — {claimed0}/{args.episodes} already claimed; "
              f"seeds continue from {resume_seed}")

    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    gpu_pool: queue.Queue = queue.Queue()
    for i in range(args.workers):
        gpu_pool.put(gpu_ids[i % len(gpu_ids)])
    state_lock = threading.Lock()
    stop = threading.Event()
    preflighted: set = set()
    attempts = itertools.count()
    fatal_streak = [0]

    def dispatch_one():
        with state_lock:
            if stop.is_set():
                return False
            if next(attempts) >= args.max_seed_attempts:
                print(f"[parallel] max-seed-attempts ({args.max_seed_attempts}) "
                      f"reached — stopping")
                stop.set()
                return False
            seed = next(next_seed)

        gpu = gpu_pool.get()
        try:
            if args.min_free_gib > 0:
                with state_lock:
                    first = gpu not in preflighted
                    preflighted.add(gpu)
                if first:
                    free = gpu_free_gib(gpu)
                    if free is not None and free < args.min_free_gib:
                        stop.set()
                        raise GpuOccupiedError(
                            f"gpu {gpu} has {free:.1f} GiB free "
                            f"(< {args.min_free_gib}); free it or drop it from the "
                            f"GPU list, then rerun (resumes from cache).")

            env = dict(os.environ)
            env["SIM_ROOT"] = str(SIM_ROOT)
            env["WORKSPACE_ROOT"] = str(REPO_ROOT)
            env["BENCH_ROOT"] = str(BENCH_ROOT)
            env["DATA_ROOT"] = os.environ["DATA_ROOT"]
            env["ASSETS_ROOT"] = os.environ.get("ASSETS_ROOT", str(REPO_ROOT / "assets"))
            env["POLICY_ROOT"] = os.environ.get("POLICY_ROOT", str(REPO_ROOT / "policy"))
            env["PYTHONWARNINGS"] = "ignore::UserWarning"
            env["PYTHONUNBUFFERED"] = "1"
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

            cmd = [args.python, "-u", str(RUN_EPISODE), task, config,
                   "--seed", str(seed)]
            log_path = log_dir / f"seed{seed:06d}.log"
            t0 = time.time()
            try:
                with open(log_path, "w", encoding="utf-8") as log:
                    proc = subprocess.run(cmd, cwd=str(SIM_ROOT), env=env,
                                          stdout=log, stderr=subprocess.STDOUT,
                                          timeout=args.timeout, check=False)
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                rc = 124
            dur = time.time() - t0
        finally:
            gpu_pool.put(gpu)

        status = {0: "saved", 3: "seed_failed", 4: "surplus",
                  5: "collect_failed", 124: "timeout"}.get(rc, f"error_rc{rc}")
        slots = read_slots(run_dir, args.episodes)
        append_index(index_path, {"task": task, "config": config, "seed": seed,
                                  "gpu": gpu, "status": status, "returncode": rc,
                                  "duration_s": round(dur, 1),
                                  "claimed": slots["count"], "target": slots["target"]})
        print(f"  seed={seed} gpu={gpu} -> {status:12s} ({dur:.0f}s)  "
              f"progress {slots['count']}/{slots['target']}")

        with state_lock:
            if status.startswith("error"):
                fatal_streak[0] += 1
                if fatal_streak[0] >= args.max_fatal_streak:
                    print(f"[parallel] {fatal_streak[0]} consecutive fatal errors "
                          f"— stopping (see {log_dir})")
                    stop.set()
            else:
                fatal_streak[0] = 0
        if rc == 4 or slots["count"] >= slots["target"]:
            stop.set()
        return True

    def worker():
        while not stop.is_set():
            if read_slots(run_dir, args.episodes)["count"] >= args.episodes:
                stop.set()
                break
            if not dispatch_one():
                break

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker) for _ in range(args.workers)]
        for fut in futures:
            fut.result()  # re-raises GpuOccupiedError

    n_final = read_slots(run_dir, args.episodes)["count"]
    lang_num = read_cfg_scalar(config, "language_num", "100")
    print(f"[parallel] {n_final}/{args.episodes} episodes — generating instructions")
    run_gen_instructions(task, config, lang_num)
    return n_final


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task_name")
    p.add_argument("task_config")
    p.add_argument("--gpus", default=None, help="comma list of GPU ids, e.g. 0,1")
    p.add_argument("--gpu", default=None, help="single GPU id")
    p.add_argument("--episodes", type=int, default=None,
                   help="successful episodes to bank (default: episode_num from config)")
    p.add_argument("--workers", type=int, default=0,
                   help="concurrent episode subprocesses (default: #GPUs)")
    p.add_argument("--start-seed", type=int, default=0,
                   help="first candidate seed (fresh runs; resume continues seed.txt)")
    p.add_argument("--min-free-gib", type=float, default=12.0,
                   help="occupied-GPU preflight threshold (0 disables)")
    p.add_argument("--timeout", type=int, default=2400, help="per-episode seconds")
    p.add_argument("--max-seed-attempts", type=int, default=200)
    p.add_argument("--max-fatal-streak", type=int, default=3)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    gpu_ids = ([g.strip() for g in args.gpus.split(",") if g.strip()] if args.gpus
               else [args.gpu] if args.gpu else ["0"])
    if args.workers <= 0:
        args.workers = len(gpu_ids)
    if args.episodes is None:
        args.episodes = int(read_cfg_scalar(args.task_config, "episode_num", "0"))
    if args.episodes <= 0:
        sys.exit("[parallel] episode_num not found in config and --episodes not given")

    save_path = read_cfg_scalar(args.task_config, "save_path", "./data")
    root = Path(resolve_save_path(save_path))
    run_dir = root / args.task_name / args.task_config
    index_path = root / "collect_parallel_index.jsonl"

    print(f"[parallel] {args.task_name} / {args.task_config}: {args.episodes} episodes, "
          f"dynamic dispatch on gpus={gpu_ids} ({args.workers} workers)")
    print(f"[parallel] run dir: {run_dir}")
    if args.dry_run:
        print("[parallel] dry-run: nothing launched.")
        return

    root.mkdir(parents=True, exist_ok=True)
    try:
        n = collect(args, args.task_name, args.task_config, gpu_ids, run_dir, index_path)
    except GpuOccupiedError as e:
        print(f"\n[parallel] ABORTED — {e}", file=sys.stderr)
        sys.exit(2)
    print(f"[parallel] DONE — {n}/{args.episodes} episodes in {run_dir}")


if __name__ == "__main__":
    main()
