"""Helpers for loading precollected benchmark evaluation seeds.

Seed files live at:
    {BENCH_ROOT}/eval_seeds/{task}/{config}.txt
and are space-separated integers (see script/precollect_eval_seeds.py).
"""

from __future__ import annotations

import os
from pathlib import Path


def _truthy(val, default=True):
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    if s in ("0", "false", "no", "off"):
        return False
    if s in ("1", "true", "yes", "on"):
        return True
    return default


def resolve_eval_seeds(task_name, task_config, usr_args):
    """Return a fixed seed list for eval, or None to scan from st_seed.

    Priority:
      1. Explicit --use_eval_seeds false / USE_EVAL_SEEDS=0 -> scan mode
      2. --eval_seed_file PATH
      3. {BENCH_ROOT}/eval_seeds/{task}/{config}.txt if present
      4. None (scan from st_seed with expert_check)
    """
    use_flag = usr_args.get("use_eval_seeds", os.environ.get("USE_EVAL_SEEDS"))
    if use_flag is not None and not _truthy(use_flag, default=True):
        return None

    seed_file = usr_args.get("eval_seed_file")
    if seed_file:
        path = Path(seed_file)
        if not path.exists():
            raise FileNotFoundError(f"eval_seed_file not found: {path}")
        seeds = [int(s) for s in path.read_text().split() if s]
        if not seeds:
            raise ValueError(f"eval_seed_file is empty: {path}")
        return seeds

    bench_root = os.environ.get("BENCH_ROOT")
    if not bench_root:
        return None
    path = Path(bench_root) / "eval_seeds" / task_name / f"{task_config}.txt"
    if path.exists():
        seeds = [int(s) for s in path.read_text().split() if s]
        if seeds:
            return seeds
    return None


def resolve_test_num(usr_args, seed_list, default=1):
    if usr_args.get("test_num") is not None:
        test_num = int(usr_args["test_num"])
    elif os.environ.get("EVAL_TEST_NUM"):
        test_num = int(os.environ["EVAL_TEST_NUM"])
    elif seed_list is not None:
        test_num = len(seed_list)
    else:
        test_num = default
    if seed_list is not None:
        test_num = min(test_num, len(seed_list))
    return test_num


def resolve_instruction_bank(bank_path):
    """Resolve an instruction_bank config path (usually 'benchmark/...') to a real file.

    The eval process runs from sim/, where 'benchmark/' does not exist,
    so a bare relative path from the config fails to resolve. Try the path as given,
    then a few BENCH_ROOT-based candidates. Returns the first existing path, else the
    original (so the caller's own os.path.exists() check still governs the fallback).
    """
    if not bank_path:
        return bank_path
    if os.path.isabs(bank_path) and os.path.exists(bank_path):
        return bank_path
    candidates = [bank_path]
    bench_root = os.environ.get("BENCH_ROOT")
    if bench_root:
        bench_root = bench_root.rstrip("/")
        base = os.path.basename(bench_root)  # e.g. "benchmark"
        # config path is repo-root-relative ("benchmark/..."); BENCH_ROOT already points at .../benchmark
        candidates.append(os.path.join(os.path.dirname(bench_root), bank_path))
        candidates.append(os.path.join(bench_root, bank_path))
        if bank_path.startswith(base + os.sep):
            candidates.append(os.path.join(bench_root, bank_path[len(base) + 1:]))
    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand
    return bank_path


def resolve_expert_check(usr_args, default=True):
    """Whether to run the live expert pre-check per seed (scan mode only).

    Precedence: `--expert_check <bool>` / `EXPERT_CHECK` env, else `default`.
    Callers AND this with `seed_list is None`, since precollected seeds are
    already expert-validated and never need a live check.
    """
    val = usr_args.get("expert_check", os.environ.get("EXPERT_CHECK"))
    if val is None:
        return default
    return _truthy(val, default=default)
