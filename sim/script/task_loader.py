"""Load RoboPRO bench tasks and their YAML configs.

Bench tasks live under benchmark/bench_envs/{office,study,kitchenl,kitchens}/.
There is no fallback to upstream RoboTwin task modules.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

BENCH_SUBDIRS = ("office", "study", "kitchenl", "kitchens")


def bench_root() -> Path:
    root = os.environ.get("BENCH_ROOT")
    if not root:
        raise SystemExit("BENCH_ROOT is not set; source set_env.sh first")
    return Path(root)


def bench_config_path(task_config: str) -> Path:
    return bench_root() / "bench_task_config" / f"{task_config}.yml"


def load_task_module(task_name: str, bench_subdir: str | None = None):
    packages = (
        [f"bench_envs.{bench_subdir}"]
        if bench_subdir
        else [f"bench_envs.{d}" for d in BENCH_SUBDIRS] + ["bench_envs"]
    )
    last_err = None
    for pkg in packages:
        try:
            return importlib.import_module(f"{pkg}.{task_name}")
        except ModuleNotFoundError as exc:
            last_err = exc
            continue
    where = f"bench_envs.{bench_subdir}" if bench_subdir else "bench_envs"
    raise SystemExit(f"No such task: {task_name} (looked in {where})") from last_err


def load_task_class(task_name: str, bench_subdir: str | None = None):
    module = load_task_module(task_name, bench_subdir=bench_subdir)
    try:
        return getattr(module, task_name)
    except AttributeError as exc:
        raise SystemExit(f"No task class '{task_name}' in {module.__name__}") from exc


def load_task(task_name: str, bench_subdir: str | None = None):
    return load_task_class(task_name, bench_subdir=bench_subdir)()
