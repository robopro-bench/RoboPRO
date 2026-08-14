"""Helpers for frozen evaluation environment packs (sibling of eval_seeds).

Packs live at:
    {BENCH_ROOT}/eval_envs/{task}/{config}/episode_XXXX.json
plus a manifest.json. Eval identity is the JSON pack, not a seed integer.
See collect/precollect_eval_envs.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from eval_seeds import _truthy, resolve_test_num  # noqa: F401  — re-exported for eval callers

SCHEMA_VERSION = 1


def eval_env_root(bench_root=None) -> Path:
    root = bench_root or os.environ.get("BENCH_ROOT")
    if not root:
        raise RuntimeError("BENCH_ROOT is not set")
    return Path(root) / "eval_envs"


def eval_env_dir(task_name, task_config, bench_root=None) -> Path:
    return eval_env_root(bench_root) / task_name / task_config


def episode_filename(idx: int) -> str:
    return f"episode_{int(idx):04d}.json"


def episode_path(env_dir: Path, idx: int) -> Path:
    return Path(env_dir) / episode_filename(idx)


def _json_default(o):
    import numpy as np
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def load_eval_env_pack(path) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        pack = json.load(f)
    if not isinstance(pack, dict):
        raise ValueError(f"eval env pack is not an object: {path}")
    return pack


def atomic_write_pack(path: Path, pack: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, indent=2, default=_json_default)
    os.replace(tmp, path)


def write_manifest(env_dir: Path, *, task_name, task_config, episodes):
    env_dir = Path(env_dir)
    env_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task_name": task_name,
        "task_config": task_config,
        "episodes": list(episodes),
    }
    path = env_dir / "manifest.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def list_eval_env_paths(env_dir: Path) -> list[Path]:
    env_dir = Path(env_dir)
    if not env_dir.is_dir():
        return []
    manifest = env_dir / "manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        episodes = data.get("episodes") or []
        paths = []
        for ep in episodes:
            if isinstance(ep, str):
                p = env_dir / ep
            elif isinstance(ep, dict):
                p = env_dir / ep.get("file", "")
            else:
                continue
            if p.is_file():
                paths.append(p)
        if paths:
            return paths
    return sorted(env_dir.glob("episode_*.json"))


def load_eval_env_list(env_dir: Path) -> list[dict]:
    packs = []
    for path in list_eval_env_paths(env_dir):
        pack = load_eval_env_pack(path)
        meta = pack.setdefault("meta", {})
        meta.setdefault("file", path.name)
        meta.setdefault("env_id", path.stem)
        packs.append(pack)
    return packs


def _task_supports_eval_env(task_env) -> bool:
    if task_env is None:
        return False
    return bool(getattr(task_env, "supports_eval_env", False))


def resolve_eval_envs(task_name, task_config, usr_args, task_env=None):
    """Return a list of env packs for eval, or None to fall through to seeds.

    Priority:
      1. Explicit --use_eval_envs false / USE_EVAL_ENVS=0 -> None
      2. Task does not set supports_eval_env -> None
      3. --eval_env_dir PATH
      4. {BENCH_ROOT}/eval_envs/{task}/{config}/ if it contains packs
      5. None

    Explicit --use_eval_envs true with a missing dir raises FileNotFoundError.
    """
    use_flag = usr_args.get("use_eval_envs", os.environ.get("USE_EVAL_ENVS"))
    if use_flag is not None and not _truthy(use_flag, default=True):
        return None

    if not _task_supports_eval_env(task_env):
        if use_flag is not None and _truthy(use_flag, default=True) and usr_args.get("eval_env_dir"):
            raise RuntimeError(
                f"task {task_name} does not set supports_eval_env; cannot load eval envs"
            )
        return None

    env_dir = usr_args.get("eval_env_dir") or os.environ.get("EVAL_ENV_DIR")
    if env_dir:
        path = Path(env_dir)
        if not path.exists():
            raise FileNotFoundError(f"eval_env_dir not found: {path}")
        packs = load_eval_env_list(path)
        if not packs:
            raise ValueError(f"eval_env_dir has no episode packs: {path}")
        return packs

    bench_root = os.environ.get("BENCH_ROOT")
    if not bench_root:
        if use_flag is not None and _truthy(use_flag, default=True):
            raise RuntimeError("USE_EVAL_ENVS is set but BENCH_ROOT is missing")
        return None

    path = eval_env_dir(task_name, task_config, bench_root)
    packs = load_eval_env_list(path) if path.is_dir() else []
    if packs:
        return packs
    if use_flag is not None and _truthy(use_flag, default=True):
        raise FileNotFoundError(
            f"use_eval_envs is set but no packs at {path}"
        )
    return None


def env_pack_label(pack: dict) -> str:
    meta = pack.get("meta") or {}
    return str(meta.get("file") or meta.get("env_id") or episode_filename(meta.get("episode_index", 0)))


def eval_setup_kwargs(now_id, now_seed, args, eval_env=None, **extra):
    """kwargs for setup_demo in eval. Pass eval_env only in pack mode."""
    kw = dict(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
    if extra:
        kw.update(extra)
    if eval_env is not None:
        kw["eval_env"] = eval_env
    return kw
