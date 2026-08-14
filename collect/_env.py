"""Bootstrap collect/ scripts so they can run from any cwd.

Puts SIM_ROOT / BENCH_ROOT / collect/ on sys.path and exports
WORKSPACE_ROOT, SIM_ROOT, BENCH_ROOT, ASSETS_ROOT, DATA_ROOT. Relative YAML
save_path values (./data/dataset, ./data/bench_data) resolve under DATA_ROOT.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

COLLECT_DIR = Path(__file__).resolve().parent
WORKSPACE = COLLECT_DIR.parent


def bootstrap() -> None:
    os.environ.setdefault("WORKSPACE_ROOT", str(WORKSPACE))
    os.environ.setdefault("SIM_ROOT", str(WORKSPACE / "sim"))
    os.environ.setdefault("BENCH_ROOT", str(WORKSPACE / "benchmark"))
    os.environ.setdefault("ASSETS_ROOT", str(WORKSPACE / "assets"))
    os.environ.setdefault("DATA_ROOT", str(WORKSPACE / "data"))
    os.environ.setdefault("POLICY_ROOT", str(WORKSPACE / "policy"))
    for p in (
        os.environ["SIM_ROOT"],
        os.environ["BENCH_ROOT"],
        os.environ["POLICY_ROOT"],
        str(COLLECT_DIR),
    ):
        if p not in sys.path:
            sys.path.insert(0, p)


def data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT") or (WORKSPACE / "data"))


def resolve_save_path(save_path: str) -> str:
    """Map a YAML save_path onto DATA_ROOT without rewriting the YAMLs.

    `./data/dataset` and `./data/bench_data` become `$DATA_ROOT/dataset`
    and `$DATA_ROOT/bench_data`. Absolute paths are left alone.
    """
    p = Path(save_path)
    if p.is_absolute():
        return str(p)
    parts = list(p.parts)
    if parts[:1] == ['.']:
        parts = parts[1:]
    if parts[:1] == ['data']:
        parts = parts[1:]
    return str(data_root().joinpath(*parts)) if parts else str(data_root())


def run_gen_instructions(task_name, task_config, language_num) -> None:
    desc = Path(os.environ["SIM_ROOT"]) / "description"
    script = desc / "gen_episode_instructions.sh"
    if not script.exists():
        print(f"[collect] skip instructions: {script} missing")
        return
    subprocess.run(
        ["bash", str(script), str(task_name), str(task_config), str(language_num)],
        cwd=str(desc),
        check=False,
    )


bootstrap()
