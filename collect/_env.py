"""Bootstrap collect/ scripts so they can run from any cwd.

Puts SIM_ROOT / SIM_ROOT/script / BENCH_ROOT / policy/ / collect/ on sys.path
and exports WORKSPACE_ROOT, SIM_ROOT, BENCH_ROOT, ASSETS_ROOT, DATA_ROOT,
POLICY_ROOT. Relative YAML save_path values (./data, ./data/...) resolve under
DATA_ROOT (repo-root data/).
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
    sim = os.environ["SIM_ROOT"]
    for p in (
        sim,
        os.environ["BENCH_ROOT"],
        os.environ["POLICY_ROOT"],
        str(COLLECT_DIR),
        # sim/script holds modules imported bare (e.g. `from test_render import
        # Sapien_TEST`), not via the `script.` package prefix.
        os.path.join(sim, "script"),
    ):
        if p not in sys.path:
            sys.path.insert(0, p)


def data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT") or (WORKSPACE / "data"))


def resolve_save_path(save_path: str) -> str:
    """Map a YAML save_path onto DATA_ROOT.

    `./data` becomes `$DATA_ROOT`. `./data/<bucket>` still becomes
    `$DATA_ROOT/<bucket>`. Absolute paths are left alone.
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
    gen = COLLECT_DIR / "generate_episode_instructions.py"
    if not gen.exists():
        print(f"[collect] skip instructions: {gen} missing")
        return
    subprocess.run(
        [sys.executable, str(gen), str(task_name), str(task_config), str(language_num)],
        cwd=str(WORKSPACE),
        check=False,
    )


bootstrap()
