"""Bootstrap eval/ scripts so they can run from any cwd.

Puts SIM_ROOT / BENCH_ROOT / policy/ / eval/ on sys.path and exports
WORKSPACE_ROOT, SIM_ROOT, BENCH_ROOT, DATA_ROOT, POLICY_ROOT.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
WORKSPACE = EVAL_DIR.parent


def bootstrap() -> None:
    os.environ.setdefault("WORKSPACE_ROOT", str(WORKSPACE))
    os.environ.setdefault("SIM_ROOT", str(WORKSPACE / "sim"))
    os.environ.setdefault("BENCH_ROOT", str(WORKSPACE / "benchmark"))
    os.environ.setdefault("DATA_ROOT", str(WORKSPACE / "data"))
    os.environ.setdefault("POLICY_ROOT", str(WORKSPACE / "policy"))
    sim = os.environ["SIM_ROOT"]
    for p in (
        sim,
        os.environ["BENCH_ROOT"],
        os.environ["POLICY_ROOT"],
        str(EVAL_DIR),
        os.path.join(sim, "description", "utils"),
    ):
        if p not in sys.path:
            sys.path.insert(0, p)


def eval_result_root() -> Path:
    return Path(os.environ.get("WORKSPACE_ROOT") or WORKSPACE) / "eval_result"


bootstrap()
