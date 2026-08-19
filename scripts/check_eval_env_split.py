#!/usr/bin/env python3
"""Guard the dual-env eval split.

Dual-env eval (policy/pi05/eval_double_env.sh) runs two processes in two
different interpreters:

    eval/policy_model_server.py  -> policy/pi05/openpi/.venv   (jax, NO sapien)
    eval/eval_policy_client.py   -> RoboPRO sim env            (sapien)

Nothing the server imports, directly or transitively, may pull in sapien --
`sim/envs/__init__.py` does, so a stray `from envs import ...` there breaks
every dual-env run with `ModuleNotFoundError: No module named 'sapien'`. That
regression is invisible in single-process (Mode A) eval and only shows up at
launch time, so check it here.

Run from the repo root with the sim env active (needs sapien on the path):

    python scripts/check_eval_env_split.py

Exit status 0 = split intact, 1 = broken.
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER = REPO_ROOT / "eval" / "policy_model_server.py"
CLIENT = REPO_ROOT / "eval" / "eval_policy_client.py"


class _BlockSapien:
    """Meta-path hook that makes `import sapien` fail, mimicking the policy venv."""

    def find_spec(self, name, path=None, target=None):
        if name == "sapien" or name.startswith("sapien."):
            raise ImportError(f"blocked import of {name}")
        return None


def _import_isolated(path, block_sapien):
    """Import `path` as a throwaway module, optionally with sapien unavailable."""
    for mod in [m for m in sys.modules if m.split(".")[0] in ("sapien", "envs", "_env")]:
        del sys.modules[mod]
    if block_sapien:
        sys.meta_path.insert(0, _BlockSapien())
    try:
        spec = importlib.util.spec_from_file_location(f"_check_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if block_sapien:
            sys.meta_path.pop(0)


def main():
    failures = []

    # 1. The server must import with sapien unavailable.
    try:
        _import_isolated(SERVER, block_sapien=True)
        print(f"ok   {SERVER.relative_to(REPO_ROOT)} imports without sapien")
    except ImportError as exc:
        failures.append(
            f"{SERVER.relative_to(REPO_ROOT)} needs sapien ({exc}).\n"
            "     It runs in policy/pi05/openpi/.venv, which has no sapien. Drop the\n"
            "     import that reaches `envs` -- see the NOTE near the top of that file."
        )

    # 2. The client legitimately uses `envs`; it must still import in the sim env.
    #    (If sapien is genuinely missing here, the sim env itself is broken.)
    try:
        _import_isolated(CLIENT, block_sapien=False)
        print(f"ok   {CLIENT.relative_to(REPO_ROOT)} imports in the sim env")
    except ImportError as exc:
        failures.append(
            f"{CLIENT.relative_to(REPO_ROOT)} failed to import in the sim env ({exc}).\n"
            "     Is sapien installed? See README Installation step 2."
        )

    if failures:
        print("\nFAIL: dual-env eval split is broken\n", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        return 1

    print("\nPASS: dual-env eval split intact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
