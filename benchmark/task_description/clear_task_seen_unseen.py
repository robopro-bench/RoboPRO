from argparse import ArgumentParser
import json
import os
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_WORKSPACE = _DIR.parent.parent
os.environ.setdefault("BENCH_ROOT", str(_WORKSPACE / "benchmark"))


def _task_path(task_name):
    return Path(os.environ["BENCH_ROOT"]) / "bench_description" / "task_instructions" / f"{task_name}.json"


def clear_seen_unseen(task_name):
    path = _task_path(task_name)
    with open(path, "r") as f:
        task_info = json.load(f)
    task_info["seen"] = []
    task_info["unseen"] = []
    with open(path, "w") as f:
        json.dump(task_info, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    args = parser.parse_args()
    clear_seen_unseen(args.task_name)
