#!/usr/bin/env python
"""
Upload generated instruction sidecars to HuggingFace, one folder per
(scene, task, split). Staging layout must be:

  <staging>/{scene}/{task}/{split}/instructions/episode{N}.json

Each dir is uploaded to the matching path on `RoboPRO/RoboPRO_data_raw`.
"""

import argparse
import os
import re
import time
from pathlib import Path

from huggingface_hub import HfApi

REPO = "RoboPRO/RoboPRO_data_raw"


def upload_with_retry(api, local, repo_path, max_attempts=8, base_sleep=30):
    """upload_folder with exponential backoff on 429 rate-limit errors."""
    attempt = 0
    while True:
        attempt += 1
        try:
            api.upload_folder(
                folder_path=str(local),
                repo_id=REPO,
                repo_type="dataset",
                path_in_repo=repo_path,
                commit_message=f"add instructions: {repo_path}",
            )
            return
        except Exception as e:
            msg = str(e)
            is_rate = "429" in msg or "Too Many Requests" in msg
            if not is_rate or attempt >= max_attempts:
                raise
            sleep_for = base_sleep * (2 ** (attempt - 1))
            sleep_for = min(sleep_for, 600)
            print(f"  [rate-limit] attempt {attempt}: sleeping {sleep_for}s then retrying {repo_path}",
                  flush=True)
            time.sleep(sleep_for)


def already_uploaded(existing, repo_path, expected_count):
    """True if HF already has `expected_count` episode*.json under repo_path."""
    prefix = repo_path + "/"
    count = sum(1 for p in existing if p.startswith(prefix) and re.match(
        r".+/episode\d+\.json$", p))
    return count >= expected_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    parser.add_argument(
        "--staging",
        default=str(Path(__file__).resolve().parent.parent / "customized_robotwin/data/bench_instructions"),
    )
    parser.add_argument("--only-scene", type=str, default=None)
    parser.add_argument("--only-task", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sleep-between", type=float, default=3.0,
                        help="Seconds to sleep between successful uploads (rate-limit friendly).")
    args = parser.parse_args()

    staging = Path(args.staging)
    api = HfApi(token=args.token)

    print("Scanning HF for already-uploaded instructions (resume)...", flush=True)
    existing = set()
    for attempt in range(5):
        try:
            existing = set(api.list_repo_files(REPO, repo_type="dataset"))
            break
        except Exception as e:
            print(f"  scan attempt {attempt}: {e}", flush=True)
            time.sleep(30)
    instr_existing = {p for p in existing if "/instructions/episode" in p and p.endswith(".json")}
    print(f"  found {len(instr_existing)} instruction files already on HF", flush=True)

    plan = []
    for scene_dir in sorted(staging.iterdir()):
        if not scene_dir.is_dir():
            continue
        if args.only_scene and scene_dir.name != args.only_scene:
            continue
        for task_dir in sorted(scene_dir.iterdir()):
            if args.only_task and task_dir.name != args.only_task:
                continue
            for split_dir in sorted(task_dir.iterdir()):
                instr_dir = split_dir / "instructions"
                if not instr_dir.is_dir():
                    continue
                n = len(list(instr_dir.glob("episode*.json")))
                if n == 0:
                    continue
                repo_path = f"{scene_dir.name}/{task_dir.name}/{split_dir.name}/instructions"
                plan.append((instr_dir, repo_path, n))

    # Skip already-uploaded folders.
    full_plan = plan
    plan = [(local, rp, n) for (local, rp, n) in plan
            if not already_uploaded(instr_existing, rp, n)]

    total_files = sum(n for _, _, n in plan)
    print(f"Planned: {len(plan)} folders, {total_files} files (skipped "
          f"{len(full_plan) - len(plan)} already-uploaded)")
    for p in plan[:3]:
        print(" ", p)

    if args.dry_run:
        print("\n[dry-run]")
        return

    fail = []
    for i, (local, repo_path, n) in enumerate(plan, 1):
        print(f"[{i}/{len(plan)}] {repo_path} ({n})", flush=True)
        try:
            upload_with_retry(api, local, repo_path)
            time.sleep(args.sleep_between)
        except Exception as e:
            print(f"  [ERROR] {e}", flush=True)
            fail.append((repo_path, str(e)))
            time.sleep(args.sleep_between * 3)

    print(f"\nDone. {len(plan)-len(fail)} ok, {len(fail)} failed.")
    for p, err in fail[:20]:
        print(f"  FAIL {p}: {err}")


if __name__ == "__main__":
    main()
