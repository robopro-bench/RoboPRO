#!/usr/bin/env python3
"""Fetch RoboPRO assets from HuggingFace into repo-root assets/.

Pulls four zip bundles (~15 GB total) from Hoshipu/RoboPRO_assets and
extracts them in place:

    assets/objects/              (~3 GB)
    assets/embodiments/          (~750 MB)
    assets/background_texture/   (~11 GB)
    assets/backgrounds/          (~24 MB)

Usage:
    python scripts/install/download_assets.py [--dest <path>] [--keep-zips]

Dest defaults to $ASSETS_DEST, then $ASSETS_ROOT, then <repo>/assets.
After download: make configure-curobo-assets && make patch-curobo-config
"""
from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "Hoshipu/RoboPRO_assets"
REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLES = ["backgrounds.zip", "embodiments.zip", "objects.zip", "background_texture.zip"]
EXPECTED_DIRS = ("objects", "embodiments", "background_texture", "backgrounds")


def default_dest() -> Path:
    env = os.environ.get("ASSETS_DEST") or os.environ.get("ASSETS_ROOT")
    return Path(env) if env else REPO_ROOT / "assets"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=default_dest(),
                        help=f"target directory (default: {default_dest()})")
    parser.add_argument("--keep-zips", action="store_true",
                        help="don't delete the .zip files after extracting")
    args = parser.parse_args()

    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    for bundle in BUNDLES:
        print(f"[download] {bundle} → {dest}")
        zip_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=bundle,
            repo_type="dataset",
            local_dir=str(dest),
        )
        print(f"[extract] {bundle}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
        if not args.keep_zips:
            Path(zip_path).unlink()

    cache_dir = dest / ".cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    missing = [name for name in EXPECTED_DIRS if not (dest / name).is_dir()]
    if missing:
        raise SystemExit(f"[error] extract finished but missing {missing} under {dest}")

    n_objects = sum(1 for p in (dest / "objects").iterdir() if p.is_dir())
    print(f"[done] assets in {dest} ({n_objects} object dirs; expect ~81)")
    print("[next] make configure-curobo-assets && make patch-curobo-config")


if __name__ == "__main__":
    main()
