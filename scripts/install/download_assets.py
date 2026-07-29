#!/usr/bin/env python
"""Fetch RoboPRO benchmark assets from HuggingFace into benchmark/assets/.

Pulls four zip bundles (~15 GB total) from RoboPRO/RoboPRO_assets and
extracts them in place:

    benchmark/assets/objects/              (~3 GB)
    benchmark/assets/embodiments/          (~750 MB)
    benchmark/assets/background_texture/   (~11 GB)
    benchmark/assets/backgrounds/          (~24 MB)

Usage:
    python scripts/install/download_assets.py [--dest <path>] [--keep-zips]
"""
import argparse
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "RoboPRO/RoboPRO_assets"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEST = REPO_ROOT / "benchmark" / "assets"
BUNDLES = ["backgrounds.zip", "embodiments.zip", "objects.zip", "background_texture.zip"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                        help=f"target directory (default: {DEFAULT_DEST})")
    parser.add_argument("--keep-zips", action="store_true",
                        help="don't delete the .zip files after extracting")
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)

    for bundle in BUNDLES:
        print(f"[download] {bundle} → {args.dest}")
        zip_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=bundle,
            repo_type="dataset",
            local_dir=str(args.dest),
        )
        print(f"[extract] {bundle}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(args.dest)
        if not args.keep_zips:
            Path(zip_path).unlink()

    # Clean HF cache dir created in the dest by hf_hub_download
    cache_dir = args.dest / ".cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    print(f"[done] assets in {args.dest}. Verify: ls {args.dest}/objects | wc -l (expect ~81)")


if __name__ == "__main__":
    main()
