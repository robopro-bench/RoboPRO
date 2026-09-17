#!/usr/bin/env python3
"""Render curobo_{left,right}.yml from the shipped *_tmp.yml templates.

CuRobo templates use ${ASSETS_PATH}/assets/embodiments/..., so ASSETS_PATH is
the parent of the assets/ directory (normally the repo root). Mesh files are
read from $ASSETS_ROOT (default: <repo>/assets).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BLUE, YELLOW, GREEN, NC = "\033[0;34m", "\033[0;33m", "\033[0;32m", "\033[0m"


def c(msg: str, color: str) -> None:
    print(f"{color}{msg}{NC}")


def resolve_roots() -> tuple[Path, Path]:
    assets_root = Path(os.environ.get("ASSETS_ROOT") or (REPO_ROOT / "assets")).resolve()
    # Templates expand ${ASSETS_PATH}/assets/embodiments/...
    assets_path = Path(os.environ.get("ASSETS_PATH") or assets_root.parent).resolve()
    return assets_root, assets_path


def main() -> None:
    assets_root, assets_path = resolve_roots()
    emb_dir = assets_root / "embodiments"
    if not emb_dir.is_dir():
        c(f"Error: {emb_dir} not found — run scripts/install/download_assets.py first", YELLOW)
        sys.exit(1)

    c(f"ASSETS_ROOT={assets_root}", BLUE)
    c(f"ASSETS_PATH={assets_path}  (used as ${{ASSETS_PATH}}/assets/embodiments/...)", BLUE)

    templates = sorted(emb_dir.rglob("*_tmp.yml"))
    if not templates:
        c(f"No *_tmp.yml files under {emb_dir}", YELLOW)
        sys.exit(1)

    n_ok = n_err = 0
    for tmp_file in templates:
        target = tmp_file.with_name(tmp_file.name.replace("_tmp.yml", ".yml"))
        print(f"Processing: {tmp_file} -> {target}")
        try:
            content = tmp_file.read_text(encoding="utf-8")
            new_content = content.replace("${ASSETS_PATH}", str(assets_path))
            new_content = new_content.replace("$ASSETS_PATH", str(assets_path))
            target.write_text(new_content, encoding="utf-8")
            c(f"  replaced ${{ASSETS_PATH}} -> {assets_path}", GREEN)
            n_ok += 1
        except OSError as exc:
            c(f"  failed: {exc}", YELLOW)
            n_err += 1

    print()
    c(f"Updated {n_ok} file(s)", GREEN)
    if n_err:
        c(f"Failed {n_err} file(s)", YELLOW)
        sys.exit(1)


if __name__ == "__main__":
    main()
