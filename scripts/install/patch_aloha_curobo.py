#!/usr/bin/env python
"""Patch aloha-agilex curobo configs so `attach_external_objects_to_robot` works.

The HuggingFace asset bundle ships `curobo_{left,right}{_tmp,}.yml` without an
`attached_object` link. CuRobo's `MotionGen.attach_external_objects_to_robot`
needs that link declared in three places (extra_links, collision_link_names,
self_collision_ignore) AND backed by sphere allocations (already present in
collision_aloha_{left,right}.yml). Without this patch, runtime fails with:

    KeyError: 'attached_object'

or returns `n_spheres == 0` and the assert in planner.attach_object fires.

Idempotent: re-running is a no-op.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EMB_DIR = REPO_ROOT / "assets" / "embodiments" / "aloha-agilex"

EXTRA_LINKS_BLOCK = """    extra_links:
      attached_object:
        link_name: "attached_object"
        parent_link_name: "{ee_link}"
        joint_name: "attached_object_joint"
        joint_type: "FIXED"
        fixed_transform: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]"""

SIDES = {
    "left":  {"prefix": "fl", "camera": "left_camera"},
    "right": {"prefix": "fr", "camera": "right_camera"},
}


def patch_one(path: Path, side: str) -> bool:
    pre = SIDES[side]["prefix"]
    cam = SIDES[side]["camera"]
    text = path.read_text()
    changed = False

    if '"attached_object"' not in text:
        text = text.replace(
            f'        "{cam}"\n      ]\n    collision_spheres',
            f'        "{cam}",\n        "attached_object"\n      ]\n    collision_spheres',
            1,
        )
        text = text.replace(
            f'        "{pre}_link7": ["{pre}_link8", "{cam}"]\n      }}',
            f'        "{pre}_link7": ["{pre}_link8", "{cam}"],\n'
            f'        "attached_object": ["{pre}_link6", "{pre}_link7", "{pre}_link8"]\n      }}',
            1,
        )
        changed = True

    if "extra_links: null" in text:
        text = text.replace(
            "    extra_links: null",
            EXTRA_LINKS_BLOCK.format(ee_link=f"{pre}_link6"),
            1,
        )
        changed = True

    if changed:
        path.write_text(text)
        print(f"[patch] {path.relative_to(REPO_ROOT)}")
    else:
        print(f"[skip ] {path.relative_to(REPO_ROOT)} (already patched)")
    return changed


def main() -> None:
    if not EMB_DIR.exists():
        raise SystemExit(f"missing {EMB_DIR} — run download_assets.py first")
    for side in ("left", "right"):
        for stem in (f"curobo_{side}.yml", f"curobo_{side}_tmp.yml"):
            patch_one(EMB_DIR / stem, side)


if __name__ == "__main__":
    main()
