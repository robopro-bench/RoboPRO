#!/usr/bin/env python3
"""
Validate OOD obstacle objects registered in task_objects.yml.

For each object_ood obstacle entry, checks:
  1. model_data{id}.json exists and has valid center/extents fields
  2. Visual and collision mesh files exist on disk
  3. Object loads in a minimal SAPIEN scene without error
  4. Object settles stably after 500 physics steps (no fall-through, no fly-away)
  5. No seen/OOD variant ID overlaps

Usage:
    source set_env.sh  # repo root
    cd sim
    python ../scripts/validation/validate_ood_objects.py

    # Validate a single scene:
    python ../scripts/validate_ood_objects.py --scene office

    # Skip SAPIEN physics (metadata-only checks):
    python ../scripts/validate_ood_objects.py --no-physics
"""
import argparse
import json
import os
import sys
from pathlib import Path

import yaml


def check_metadata(objects_dir, model_name, model_id):
    """Check that model_data{id}.json exists and has required fields."""
    json_path = objects_dir / model_name / f"model_data{model_id}.json"
    if not json_path.exists():
        return False, f"model_data{model_id}.json missing"
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as e:
        return False, f"model_data{model_id}.json parse error: {e}"
    if "center" not in data:
        return False, "missing 'center' field"
    if "extents" not in data:
        return False, "missing 'extents' field"
    return True, data


def check_mesh_files(objects_dir, model_name, model_id):
    """Check that visual and collision mesh files exist."""
    model_dir = objects_dir / model_name
    collision_dir = model_dir / "collision"
    visual_dir = model_dir / "visual"

    def find_mesh(base_dir, mid):
        for ext in [f"base{mid}.glb", f"base{mid}.obj", f"textured{mid}.obj"]:
            if (base_dir / ext).exists():
                return True
        if mid is not None:
            return False
        for ext in ["base.glb", "textured.obj"]:
            if (base_dir / ext).exists():
                return True
        return False

    has_collision = find_mesh(collision_dir, model_id) if collision_dir.exists() else find_mesh(model_dir, model_id)
    has_visual = find_mesh(visual_dir, model_id) if visual_dir.exists() else find_mesh(model_dir, model_id)

    if not has_collision:
        return False, "collision mesh not found"
    if not has_visual:
        return False, "visual mesh not found"
    return True, "ok"


def check_physics_stability(scene_engine, objects_dir, model_name, model_id, scale_override=None):
    """Load object in SAPIEN, drop it, check it settles stably."""
    import sapien
    import sapien.physx
    import numpy as np

    scene = sapien.Scene()
    scene.set_timestep(1 / 250)
    scene.add_ground(0)
    scene.default_physical_material = scene.create_physical_material(0.5, 0.5, 0)
    scene.set_ambient_light([0.5, 0.5, 0.5])

    json_path = objects_dir / model_name / f"model_data{model_id}.json"
    try:
        with open(json_path) as f:
            model_data = json.load(f)
    except Exception:
        model_data = {}

    if scale_override is not None:
        if isinstance(scale_override, (int, float)):
            scale = [scale_override] * 3
        else:
            scale = list(scale_override)
    elif "scale" in model_data:
        s = model_data["scale"]
        scale = [s, s, s] if isinstance(s, (int, float)) else list(s)
    else:
        scale = [1.0, 1.0, 1.0]

    model_dir = objects_dir / model_name
    collision_dir = model_dir / "collision"
    visual_dir = model_dir / "visual"

    def find_file(base_dir, mid):
        for pattern in [f"base{mid}.glb", f"textured{mid}.obj"]:
            p = base_dir / pattern
            if p.exists():
                return p
        return None

    col_file = None
    vis_file = None
    if collision_dir.exists():
        col_file = find_file(collision_dir, model_id)
    if col_file is None:
        col_file = find_file(model_dir, model_id)
    if visual_dir.exists():
        vis_file = find_file(visual_dir, model_id)
    if vis_file is None:
        vis_file = find_file(model_dir, model_id)

    if col_file is None or vis_file is None:
        scene = None
        return False, "mesh file not found"

    try:
        builder = scene.create_actor_builder()
        builder.add_multiple_convex_collisions_from_file(filename=str(col_file), scale=scale)
        builder.add_visual_from_file(filename=str(vis_file), scale=scale)
        spawn_pose = sapien.Pose([0, 0, 0.5])
        actor = builder.build(name=f"{model_name}_{model_id}")
        actor.set_pose(spawn_pose)
    except Exception as e:
        scene = None
        return False, f"load failed: {e}"

    initial_pos = np.array(actor.get_pose().p)

    for _ in range(500):
        scene.step()

    final_pos = np.array(actor.get_pose().p)
    rb = actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    vel = np.array(rb.linear_velocity) if rb else np.zeros(3)

    scene = None

    if final_pos[2] < -0.1:
        return False, f"fell through ground (z={final_pos[2]:.3f})"

    displacement = np.linalg.norm(final_pos[:2] - initial_pos[:2])
    if displacement > 0.5:
        return False, f"slid too far (displacement={displacement:.3f}m)"

    speed = np.linalg.norm(vel)
    if speed > 0.5:
        return False, f"still moving (speed={speed:.3f}m/s)"

    return True, f"settled at z={final_pos[2]:.3f}"


def check_id_overlaps(task_cfg):
    """Check for seen/OOD variant ID overlaps."""
    objects = task_cfg.get("objects", {})
    object_ood = task_cfg.get("object_ood", {})
    bugs = []

    for scene in objects:
        seen_ids = {}
        for role in ["targets", "obstacles"]:
            pool = objects.get(scene, {}).get(role, {})
            if role == "obstacles":
                for size in ["short", "tall"]:
                    for obj, ids in (pool.get(size, {}) or {}).items():
                        seen_ids.setdefault(obj, set()).update(str(i) for i in ids)
            else:
                for obj, ids in pool.items():
                    seen_ids.setdefault(obj, set()).update(str(i) for i in ids)

        ood_ids = {}
        for role in ["targets", "obstacles"]:
            pool = object_ood.get(scene, {}).get(role, {})
            if role == "obstacles":
                for size in ["short", "tall"]:
                    for obj, ids in (pool.get(size, {}) or {}).items():
                        ood_ids.setdefault(obj, set()).update(str(i) for i in ids)
            else:
                for obj, ids in (pool or {}).items():
                    ood_ids.setdefault(obj, set()).update(str(i) for i in ids)

        for obj in set(seen_ids) | set(ood_ids):
            overlap = seen_ids.get(obj, set()) & ood_ids.get(obj, set())
            if overlap:
                bugs.append((scene, obj, sorted(overlap)))

    return bugs


def main():
    parser = argparse.ArgumentParser(description="Validate OOD obstacle objects")
    parser.add_argument("--scene", type=str, default=None, help="Validate a single scene (office/study/kitchens/kitchenl)")
    parser.add_argument("--no-physics", action="store_true", help="Skip SAPIEN physics checks (metadata only)")
    parser.add_argument("--task-objects", type=str, default=None, help="Path to task_objects.yml")
    args = parser.parse_args()

    bench_root = os.environ.get("BENCH_ROOT")
    if bench_root:
        bench_root = Path(bench_root)
    else:
        bench_root = Path(__file__).resolve().parent.parent / "benchmark"
        if not bench_root.exists():
            print("ERROR: Set BENCH_ROOT or source set_env.sh from the repo root")
            sys.exit(1)

    assets_root = Path(os.environ.get("ASSETS_ROOT") or Path(__file__).resolve().parents[2] / "assets")
    objects_dir = assets_root / "objects"
    task_objects_path = Path(args.task_objects) if args.task_objects else bench_root / "bench_task_config" / "task_objects.yml"

    with open(task_objects_path) as f:
        task_cfg = yaml.safe_load(f)

    object_ood = task_cfg.get("object_ood", {})
    scales_cfg = task_cfg.get("scales", {}) or {}
    scenes = [args.scene] if args.scene else ["office", "study", "kitchens", "kitchenl"]

    # ID overlap check
    print("=" * 70)
    print("SEEN/OOD ID OVERLAP CHECK")
    print("=" * 70)
    bugs = check_id_overlaps(task_cfg)
    if bugs:
        for scene, obj, ids in bugs:
            print(f"  BUG [{scene}] {obj}: IDs {ids} in both seen and OOD")
    else:
        print("  No overlaps found.")

    # Set up SAPIEN if needed
    scene_engine = None
    if not args.no_physics:
        import sapien
        sapien.render.set_camera_shader_dir("default")
        scene_engine = True

    total_pass = 0
    total_fail = 0
    total_skip = 0
    failures = []

    for scene in scenes:
        ood_scene = object_ood.get(scene, {})
        obstacles = ood_scene.get("obstacles", {})

        print(f"\n{'=' * 70}")
        print(f"SCENE: {scene.upper()}")
        print(f"{'=' * 70}")

        for size in ["short", "tall"]:
            pool = obstacles.get(size, {})
            if not pool:
                continue

            print(f"\n  [{size}]")
            for model_name, ids in pool.items():
                for mid in ids:
                    mid_str = str(mid)
                    label = f"    {model_name}/{mid_str}"

                    # Metadata check
                    meta_ok, meta_result = check_metadata(objects_dir, model_name, mid_str)
                    if not meta_ok:
                        print(f"{label:<45} FAIL  metadata: {meta_result}")
                        total_fail += 1
                        failures.append((scene, size, model_name, mid_str, f"metadata: {meta_result}"))
                        continue

                    # Mesh file check
                    mesh_ok, mesh_result = check_mesh_files(objects_dir, model_name, mid_str)
                    if not mesh_ok:
                        print(f"{label:<45} FAIL  {mesh_result}")
                        total_fail += 1
                        failures.append((scene, size, model_name, mid_str, mesh_result))
                        continue

                    # Physics check
                    if args.no_physics:
                        print(f"{label:<45} PASS  (physics skipped)")
                        total_pass += 1
                        continue

                    scale_entry = scales_cfg.get(model_name)
                    scale_override = None
                    if isinstance(scale_entry, dict):
                        scale_override = scale_entry.get(mid_str)
                    elif scale_entry is not None:
                        scale_override = scale_entry

                    phys_ok, phys_result = check_physics_stability(scene_engine, objects_dir, model_name, mid_str, scale_override)
                    if phys_ok:
                        print(f"{label:<45} PASS  {phys_result}")
                        total_pass += 1
                    else:
                        print(f"{label:<45} FAIL  {phys_result}")
                        total_fail += 1
                        failures.append((scene, size, model_name, mid_str, phys_result))

    # Summary
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Passed: {total_pass}")
    print(f"  Failed: {total_fail}")
    if failures:
        print(f"\n  Failures:")
        for scene, size, model, mid, reason in failures:
            print(f"    [{scene}/{size}] {model}/{mid}: {reason}")

    id_bug_count = len(bugs)
    if id_bug_count:
        print(f"\n  ID overlap bugs: {id_bug_count}")

    if total_fail == 0 and id_bug_count == 0:
        print("\n  All checks passed.")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
