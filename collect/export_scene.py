"""Per-episode scene geometry export for offline robot-proximity relabeling.

Writes, once per collected episode (after setup_demo, post-settle), the STATIC
scene geometry:

    scene.npz       samples (N,3) f32, normals (N,3) f32, obj_id (N,) i32
    objects.json    per object: per_scene_id, name, tags, pose_p, pose_q, n_samples
    scene_hash.txt  identity hash over (name, id, rounded pose)

(No pointcloud.ply — it carried zero unique data. To view a scene, regenerate it
from scene.npz: downsample samples/normals to <=50k pts and write ascii ply.)

This is the enabler for computing proximity to ANY robot geometry (all links,
all ~N collision spheres) AFTER collection: combine these obstacle surface
samples with the per-frame robot config (joint_action in the hdf5) run through
the robot's sphere/link FK. Nothing about the robot is baked in here — the
export is purely the obstacle side, tagged (target / container / furniture /
clutter / articulation) so the relabel pass decides what to exclude.

Reuses task_env._proximity_mesh_cache (scaled trimeshes; built by
Bench_base_task._init_proximity_tracking) for accurate surface sampling; falls
back to the actor's world AABB box when no mesh is cached.
"""
import hashlib
import json
import os

import numpy as np
import trimesh

_SAMPLE_DENSITY = 0.25e-4   # m² per surface sample (~0.5 cm spacing) for objects
_COARSE_DENSITY = 16e-4     # m² per sample (~4 cm) for furniture / articulations
_SKIP_EXPORT = {"ground", "wall"}   # unbounded planes — never sampled


def _pose_to_mat(pose) -> np.ndarray:
    return np.asarray(pose.to_transformation_matrix(), dtype=np.float64)


def export_scene(task_env, out_dir):
    """Write scene.npz / objects.json / scene_hash.txt into out_dir.
    Call AFTER setup_demo (post-settle poses). Returns the number of objects exported."""
    os.makedirs(out_dir, exist_ok=True)
    mesh_cache   = getattr(task_env, "_proximity_mesh_cache", {})
    target_names = set(getattr(task_env, "target_object_names", set()))
    furn_names   = set(getattr(task_env, "furniture_names", set()))
    container_names = set()
    for attr in ("box", "des_obj"):
        a = getattr(task_env, attr, None)
        try:
            if a is not None:
                container_names.add(a.get_name())
        except Exception:
            pass

    samples, normals, obj_ids, objects = [], [], [], []

    def _sample(mesh, oid, name, tags, pose_p, pose_q):
        density = _COARSE_DENSITY if ("furniture" in tags or "articulation" in tags) \
                  else _SAMPLE_DENSITY
        n = max(16, int(mesh.area / density))
        try:
            pts, fidx = trimesh.sample.sample_surface(mesh, n)
        except Exception:
            return 0
        samples.append(np.asarray(pts, dtype=np.float32))
        normals.append(np.asarray(mesh.face_normals[fidx], dtype=np.float32))
        obj_ids.append(np.full(len(pts), oid, dtype=np.int32))
        objects.append({"per_scene_id": int(oid), "name": name, "tags": sorted(tags),
                        "pose_p": [round(float(x), 4) for x in pose_p],
                        "pose_q": [round(float(x), 4) for x in pose_q],
                        "n_samples": int(len(pts))})
        return len(pts)

    robot = task_env.robot
    robot_entities = {robot.left_entity, robot.right_entity}

    for actor in task_env.scene.get_all_actors():
        name = actor.get_name()
        tags = set()
        if name in target_names:    tags.add("target")
        if name in container_names: tags.add("container")
        if name in furn_names:      tags.add("furniture")
        if not tags:                tags.add("clutter")
        pose = actor.get_pose()
        if name in _SKIP_EXPORT:
            objects.append({"per_scene_id": int(actor.per_scene_id), "name": name,
                            "tags": sorted(tags | {"skipped"}),
                            "pose_p": [round(float(x), 4) for x in pose.p],
                            "pose_q": [round(float(x), 4) for x in pose.q],
                            "n_samples": 0})
            continue
        aabb = None
        for comp in actor.get_components():
            if hasattr(comp, "get_global_aabb_fast"):
                try:
                    aabb = comp.get_global_aabb_fast()
                except RuntimeError:
                    pass
                break
        if aabb is None:
            continue
        T = _pose_to_mat(pose)
        if name in mesh_cache:
            m = mesh_cache[name].copy()
            m.apply_transform(T)
            try:
                m.fix_normals()
            except Exception:
                pass
        else:
            lo = np.asarray(aabb[0], dtype=np.float64)
            hi = np.asarray(aabb[1], dtype=np.float64)
            if np.any(hi - lo < 1e-6):
                continue
            box_T = np.eye(4); box_T[:3, 3] = (lo + hi) / 2
            m = trimesh.creation.box(extents=hi - lo, transform=box_T)
        _sample(m, actor.per_scene_id, name, tags, pose.p, pose.q)

    # Articulations (fridge/cabinet/...): AABB box per collision link, negative ids.
    target_pos = [np.asarray(a.get_pose().p, dtype=np.float64)
                  for a in task_env.scene.get_all_actors()
                  if a.get_name() in target_names]
    for j, art in enumerate(task_env.scene.get_all_articulations()):
        if art in robot_entities:
            continue
        try:
            art_name = art.get_name() or f"articulation_{j}"
        except Exception:
            art_name = f"articulation_{j}"
        boxes = []
        for link in art.get_links():
            if not link.collision_shapes:
                continue
            for comp in link.entity.get_components():
                if hasattr(comp, "get_global_aabb_fast"):
                    try:
                        aabb = comp.get_global_aabb_fast()
                    except RuntimeError:
                        break
                    lo = np.asarray(aabb[0], dtype=np.float64)
                    hi = np.asarray(aabb[1], dtype=np.float64)
                    if np.all(hi - lo > 1e-6):
                        boxes.append((lo, hi))
                    break
        if not boxes:
            continue
        tags = {"articulation"}
        art_lo = np.min([b[0] for b in boxes], axis=0) - 0.02
        art_hi = np.max([b[1] for b in boxes], axis=0) + 0.02
        if any(np.all(p >= art_lo) and np.all(p <= art_hi) for p in target_pos):
            tags.add("container")
        oid = -(1000 + j)
        box_meshes = []
        for lo, hi in boxes:
            box_T = np.eye(4); box_T[:3, 3] = (lo + hi) / 2
            box_meshes.append(trimesh.creation.box(extents=hi - lo, transform=box_T))
        mesh = trimesh.util.concatenate(box_meshes)
        center = (art_lo + art_hi) / 2
        _sample(mesh, oid, art_name, tags, center, [1, 0, 0, 0])

    if not samples:
        raise RuntimeError("export_scene: no geometry exported")
    S  = np.concatenate(samples)
    Nm = np.concatenate(normals)
    I  = np.concatenate(obj_ids)
    np.savez_compressed(os.path.join(out_dir, "scene.npz"),
                        samples=S, normals=Nm, obj_id=I)
    with open(os.path.join(out_dir, "objects.json"), "w") as f:
        json.dump(objects, f, indent=1)
    ident = sorted((o["name"], o["per_scene_id"], tuple(o["pose_p"]), tuple(o["pose_q"]))
                   for o in objects)
    h = hashlib.sha1(repr(ident).encode()).hexdigest()
    with open(os.path.join(out_dir, "scene_hash.txt"), "w") as f:
        f.write(h + "\n")
    return len(objects)
