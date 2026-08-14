"""
Standalone object viewer: pick an object (and optionally a model id) and render
it close-up in an EMPTY scene (no robot, no table) so you can clearly see what it
looks like and how big it is.

USAGE (run from the benchmark folder):
    cd benchmark
    source set_env.sh

    # list available objects + the office obstacle pool with heights/widths
    python script/bench_script/view_object.py --list

    # render every available id of one object
    python script/bench_script/view_object.py 089_globe

    # render specific id(s)
    python script/bench_script/view_object.py 089_globe --id 2
    python script/bench_script/view_object.py 001_bottle --id 1,14,21

    # override the scale (otherwise uses the clutter/task scale)
    python script/bench_script/view_object.py 089_globe --id 2 --scale 0.12

Images are written to <out-dir> (default ./object_views), one PNG per id, each
labelled with name/id and its standing height (z_max) and footprint radius.
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np

from setup_paths import setup_paths
setup_paths()

bench_root = Path(os.environ["BENCH_ROOT"])
sim_root = Path(os.environ["SIM_ROOT"])
os.chdir(sim_root)

import json
import yaml
import imageio
import cv2
import sapien.core as sapien
from sapien.render import set_global_config
from envs.utils import create_actor
from envs.utils.rand_create_cluttered_actor import (
    get_obstacle_objects_subset, _scale_vec3_from_task_yaml,
)

OBJECTS_DIR = bench_root / "assets" / "objects"


def available_ids(name):
    ids = []
    for f in (OBJECTS_DIR / name).glob("model_data*.json"):
        s = f.stem.replace("model_data", "")
        if s.isdigit():
            ids.append(int(s))
    return sorted(ids)


def obstacle_params():
    """name -> {id(str) -> {scale, z_max, radius}} for the office obstacle pool."""
    try:
        info, _, _ = get_obstacle_objects_subset("office", "objects", [])
        return {n: info[n]["params"] for n in info}
    except Exception as e:
        print(f"(could not load obstacle pool: {e})")
        return {}


def _load_model_config(name, mid):
    p = OBJECTS_DIR / name / f"model_data{mid}.json"
    if p.exists():
        try:
            return json.load(open(p))
        except Exception:
            return None
    return None


def resolve_object(name, mid, params, override):
    """
    Return (scale, height, radius, max_dim) using the SAME scale logic the
    benchmark uses: task_objects.yml `scales:` -> model_data.json `scale` -> [1,1,1].
    Dimensions are computed from model extents so the label and camera framing are
    correct regardless of how big the scale numbers are.
    """
    model_config = _load_model_config(name, mid) or {}

    p = params.get(name, {}).get(str(mid))
    if override is not None:
        scale = [float(override)] * 3
    elif p is not None:                       # office obstacle pool: exact scale
        scale = list(p["scale"])
    else:                                     # general object: replicate system logic
        cfg = yaml.safe_load(open(bench_root / "bench_task_config" / "task_objects.yml")) or {}
        sc = (cfg.get("scales") or {}).get(name)
        if isinstance(sc, dict):
            sc = sc.get(str(mid))
        scale = _scale_vec3_from_task_yaml(sc, model_config)

    ext = model_config.get("extents")
    cen = model_config.get("center", [0, 0, 0])
    if ext:
        # glb objects are stood upright (90deg about x): model y -> world height
        height = (ext[1] + cen[1]) * scale[1]
        radius = (ext[0] * scale[0] + ext[2] * scale[2]) / 4
        max_dim = max(ext[0] * scale[0], ext[1] * scale[1], ext[2] * scale[2])
    else:
        height = radius = None
        max_dim = 0.3
    return scale, height, radius, max_dim


def make_scene():
    """Minimal SAPIEN scene: ground + lights only (no robot, no table)."""
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)
    set_global_config(max_num_materials=50000, max_num_textures=50000)
    # same render config as the benchmark's setup_scene (known-good in this env)
    sapien.render.set_camera_shader_dir("rt")
    sapien.render.set_ray_tracing_samples_per_pixel(32)
    sapien.render.set_ray_tracing_path_depth(8)
    sapien.render.set_ray_tracing_denoiser("oidn")

    scene = engine.create_scene(sapien.SceneConfig())
    scene.set_timestep(1 / 250)
    scene.add_ground(0)
    scene.default_physical_material = scene.create_physical_material(0.5, 0.5, 0)
    scene.set_ambient_light([0.4, 0.4, 0.4])
    scene.add_directional_light([0, 0.5, -1], [0.8, 0.8, 0.8], shadow=True)
    scene.add_point_light([0.4, -0.4, 1.0], [1, 1, 1])
    scene.add_point_light([-0.4, 0.4, 1.0], [1, 1, 1])
    return engine, renderer, scene


def add_camera(scene, w=720, h=720, fovy=38):
    return scene.add_camera(name="view", width=w, height=h,
                            fovy=np.deg2rad(fovy), near=0.05, far=100)


def aim_camera(cam, eye, look_at):
    eye = np.array(eye, dtype=float)
    forward = np.array(look_at, dtype=float) - eye
    forward /= np.linalg.norm(forward)
    left = np.array([-forward[1], forward[0], 0.0])
    if np.linalg.norm(left) < 1e-6:
        left = np.array([0.0, 1.0, 0.0])
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    mat = np.eye(4)
    mat[:3, :3] = np.stack([forward, left, up], axis=1)
    mat[:3, 3] = eye
    cam.entity.set_pose(sapien.Pose(mat))


def render(cam, scene):
    scene.update_render()
    cam.take_picture()
    rgba = cam.get_picture("Color")
    return np.ascontiguousarray((rgba * 255).clip(0, 255).astype(np.uint8)[:, :, :3])


# Every mesh object that can appear in the phase1_handcrafted office scene:
# handcrafted clutter + always-present basic items/furniture + task objects.
# (036_cabinet/wall/floor/box-pad are primitives or articulations, omitted.)
OFFICE_SCENE_OBJECTS = [
    "089_globe", "001_bottle", "090_trophy", "099_fan", "119_mini-chalkboard",  # handcrafted clutter
    "120_plant", "122_file-holder", "121_wall-shelf", "042_wooden_box",          # basic items / furniture
    "047_mouse", "038_milk-box",                                                 # task objects
]


def render_object(scene, cam, name, ids, params, override, out_dir, direction):
    glb_upright = [0.707107, 0.707107, 0, 0]  # stand glb meshes up (same as clutter)
    for mid in ids:
        scale, height, radius, max_dim = resolve_object(name, mid, params, override)
        cz = max(0.15, max_dim * 0.5)  # centre the object above ground
        actor = create_actor(
            scene=scene,
            pose=sapien.Pose([0.0, 0.0, cz], glb_upright),
            modelname=name, model_id=int(mid), convex=True, scale=scale,
        )
        if actor is None:
            print(f"  {name}/{mid}: missing model files, skipping")
            continue
        look_at = np.array([0.0, 0.0, cz])
        eye = look_at + (1.9 * max_dim + 0.3) * direction  # frame camera to object size
        aim_camera(cam, eye, look_at)
        img = render(cam, scene)
        dims = (f"h={height:.3f} r={radius:.3f}" if height is not None
                else f"scale={[round(s, 3) for s in scale]}")
        cv2.putText(img, f"{name} / {mid}   {dims}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        path = out_dir / f"{name}_{mid}.png"
        imageio.imwrite(path, img)
        print(f"  saved {path}  ({dims})")
        actor.actor.set_pose(sapien.Pose([100, 100, 100]))  # remove before next id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("object", nargs="?", default=None, help="object name, e.g. 089_globe")
    ap.add_argument("--id", default=None, help="model id, or comma list (default: all available)")
    ap.add_argument("--scale", type=float, default=None, help="override uniform scale")
    ap.add_argument("--out-dir", default="./object_views")
    ap.add_argument("--list", action="store_true", help="list objects + office obstacle dims and exit")
    ap.add_argument("--catalog", action="store_true",
                    help="render every object (all ids) that can appear in the office handcrafted scene")
    args = ap.parse_args()

    params = obstacle_params()

    if args.list or (args.object is None and not args.catalog):
        print("\n=== office obstacle pool (height z_max / footprint radius) ===")
        rows = [(p["z_max"], p["radius"], name, mid)
                for name, ids in params.items() for mid, p in ids.items()]
        for z, r, name, mid in sorted(rows, reverse=True):
            print(f"  h={z:5.3f}  r={r:5.3f}   {name} / {mid}")
        print("\n=== all object folders in assets/objects ===")
        names = sorted(d.name for d in OBJECTS_DIR.iterdir() if d.is_dir())
        print("  " + ", ".join(names))
        if args.object is None and not args.catalog:
            return

    # decide which (name -> ids) to render
    if args.catalog:
        targets = [(n, available_ids(n)) for n in OFFICE_SCENE_OBJECTS if (OBJECTS_DIR / n).is_dir()]
    else:
        name = args.object
        if not (OBJECTS_DIR / name).is_dir():
            raise SystemExit(f"No object folder: {OBJECTS_DIR / name}")
        ids = ([int(x) for x in str(args.id).split(",")] if args.id is not None
               else available_ids(name))
        if not ids:
            raise SystemExit(f"No model ids found for {name}")
        targets = [(name, ids)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine, renderer, scene = make_scene()
    cam = add_camera(scene)
    direction = np.array([0.6, -0.6, 0.45]); direction /= np.linalg.norm(direction)

    for name, ids in targets:
        print(f"rendering {name} ids={ids}")
        render_object(scene, cam, name, ids, params, args.scale, out_dir, direction)

    print(f"\ndone -> {out_dir}")


if __name__ == "__main__":
    main()
