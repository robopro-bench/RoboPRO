import os
import sapien.core as sapien
import numpy as np
import transforms3d as t3d
import sapien.physx as sapienp
from .create_actor import *

import re
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml


def _model_config_scale_vec3(model_config: dict) -> list[float]:
    """Normalize model_data.json ``scale`` to a length-3 vector."""
    default = model_config.get("scale", [1.0, 1.0, 1.0])
    if isinstance(default, (int, float)):
        s = float(default)
        return [s, s, s]
    if isinstance(default, (list, tuple)):
        if len(default) == 1:
            s = float(default[0])
            return [s, s, s]
        if len(default) >= 3:
            return [float(default[0]), float(default[1]), float(default[2])]
    return [1.0, 1.0, 1.0]


def _scale_vec3_from_task_yaml(scale_val, model_config: dict) -> list[float]:
    """
    Resolve scale from task_objects.yml ``scales:`` entry.
    Supports scalar or per-axis list (e.g. 025_chips-tub ``0``: [0.08, 0.08, 0.04]).
    """
    if scale_val is None:
        return _model_config_scale_vec3(model_config)
    if isinstance(scale_val, (int, float)):
        s = float(scale_val)
        return [s, s, s]
    if isinstance(scale_val, (list, tuple)):
        if len(scale_val) == 1:
            s = float(scale_val[0])
            return [s, s, s]
        if len(scale_val) >= 3:
            return [float(scale_val[0]), float(scale_val[1]), float(scale_val[2])]
    return _model_config_scale_vec3(model_config)


def get_all_cluttered_objects():
    cluttered_objects_info = {}
    cluttered_objects_name = []

    # load from cluttered_objects
    cluttered_objects_config = json.load(open(Path(os.environ["BENCH_ROOT"]) / "assets" / "objects" / "objaverse" / "list.json", "r", encoding="utf-8"))
    cluttered_objects_name += cluttered_objects_config["item_names"]
    for model_name, model_ids in cluttered_objects_config["list_of_items"].items():
        cluttered_objects_info[model_name] = {
            "ids": model_ids,
            "type": "urdf",
            "root": f"objects/objaverse/{model_name}",
        }
        params = {}
        for model_id in model_ids:
            model_full_name = f"{model_name}_{model_id}"
            params[model_id] = {
                "z_max": cluttered_objects_config["z_max"][model_full_name],
                "radius": cluttered_objects_config["radius"][model_full_name],
                "z_offset": cluttered_objects_config["z_offset"][model_full_name],
            }
        cluttered_objects_info[model_name]["params"] = params

    # load from objects
    objects_dir = Path(os.environ["BENCH_ROOT"]) / "assets" / "objects"
    for model_dir in objects_dir.iterdir():
        if not model_dir.is_dir():
            continue
        if re.search(r"^(\d+)_(.*)", model_dir.name) is None:
            continue
        model_name = model_dir.name
        model_id_list, params = [], {}
        for model_cfg in model_dir.iterdir():
            if model_cfg.is_dir() or model_cfg.suffix != ".json":
                continue

            # get model id
            model_id = re.search(r"model_data(\d+)", model_cfg.name)
            if not model_id:
                continue
            model_id = model_id.group(1)

            try:
                # get model params
                model_config: dict = json.load(open(model_cfg, "r", encoding="utf-8"))
                if "center" not in model_config or "extents" not in model_config:
                    continue
                # if model_config.get("stable", False) is False:
                #     continue
                center = model_config["center"]
                extents = model_config["extents"]
                scale = model_config.get("scale", [1.0, 1.0, 1.0])
                # 0: x, 1: z, 2: y
                params[model_id] = {
                    "z_max": (extents[1] + center[1]) * scale[1],
                    "radius": max(extents[0] * scale[0], extents[2] * scale[2]) / 2,
                    "z_offset": 0,
                }
                model_id_list.append(model_id)
            except Exception as e:
                print(f"Error loading model config {model_cfg}: {e}")
        if len(model_id_list) == 0:
            continue
        cluttered_objects_name.append(model_name)
        model_id_list.sort()
        cluttered_objects_info[model_name] = {
            "ids": model_id_list,
            "type": "glb",
            "root": f"objects/{model_name}",
            "params": params,
        }

    same_obj = json.load(open(Path(os.environ["BENCH_ROOT"]) / "assets" / "objects" / "same.json", "r", encoding="utf-8"))
    cluttered_objects_name = list(cluttered_objects_name)
    cluttered_objects_name.sort()
    return cluttered_objects_info, cluttered_objects_name, same_obj


def get_obstacle_objects_subset(env_name: str, distribution: str, entities_on_scene: list):
    """
    Load cluttered object info for the obstacle objects of the given env from
    benchmark/bench_task_config/task_objects.yml. Only includes model ids listed
    for each obstacle. Excludes objects already in entities_on_scene. Uses scales
    from the YAML to compute params.

    It returns:
        - cluttered_objects_info: dict model_name -> {ids, type, root, params}
          containing all models (both short and tall) that are allowed and not
          already on the scene.
        - cluttered_objects_name_short: list of model names that come from the
          "short" obstacle group and are actually available.
        - cluttered_objects_name_tall: list of model names that come from the
          "tall" obstacle group and are actually available.
    """
    same_obj = json.load(open(Path(f"{os.environ['BENCH_ROOT']}/bench_task_config/object_type_equivalencies.json"), "r", encoding="utf-8"))

    # loading scales and obstacles from task_objects.yml
    task_cfg_path = Path(f"{os.environ['BENCH_ROOT']}/bench_task_config/task_objects.yml")
    with open(task_cfg_path, "r", encoding="utf-8") as f:
        task_cfg = yaml.safe_load(f) or {}
    scales_cfg = task_cfg.get("scales", {}) or {}
    distribution_cfg = task_cfg.get(distribution) or {}
    env_cfg = distribution_cfg.get(env_name) or {}
    obstacles_cfg = env_cfg.get("obstacles") or {}

    short_cfg = obstacles_cfg.get("short") or {}
    tall_cfg = obstacles_cfg.get("tall") or {}

    if not short_cfg and not tall_cfg:
        return {}, [], []

    # filter out models that are already on scene
    models_in_use = set()
    for entity_name in entities_on_scene:
        if same_obj.get(entity_name) is not None:
            models_in_use.update(same_obj[entity_name])
        models_in_use.add(entity_name)

    cluttered_objects_info = {}
    cluttered_objects_name_short = []
    cluttered_objects_name_tall = []
    objects_dir = Path(os.environ["BENCH_ROOT"]) / "assets" / "objects"

    def _process_group(group_cfg, group_name_list):
        nonlocal cluttered_objects_info

        if not group_cfg:
            return

        # obstacle name -> allowed model id strings
        allowed_ids_by_obj = {
            obj_name: set(str(i) for i in id_list)
            for obj_name, id_list in group_cfg.items()
        }

        requested_names = set(allowed_ids_by_obj.keys())
        allowed_names = requested_names - models_in_use

        for model_name in sorted(allowed_names):
            model_dir = objects_dir / model_name
            allowed_ids = allowed_ids_by_obj.get(model_name, set())
            model_id_list = []
            params = {}
            for model_id in allowed_ids:
                model_cfg = model_dir / f"model_data{model_id}.json"
                try:
                    model_config = json.load(open(model_cfg, "r", encoding="utf-8"))
                    if "center" not in model_config or "extents" not in model_config:
                        continue
                    center = model_config["center"]
                    extents = model_config["extents"]

                    obj_scale_entry = scales_cfg.get(model_name)
                    if isinstance(obj_scale_entry, dict):
                        scale_val = obj_scale_entry.get(str(model_id), None)
                        scale = _scale_vec3_from_task_yaml(scale_val, model_config)
                    else:
                        scale = _scale_vec3_from_task_yaml(None, model_config)

                    params[model_id] = {
                        "z_max": (extents[1] + center[1]) * scale[1],
                        "radius": (extents[0] * scale[0] + extents[2] * scale[2]) / 4,
                        "z_offset": 0,
                        "scale": scale,
                    }
                    model_id_list.append(model_id)
                except Exception as e:
                    print(f"Error loading model config {model_cfg}: {e}")

            if len(model_id_list) == 0:
                continue
            model_id_list.sort()

            # Merge into shared dict; last group wins if duplicated.
            cluttered_objects_info[model_name] = {
                "ids": model_id_list,
                "type": "glb",
                "root": f"objects/{model_name}",
                "params": params,
            }
            group_name_list.append(model_name)

    _process_group(short_cfg, cluttered_objects_name_short)
    _process_group(tall_cfg, cluttered_objects_name_tall)

    cluttered_objects_name_short = sorted(set(cluttered_objects_name_short))
    cluttered_objects_name_tall = sorted(set(cluttered_objects_name_tall))

    return cluttered_objects_info, cluttered_objects_name_short, cluttered_objects_name_tall


def get_target_objects_subset(env_name: str, distribution: str):
    """
    Load target object info for the given env/distribution from
    benchmark/bench_task_config/task_objects.yml.

    Returns:
        target_objects_info: dict model_name -> {ids, type, root, params}
    """
    task_cfg_path = Path(f"{os.environ['BENCH_ROOT']}/bench_task_config/task_objects.yml")
    with open(task_cfg_path, "r", encoding="utf-8") as f:
        task_cfg = yaml.safe_load(f) or {}

    scales_cfg = task_cfg.get("scales", {}) or {}
    distribution_cfg = task_cfg.get(distribution) or {}
    env_cfg = distribution_cfg.get(env_name) or {}
    targets_cfg = env_cfg.get("targets") or {}

    if not targets_cfg:
        return {}

    target_objects_info = {}
    objects_dir = Path(os.environ["BENCH_ROOT"]) / "assets" / "objects"

    # target name -> allowed model id strings
    allowed_ids_by_obj = {
        obj_name: set(str(i) for i in id_list)
        for obj_name, id_list in targets_cfg.items()
    }

    for model_name in sorted(allowed_ids_by_obj.keys()):
        model_dir = objects_dir / model_name
        allowed_ids = allowed_ids_by_obj.get(model_name, set())
        model_id_list = []
        params = {}

        for model_id in allowed_ids:
            model_cfg = model_dir / f"model_data{model_id}.json"
            try:
                model_config = json.load(open(model_cfg, "r", encoding="utf-8"))
                if "center" not in model_config or "extents" not in model_config:
                    continue
                center = model_config["center"]
                extents = model_config["extents"]

                obj_scale_entry = scales_cfg.get(model_name)
                if isinstance(obj_scale_entry, dict):
                    scale_val = obj_scale_entry.get(str(model_id), None)
                    scale = _scale_vec3_from_task_yaml(scale_val, model_config)
                else:
                    scale = _scale_vec3_from_task_yaml(None, model_config)

                params[model_id] = {
                    "z_max": (extents[1] + center[1]) * scale[1],
                    "radius": (extents[0] * scale[0] + extents[2] * scale[2]) / 4,
                    "z_offset": 0,
                    "scale": scale,
                }
                model_id_list.append(model_id)
            except Exception as e:
                print(f"Error loading model config {model_cfg}: {e}")

        if len(model_id_list) == 0:
            continue

        model_id_list.sort()
        target_objects_info[model_name] = {
            "ids": model_id_list,
            "type": "glb",
            "root": f"objects/{model_name}",
            "params": params,
        }
    return target_objects_info


cluttered_objects_info, cluttered_objects_list, same_obj = get_all_cluttered_objects()

def get_cluttered_objects_info():
    global cluttered_objects_info
    return cluttered_objects_info


def get_available_cluttered_objects(entity_on_scene: list):
    global cluttered_objects_info, cluttered_objects_list, same_obj

    model_in_use = []
    for entity_name in entity_on_scene:
        if same_obj.get(entity_name) is not None:
            model_in_use += same_obj[entity_name]
        model_in_use.append(entity_name)

    available_models = set(cluttered_objects_list) - set(model_in_use)
    available_models = list(available_models)
    available_models.sort()
    return available_models, cluttered_objects_info



def check_overlap(radius, x, y, area):
    if x <= area[0]:
        dx = area[0] - x
    elif area[0] < x and x < area[2]:
        dx = 0
    elif x >= area[2]:
        dx = x - area[2]
    if y <= area[1]:
        dy = area[1] - y
    elif area[1] < y and y < area[3]:
        dy = 0
    elif y >= area[3]:
        dy = y - area[3]

    return dx * dx + dy * dy <= radius * radius


def rand_pose_cluttered(
    xlim: np.ndarray,
    ylim: np.ndarray,
    zlim: np.ndarray,
    ylim_prop=False,
    rotate_rand=False,
    rotate_lim=[0, 0, 0],
    qpos=[1, 0, 0, 0],
    size_dict=None,
    obj_radius=0.1,
    z_offset=0.001,
    z_max=0,
    prohibited_area=None,
    obj_margin=0.005,
) -> sapien.Pose:
    if len(xlim) < 2 or xlim[1] < xlim[0]:
        xlim = np.array([xlim[0], xlim[0]])
    if len(ylim) < 2 or ylim[1] < ylim[0]:
        ylim = np.array([ylim[0], ylim[0]])
    if len(zlim) < 2 or zlim[1] < zlim[0]:
        zlim = np.array([zlim[0], zlim[0]])

    times = 0
    while True:
        times += 1
        if times > 100:
            return False, None
        x = np.random.uniform(xlim[0], xlim[1])
        y = np.random.uniform(ylim[0], ylim[1])
        new_obj_radius = obj_radius + obj_margin
        is_overlap = False
        for area in prohibited_area:
            if check_overlap(new_obj_radius, x, y, area):
                is_overlap = True
                break
        if is_overlap:
            continue
        distances = np.sqrt((np.array([sub_list[0] for sub_list in size_dict]) - x)**2 +
                            (np.array([sub_list[1] for sub_list in size_dict]) - y)**2)
        max_distances = np.array([sub_list[3] + new_obj_radius + obj_margin for sub_list in size_dict])

        if y - new_obj_radius < 0:
            if z_max > 0.05:
                continue
        if (x - new_obj_radius < -0.6 or x + new_obj_radius > 0.6 or y - new_obj_radius < -0.34
                or y + new_obj_radius > 0.34):
            continue
        if np.all(distances > max_distances) and y + new_obj_radius < ylim[1]:
            break

    z = np.random.uniform(zlim[0], zlim[1])
    z = z - z_offset

    rotate = qpos
    if rotate_rand:
        angles = [0, 0, 0]
        for i in range(3):
            angles[i] = np.random.uniform(-rotate_lim[i], rotate_lim[i])
        rotate_quat = t3d.euler.euler2quat(angles[0], angles[1], angles[2])
        rotate = t3d.quaternions.qmult(rotate, rotate_quat)

    return True, sapien.Pose([x, y, z], rotate)

def rand_pose_cluttered_unconstrained(
    xlim: np.ndarray,
    ylim: np.ndarray,
    zlim: np.ndarray,
    ylim_prop=False,
    rotate_rand=False,
    rotate_lim=[0, 0, 0],
    qpos=[1, 0, 0, 0],
    size_dict=None,
    obj_radius=0.1,
    z_offset=0.001,
    z_max=0,
    prohibited_area=None,
    obj_margin=0.005,
) -> sapien.Pose:
    if len(xlim) < 2 or xlim[1] < xlim[0]:
        xlim = np.array([xlim[0], xlim[0]])
    else:
        xlim = np.array([xlim[0] + obj_radius, xlim[1] - obj_radius])
        if xlim[0] > xlim[1]:
            # raise ValueError(
            #     f"obj_radius={obj_radius} too large for xlim: bounds leave no valid center range after inset."
            # )
            return False, None
    if len(ylim) < 2 or ylim[1] < ylim[0]:
        ylim = np.array([ylim[0], ylim[0]])
    else:
        ylim = np.array([ylim[0] + obj_radius, ylim[1] - obj_radius])
        if ylim[0] > ylim[1]:
            # raise ValueError(
            #     f"obj_radius={obj_radius} too large for ylim: bounds leave no valid center range after inset."
            # )
            return False, None
    if len(zlim) < 2 or zlim[1] < zlim[0]:
        zlim = np.array([zlim[0], zlim[0]])

    times = 0
    while True:
        times += 1
        if times > 100:
            return False, None
        x = np.random.uniform(xlim[0], xlim[1])
        y = np.random.uniform(ylim[0], ylim[1])
        new_obj_radius = obj_radius + obj_margin
        is_overlap = False
        for area in prohibited_area:
            if check_overlap(new_obj_radius, x, y, area):
                is_overlap = True
                break
        if is_overlap:
            continue
        distances = np.sqrt((np.array([sub_list[0] for sub_list in size_dict]) - x)**2 +
                            (np.array([sub_list[1] for sub_list in size_dict]) - y)**2)
        max_distances = np.array([sub_list[3] + new_obj_radius + obj_margin for sub_list in size_dict])

        # if y - new_obj_radius < 0:
        #     if z_max > 0.05:
        #         continue
        # if (x - new_obj_radius < -0.6 or x + new_obj_radius > 0.6 or y - new_obj_radius < -0.34
        #         or y + new_obj_radius > 0.34):
        #     continue
        if np.all(distances > max_distances):
            break

    z = np.random.uniform(zlim[0], zlim[1])
    z = z - z_offset

    rotate = qpos
    if rotate_rand:
        angles = [0, 0, 0]
        for i in range(3):
            angles[i] = np.random.uniform(-rotate_lim[i], rotate_lim[i])
        rotate_quat = t3d.euler.euler2quat(angles[0], angles[1], angles[2])
        rotate = t3d.quaternions.qmult(rotate, rotate_quat)

    return True, sapien.Pose([x, y, z], rotate)


def rand_create_cluttered_actor(
    scene,
    modelname: str,
    modelid: str,
    modeltype: str,
    xlim: np.ndarray,
    ylim: np.ndarray,
    zlim: np.ndarray,
    ylim_prop=False,
    rotate_rand=False,
    rotate_lim=[0, 0, 0],
    qpos=None,
    scale=None,
    convex=True,
    is_static=False,
    size_dict=None,
    obj_radius=0.1,
    z_offset=0.001,
    z_max=0,
    fix_root_link=True,
    prohibited_area=None,
    constrained=True,
) -> tuple[bool, Actor | None]:
    """
    constrained: True for constrained placement (ie table), False for unconstrained placement
    """
    if qpos is None:
        if modeltype == "glb":
            qpos = [0.707107, 0.707107, 0, 0]
            rotate_lim = [rotate_lim[0], rotate_lim[2], rotate_lim[1]]
        else:
            qpos = [1, 0, 0, 0]
    if constrained:
        success, obj_pose = rand_pose_cluttered(
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            ylim_prop=ylim_prop,
            rotate_rand=rotate_rand,
            rotate_lim=rotate_lim,
            qpos=qpos,
            size_dict=size_dict,
            obj_radius=obj_radius,
            z_offset=z_offset,
            z_max=z_max,
            prohibited_area=prohibited_area,
        )
    else:
        success, obj_pose = rand_pose_cluttered_unconstrained(
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            ylim_prop=ylim_prop,
            rotate_rand=rotate_rand,
            rotate_lim=rotate_lim,
            qpos=qpos,
            size_dict=size_dict,
            obj_radius=obj_radius,
            z_offset=z_offset,
            z_max=z_max,
            prohibited_area=prohibited_area,
        )

    if not success:
        return False, None

    if modeltype == "urdf":
        obj = create_cluttered_urdf_obj(
            scene=scene,
            pose=obj_pose,
            modelname=f"objects/objaverse/{modelname}/{modelid}",
            scale=scale,
            fix_root_link=fix_root_link,
        )
        if obj is None:
            return False, None
        else:
            return True, obj
    elif modeltype == "sapien_urdf":
        # Sapien URDF models live under assets/objects/{modelname}/, potentially with
        # multiple numbered subdirectories. Use create_sapien_urdf_obj to load them.
        obj = create_sapien_urdf_obj(
            scene=scene,
            pose=obj_pose,
            modelname=modelname,
            scale=scale,
            modelid=modelid,
            fix_root_link=fix_root_link,
        )
        if obj is None:
            return False, None
        else:
            return True, obj
    else:
        obj = create_actor(
            scene=scene,
            pose=obj_pose,
            modelname=modelname,
            model_id=modelid,
            scale=scale,
            convex=convex,
            is_static=is_static,
        )
        if obj is None:
            return False, None
        else:
            return True, obj


def create_cluttered_urdf_obj(scene, pose: sapien.Pose, modelname: str, scale=1.0, fix_root_link=True) -> Actor:
    scene, pose = preprocess(scene, pose)
    modeldir = Path(os.environ["BENCH_ROOT"]) / "assets" / modelname

    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.scale = scale
    loader.fix_root_link = fix_root_link
    loader.load_multiple_collisions_from_file = False
    object: sapien.Articulation = loader.load_multiple(str(modeldir / "model.urdf"))[1][0]
    object.set_pose(pose)

    # combined_scale = urdf_mesh_scales(modeldir / "model.urdf") * scale
    if isinstance(object, sapien.physx.PhysxArticulation):
        return ArticulationActor(object, None, scale=[scale,scale,scale])
    else:
        return Actor(object, None, scale=[scale,scale,scale])

def urdf_mesh_scales(urdf_path: str | Path):
    urdf_path = Path(urdf_path)
    root = ET.parse(urdf_path).getroot()
    out = []
    for tag in ("visual", "collision"):
        for node in root.findall(f".//{tag}//geometry//mesh"):
            fname = node.get("filename") or node.get("file")  # some URDFs use file
            scale = node.get("scale") or "1 1 1"
            scale_xyz = tuple(float(x) for x in scale.split())
            out.append((tag, fname, scale_xyz))
    out = set(out)
    return next(iter(out))[2]
