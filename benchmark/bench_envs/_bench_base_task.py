import math
import os
import re
import sapien.core as sapien
from sapien.render import clear_cache as sapien_clear_cache
from sapien.utils.viewer import Viewer
import numpy as np
import gymnasium as gym
import pdb
import toppra as ta
import json
import transforms3d as t3d
from collections import OrderedDict
import torch, random

from envs.utils import *
from bench_envs.utils import *
import math
from envs.robot import Robot
from envs.camera import Camera
from envs.utils.actor_utils import Actor, ArticulationActor
from envs._base_task import Base_Task

from copy import deepcopy
import subprocess
from pathlib import Path
import trimesh
import imageio
import glob


from envs._GLOBAL_CONFIGS import *

from typing import Optional, Literal

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)



# --------------------------------------------------------------------------- #
# Vector arithmetic that runs once per contact per PHYSICS SUBSTEP -- 1.25 M
# times in a 350-step episode, always on three or four numbers. numpy's cost
# there is allocation and dispatch rather than arithmetic: measured over one
# episode of `drop_apple_in_bin_ks`, `check_collisions` fell from 1.60 ms to
# 0.87 ms a call and the episode from 79.9 s to 67.7 s, with the verdict and
# every collision counter unchanged.
# --------------------------------------------------------------------------- #
def _pair_impulse_of(points):
    """Magnitude of the impulse a contact pair exchanged this substep (N*s)."""
    ix = iy = iz = 0.0
    for _p in points:
        _imp = _p.impulse
        ix += float(_imp[0])
        iy += float(_imp[1])
        iz += float(_imp[2])
    return math.sqrt(ix * ix + iy * iy + iz * iz)


def _dist3(a, b):
    dx, dy, dz = float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _quat_angle(q, r):
    """Angle between two unit quaternions."""
    d = abs(float(q[0]) * float(r[0]) + float(q[1]) * float(r[1])
            + float(q[2]) * float(r[2]) + float(q[3]) * float(r[3]))
    return 2 * math.acos(min(1.0, d))

class Bench_base_task(Base_Task):
    """
    Base task for all benchmark tasks. Mimics robotwin base task, with some functionality changes
    """
    FURNITURE_NAMES = {"table", "wall", "ground"}
    # Camera used for eval videos, first match wins. countertop_camera is the
    # elevated overhead view added per-scene by add_extra_cameras(); the rest are
    # fallbacks for scenes/embodiments that don't define it. All are D435, which
    # matches the video_size the eval client derives from head_camera_type.
    EVAL_VIDEO_CAMERAS = ("countertop_camera", "demo_camera", "head_camera")
    # Gripper links excluded from robot-to-furniture/static collision metrics (expected contact during manipulation)
    GRIPPER_LINK_NAMES = {"fr_link7", "fr_link8", "fl_link7", "fl_link8"}
    # Collision force threshold (N): ignore contacts with avg force below this.
    COLLISION_FORCE_THRESHOLD_N = 10.0
    # Static object pose thresholds: only count robot/target-to-static collisions when
    # the static object has moved beyond these from the previous step.
    STATIC_OBJECT_POSITION_THRESHOLD_M = 0.01   # 1 cm
    STATIC_OBJECT_ORIENTATION_THRESHOLD_RAD = 0.1  # ~5.7 deg
    # Motion-measurement window (substeps): the sampling period of the slow-
    # motion detector tier, and how long the per-frame "in motion" state
    # persists past the last observed movement (~0.12 s). The WATCH length and
    # re-baseline stillness use the longer STATIC_WATCH_WINDOW_STEPS below.
    STATIC_SETTLE_WINDOW_STEPS = 30
    # "Actively moving" gates for the per-frame collision flag: an object that
    # was knocked past the displacement threshold but has come to REST (e.g.
    # leaning against a destination box for the rest of the episode) must stop
    # flagging. Fast motion is caught instantly (per-substep delta), slow real
    # motion by a rolling window; sub-threshold creep (<~8 mm/s) and solver
    # jitter are ignored.
    STATIC_ACTIVE_MOVE_EPS_NOW_M = 1e-4     # >=0.1 mm per 4 ms substep (~25 mm/s)
    STATIC_ACTIVE_MOVE_EPS_NOW_RAD = 0.001  # ~0.06 deg per substep (~14 deg/s)
    STATIC_ACTIVE_MOVE_EPS_WIN_M = 0.001    # >=1 mm per settle window (~8 mm/s)
    STATIC_ACTIVE_MOVE_EPS_WIN_RAD = 0.01   # ~0.57 deg per window (~5 deg/s)
    # Contact frames require actual force exchange: PhysX also reports zero-
    # impulse "contacts" for shapes hovering inside the contact offset or in
    # stale/separating manifolds — those are not touches. Any real touch, even
    # a ~1 g object resting, transfers >= ~4e-5 N*s per substep, orders of
    # magnitude above this gate.
    CONTACT_MIN_IMPULSE_NS = 1e-6
    # Episode-counting watch length AND the stillness required before an
    # object's reference (baseline) pose updates — one clock, 90 substeps
    # (0.36 s). The watch arms on a forceful touch, extends on forceful touch
    # or motion, and dies after 90 substeps with neither (no revival). The
    # re-baseline fires only after 90 UNINTERRUPTED still substeps: motion
    # inside the window restarts the stillness count, so a paused-then-
    # resuming consequence keeps the original reference until it truly rests.
    STATIC_WATCH_WINDOW_STEPS = 90
    STATIC_BASELINE_RESET_STEPS = STATIC_WATCH_WINDOW_STEPS

    def __init__(self):
        pass

    # =========================================================== Init Task Env ===========================================================
    def _init_task_env_(self, table_xy_bias=[0, 0], table_height_bias=0, **kwags):
        pass

    # =========================================================== Frozen eval env packs ===========================================================
    # Opt-in: tasks set supports_eval_env = True and implement capture_task_spec /
    # load_actors_from_spec. Search writes JSON packs; eval rebuilds from them
    # without replaying integer seeds.
    supports_eval_env = False

    def capture_task_spec(self) -> dict:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement capture_task_spec "
            "(required when supports_eval_env is True)"
        )

    def load_actors_from_spec(self, spec: dict, init_state: dict):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement load_actors_from_spec "
            "(required when supports_eval_env is True)"
        )

    def after_eval_env_restore(self):
        """Hook after restore_init_state in pack mode (recompute derived poses)."""
        pass

    def _bind_eval_env(self, kwags) -> bool:
        """Parse eval_env pack from kwargs. Sets self.seed to episode_index in pack mode."""
        eval_env = kwags.get("eval_env")
        self._eval_env = eval_env
        construction = (eval_env or {}).get("construction") or {}
        self._eval_scene_spec = construction.get("scene") or {}
        self._eval_task_spec = construction.get("task") or {}
        self._eval_init_state = (eval_env or {}).get("init_state")
        self.task_config = kwags.get("task_config", getattr(self, "task_config", None))
        if eval_env:
            meta = eval_env.get("meta") or {}
            episode_index = int(meta.get("episode_index", kwags.get("now_ep_num", 0)))
            self.seed = episode_index
            self._eval_env_id = meta.get("env_id") or f"episode_{episode_index:04d}"
            return True
        self._eval_env_id = None
        self.seed = kwags.get("seed", 0)
        return False

    def _collision_cache_id(self):
        if getattr(self, "_eval_env_id", None):
            return self._eval_env_id
        return getattr(self, "seed", 0)

    def _apply_eval_env_textures(self) -> bool:
        if not getattr(self, "_eval_env", None):
            return False
        textures = (self._eval_scene_spec or {}).get("textures")
        if textures is None:
            return False
        self.wall_texture = textures.get("wall_texture")
        self.table_texture = textures.get("table_texture")
        self.floor_texture = textures.get("floor_texture")
        return True

    @staticmethod
    def _pose_to_list(pose):
        return [np.asarray(pose.p, dtype=float).tolist(),
                np.asarray(pose.q, dtype=float).tolist()]

    @staticmethod
    def pose_from_list(p_q) -> sapien.Pose:
        p, q = p_q
        return sapien.Pose(p=p, q=q)

    @staticmethod
    def pose_from_init_state(init_state: dict, name: str, occurrence: int = 0) -> sapien.Pose:
        matches = [a for a in (init_state or {}).get("actors", []) if a.get("name") == name]
        if occurrence >= len(matches):
            raise KeyError(
                f"init_state has no actor named '{name}' at occurrence {occurrence} "
                f"({len(matches)} match(es))"
            )
        return Bench_base_task.pose_from_list(matches[occurrence]["pose"])

    def capture_scene_spec(self) -> dict:
        cameras = []
        cam_list = getattr(getattr(self, "cameras", None), "static_camera_list", None) or []
        for cam in cam_list:
            try:
                name = cam.get_name() if hasattr(cam, "get_name") else cam.entity.get_name()
                cameras.append({"name": name, "pose": self._pose_to_list(cam.entity.get_pose())})
            except Exception as e:
                print(f"[capture_scene_spec] skip camera: {e}")
        clutter = []
        for rec in getattr(self, "record_cluttered_objects", None) or []:
            clutter.append(dict(rec))
        return {
            "scene_id": int(self.scene_id) if getattr(self, "scene_id", None) is not None else None,
            "arr_v": int(self.arr_v) if getattr(self, "arr_v", None) is not None else None,
            "table_z_bias": float(getattr(self, "table_z_bias", 0.0)),
            "crazy_random_light": bool(getattr(self, "crazy_random_light", False)),
            "textures": {
                "wall_texture": getattr(self, "wall_texture", None),
                "table_texture": getattr(self, "table_texture", None),
                "floor_texture": getattr(self, "floor_texture", None),
            },
            "cameras": cameras,
            "clutter": clutter,
        }

    def apply_camera_spec(self, cameras):
        if not cameras:
            return
        by_name = {c["name"]: c for c in cameras if c.get("name")}
        cam_list = getattr(getattr(self, "cameras", None), "static_camera_list", None) or []
        for cam in cam_list:
            try:
                name = cam.get_name() if hasattr(cam, "get_name") else cam.entity.get_name()
            except Exception:
                continue
            rec = by_name.get(name)
            if rec and rec.get("pose"):
                cam.entity.set_pose(self.pose_from_list(rec["pose"]))

    def _record_clutter_item(self, actor, obj_name, obj_idx, modeltype="glb"):
        pose = actor.get_pose()
        scale = getattr(actor, "scale", None)
        if isinstance(scale, np.ndarray):
            scale = scale.tolist()
        rec = {
            "object_type": obj_name,
            "object_index": obj_idx,
            "name": actor.get_name(),
            "scale": scale,
            "type": modeltype,
            "pose": self._pose_to_list(pose),
        }
        self.record_cluttered_objects.append(rec)
        return rec

    def spawn_clutter_from_spec(self, clutter_list):
        """Spawn recorded clutter at recorded poses. No RNG / rejection sampling."""
        self.record_cluttered_objects = []
        self.cluttered_objs = []
        if not clutter_list:
            return
        for rec in clutter_list:
            obj_name = rec["object_type"]
            obj_idx = rec["object_index"]
            scale = rec.get("scale")
            pose = self.pose_from_list(rec["pose"]) if rec.get("pose") else sapien.Pose()
            modeltype = rec.get("type", "glb")
            if modeltype == "urdf":
                actor = create_urdf_obj(
                    self,
                    pose=pose,
                    modelname=obj_name,
                    scale=scale[0] if isinstance(scale, (list, tuple)) else scale,
                    fix_root_link=False,
                )
            else:
                actor = create_actor(
                    scene=self,
                    pose=pose,
                    modelname=obj_name,
                    convex=True,
                    model_id=obj_idx if not isinstance(obj_idx, str) else int(obj_idx) if str(obj_idx).isdigit() else obj_idx,
                    scale=scale,
                    is_static=False,
                )
            if actor is None:
                print(f"[spawn_clutter_from_spec] failed to spawn {obj_name} id={obj_idx}")
                continue
            actor.set_name(rec.get("name") or obj_name)
            self.stabilize_object(actor)
            self.cluttered_objs.append(actor)
            self._record_clutter_item(actor, obj_name, obj_idx, modeltype=modeltype)
            if modeltype == "urdf":
                path = f"{os.environ['BENCH_ROOT']}/assets/objects/objaverse/{obj_name}/{obj_idx}/coacd_collision.obj"
            else:
                path = f"{os.environ['BENCH_ROOT']}/assets/objects/{obj_name}/collision/base{obj_idx}.glb"
            self.collision_list.append({
                "actor": actor,
                "collision_path": path,
                "is_obstacle": True,
            })

    def build_eval_env_pack(self, episode_index: int, search_seed=None) -> dict:
        return {
            "construction": {
                "scene": self.capture_scene_spec(),
                "task": self.capture_task_spec(),
            },
            "init_state": self.capture_init_state(),
            "meta": {
                "schema_version": 1,
                "task_name": getattr(self, "task_name", None),
                "task_config": getattr(self, "task_config", None),
                "episode_index": int(episode_index),
                "env_id": f"episode_{int(episode_index):04d}",
                "search_seed": search_seed,
            },
        }

    def _finish_eval_env_restore(self):
        if self._eval_init_state:
            self.restore_init_state(self._eval_init_state)
        self.apply_camera_spec((self._eval_scene_spec or {}).get("cameras") or [])
        self.after_eval_env_restore()

    def setup_scene(self, **kwargs):
        """
        Set the scene
            - Set up the basic scene: light source, viewer.
        """
        self.engine = sapien.Engine()
        # declare sapien renderer
        from sapien.render import set_global_config

        set_global_config(max_num_materials=50000, max_num_textures=50000)
        self.renderer = sapien.SapienRenderer()
        # give renderer to sapien sim
        self.engine.set_renderer(self.renderer)

        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("optix")

        # declare sapien scene
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)
        # set simulation timestep
        self.timestep = kwargs.get("timestep", 1 / 250)
        self.scene.set_timestep(self.timestep)
        # Impulse threshold = force_threshold * timestep (impulse in N·s)
        self.collision_impulse_threshold = max(
            self.COLLISION_FORCE_THRESHOLD_N * self.timestep,
            1e-6,  # floor to avoid numerical noise
        )
        # add ground to scene
        self.scene.add_ground(kwargs.get("ground_height", 0))
        # set default physical material
        self.scene.default_physical_material = self.scene.create_physical_material(
            kwargs.get("static_friction", 2),
            kwargs.get("dynamic_friction", 1),
            kwargs.get("restitution", 0),
        )
        # give some white ambient light of moderate intensity
        self.scene.set_ambient_light(kwargs.get("ambient_light", [0.5, 0.5, 0.5]))
        # default enable shadow unless specified otherwise
        shadow = kwargs.get("shadow", True)
        direction_lights = kwargs.get("direction_lights", [[[0, 0.5, -1], [0.5, 0.5, 0.5]]])
        point_lights = kwargs.get("point_lights", [[[1, 0, 1.8], [1, 1, 1]], [[-1, 0, 1.8], [1, 1, 1]]])

        apply_lighting_ablation = getattr(self, "apply_lighting_ablation", False)
        if apply_lighting_ablation:
            apply_color = getattr(self, "_lighting_color_enabled", True)
            apply_shadow = getattr(self, "_lighting_shadow_enabled", True) and (np.random.rand() < 0.5)
            apply_direction = getattr(self, "_lighting_direction_enabled", True)
            color_range = getattr(self, "_lighting_color_range", [0.4, 1.8])

            self.direction_light_lst = []
            for direction_light in direction_lights:
                direction, color = list(direction_light[0]), list(direction_light[1])
                if apply_color:
                    color = [float(np.random.uniform(color_range[0], color_range[1])) for _ in range(3)]
                if apply_direction:
                    theta = np.random.uniform(np.deg2rad(8), np.deg2rad(82))
                    phi = np.random.uniform(0, 2 * np.pi)
                    direction = [
                        float(np.sin(theta) * np.cos(phi)),
                        float(np.sin(theta) * np.sin(phi)),
                        float(np.cos(theta)),
                    ]
                self.direction_light_lst.append(
                    self.scene.add_directional_light(direction, color, shadow=apply_shadow))

            self.point_light_lst = []
            for point_light in point_lights:
                pos, color = list(point_light[0]), list(point_light[1])
                if apply_color:
                    color = [float(np.random.uniform(color_range[0], color_range[1])) for _ in range(3)]
                self.point_light_lst.append(self.scene.add_point_light(pos, color, shadow=apply_shadow))
            print(f"[Lighting] color={apply_color} direction={apply_direction} shadow={apply_shadow}")
        else:
            self.direction_light_lst = []
            for direction_light in direction_lights:
                if self.random_light and not getattr(self, "_eval_env", None):
                    direction_light[1] = [
                        np.random.rand(),
                        np.random.rand(),
                        np.random.rand(),
                    ]
                self.direction_light_lst.append(
                    self.scene.add_directional_light(direction_light[0], direction_light[1], shadow=shadow))
            self.point_light_lst = []
            for point_light in point_lights:
                if self.random_light and not getattr(self, "_eval_env", None):
                    point_light[1] = [np.random.rand(), np.random.rand(), np.random.rand()]
                self.point_light_lst.append(self.scene.add_point_light(point_light[0], point_light[1], shadow=shadow))

        # initialize viewer with camera position and orientation
        if self.render_freq:
            self.viewer = Viewer(self.renderer)
            self.viewer.set_scene(self.scene)
            self.viewer.set_camera_xyz(
                x=kwargs.get("camera_xyz_x", 0.4),
                y=kwargs.get("camera_xyz_y", 0.22),
                z=kwargs.get("camera_xyz_z", 1.5),
            )
            self.viewer.set_camera_rpy(
                r=kwargs.get("camera_rpy_r", 0),
                p=kwargs.get("camera_rpy_p", -0.8),
                y=kwargs.get("camera_rpy_y", 2.45),
            )

    def create_static_elements(self, table_xy_bias=[0, 0], table_height=0.74):
        pass

    # ==================================================================
    # Perturbation parsing + application helpers.
    # Called by subclasses (Office/Study/Kitchen base tasks) from their
    # own _init_task_env_ so vision / ood / object / language
    # flags get wired uniformly.
    # ==================================================================
    def _parse_perturbations(self, random_setting):
        random_setting = random_setting or {}

        # Vision — lighting (color tint, direction, shadow toggle). Specular
        # moved to ood_perturbation.specular below.
        vision = random_setting.get("vision_perturbation", {}) or {}
        lighting = vision.get("lighting", {}) or {}
        self.apply_lighting_ablation = bool(lighting.get("enabled", False))
        self._lighting_color_enabled = bool(lighting.get("color", True))
        self._lighting_direction_enabled = bool(lighting.get("direction", True))
        self._lighting_shadow_enabled = bool(lighting.get("shadow", True))
        self._lighting_color_range = lighting.get("color_range", [0.4, 1.8])

        # Vision — gaussian blur (per-frame, applied in envs/_base_task.py:get_obs).
        # σ = ((severity-1)/4) * 10 * strength;  severity 2 + strength 1.0 → σ=2.5.
        blur = vision.get("blur", {}) or {}
        self.blur_perturb_enabled = bool(blur.get("enabled", False))
        if self.blur_perturb_enabled:
            severity = int(blur.get("severity", 2))
            severity = max(1, min(5, severity))
            self.blur_strength = float(blur.get("strength", 1.0))
            self.blur_sigma = ((severity - 1) / 4.0) * 10.0 * self.blur_strength
            print(f"[Vision] gaussian blur severity={severity} σ={self.blur_sigma:.2f}")

        # Vision — pixel shift (per-frame; randomized inside consumer loop).
        shift = vision.get("pixel_shift", {}) or {}
        self.pixel_shift_enabled = bool(shift.get("enabled", False))
        self.pixel_shift_max = float(shift.get("max_shift", 5))
        self.pixel_shift_strength = float(shift.get("strength", 1.0))

        # OOD perturbation — specular + target texture swap + surface
        # material. Grouped here because all change material/appearance away
        # from the training distribution, vs. lighting which only changes
        # photometry.
        ood = random_setting.get("ood_perturbation", {}) or {}
        spec = ood.get("specular", {}) or {}
        self._specular_enabled = bool(spec.get("enabled", False))
        surf = ood.get("surface_material", {}) or {}
        self._surface_material_enabled = bool(surf.get("enabled", False))
        self._surface_material_metallic_range = surf.get("metallic_range", [0.0, 0.8])
        self._surface_material_roughness_range = surf.get("roughness_range", [0.05, 0.95])
        furn = ood.get("furniture_texture", {}) or {}
        self._furniture_texture_enabled = bool(furn.get("enabled", False))

        # Object perturbation (non-visual: scene-level instance distributions).
        # Obstacles and targets are independently swappable to the OOD pool
        # (task_objects.yml:object_ood). Tasks whose target model has no OOD
        # entry fall back to the seen distribution via _target_ids().
        obj = random_setting.get("object_perturbation", {}) or {}
        self.unseen_obstacles = bool(obj.get("unseen_obstacles", False))
        self.unseen_targets = bool(obj.get("unseen_targets", False))
        self.obstacle_distribution = "object_ood" if self.unseen_obstacles else getattr(self, "sample_d", "objects")
        self.target_distribution = "object_ood" if self.unseen_targets else getattr(self, "sample_d", "objects")

        # Language
        lang = random_setting.get("language_perturbation", {}) or {}
        self.language_perturbation_enabled = bool(lang.get("enabled", False))
        self._instruction_bank_path = lang.get("instruction_bank")

    def _apply_specular_ood(self):
        """Apply specular/shininess variation to all actors in the scene.
        Must be called after load_actors. Gated on ood_perturbation.specular.
        """
        if not getattr(self, "_specular_enabled", False):
            return
        specular_strength = float(np.random.uniform(0.3, 6.0))
        shininess = float(np.random.uniform(10, 250))
        for actor in self.scene.get_all_actors():
            mats = actor.get_materials() if hasattr(actor, 'get_materials') else []
            for mat in mats:
                if mat is None:
                    continue
                try:
                    mat.set_specular(specular_strength)
                    mat.set_shininess(shininess)
                except Exception:
                    pass
        print(f"[OOD] specular={specular_strength:.2f} shine={shininess:.0f}")

    def _apply_furniture_texture_ood(self):
        """Swap base-color textures of large scene furniture (shelf/cabinet/
        fridge/bookcase/...) with random PNGs from assets/backgrounds/
        <type>/. Each scene declares the targets via self._furniture_texture_targets
        as a list of (attr_name, object_type) pairs.
        """
        if not getattr(self, "_furniture_texture_enabled", False):
            return
        targets = getattr(self, "_furniture_texture_targets", []) or []
        if not targets:
            return
        from bench_envs.utils.scene_gen_utils import change_object_texture
        applied = []
        for attr_name, obj_type in targets:
            actor = getattr(self, attr_name, None)
            if actor is None:
                continue
            try:
                change_object_texture(self, actor, None, obj_type, refresh_render=False)
                applied.append(attr_name)
            except Exception as e:
                print(f"[OOD] furniture_texture {attr_name}({obj_type}) failed: {e}")
        if applied:
            try:
                self.scene.update_render()
            except Exception:
                pass
            print(f"[OOD] furniture_texture applied to {applied}")

    def _target_ids(self, scene, model):
        """Return the list of target model_ids to sample from for (scene, model).
        Honors self.target_distribution; falls back to the seen pool if the
        chosen distribution has no entry for this model (so tasks whose target
        has no OOD ids still run instead of KeyError'ing). Auto-loads
        task_objects.yml on first use if self.item_info is not set."""
        item_info = getattr(self, "item_info", None)
        if not item_info:
            try:
                from bench_envs.utils.scene_gen_utils import get_task_objects_config
                item_info = get_task_objects_config()
            except Exception:
                import yaml
                with open(f"{os.environ['BENCH_ROOT']}/bench_task_config/task_objects.yml") as f:
                    item_info = yaml.safe_load(f) or {}
            self.item_info = item_info
        dist = getattr(self, "target_distribution", "objects")
        pool = (item_info.get(dist) or {}).get(scene, {}).get("targets", {}) or {}
        ids = pool.get(model)
        if ids:
            return ids
        return item_info["objects"][scene]["targets"][model]

    def _merge_ood_target_info(self, scene):
        """Merge object_ood target params into self.target_objects_info so OOD
        ids sampled by _target_ids() have radius/extra params available."""
        try:
            from envs.utils.rand_create_cluttered_actor import get_target_objects_subset
        except Exception:
            from envs.utils import get_target_objects_subset
        try:
            ood_info = get_target_objects_subset(scene, "object_ood") or {}
        except Exception as e:
            print(f"[OOD] _merge_ood_target_info({scene}) failed: {e}")
            return
        seen = self.target_objects_info or {}
        for m, info in ood_info.items():
            if m in seen:
                seen_ids = list(seen[m].get("ids", []))
                ood_ids = list(info.get("ids", []))
                seen[m]["ids"] = sorted(set(seen_ids) | set(ood_ids), key=lambda x: int(x) if str(x).isdigit() else x)
                seen_params = seen[m].setdefault("params", {})
                seen_params.update(info.get("params", {}) or {})
            else:
                seen[m] = info
        self.target_objects_info = seen

    def _maybe_apply_language_perturbation(self):
        """If enabled, pick a random instruction from instruction_bank.json for
        the current task_name and set it as the active instruction. Records the
        result to self.info so collect_data.py persists it into scene_info.json.
        """
        if not getattr(self, "language_perturbation_enabled", False):
            return None
        bank_path = getattr(self, "_instruction_bank_path", None)
        if not bank_path:
            return None
        if not os.path.isabs(bank_path):
            # Try BENCH_ROOT (where bench_task_config lives) then ROBOTWIN_ROOT.
            for root in (os.environ.get("BENCH_ROOT"), os.environ.get("ROBOTWIN_ROOT"), "."):
                if not root:
                    continue
                # Config path may already start with 'benchmark/…' when BENCH_ROOT
                # is the 'benchmark' dir, so try both joined and stripped forms.
                candidates = [os.path.join(root, bank_path)]
                if bank_path.startswith("benchmark/"):
                    candidates.append(os.path.join(root, bank_path[len("benchmark/"):]))
                for c in candidates:
                    if os.path.exists(c):
                        bank_path = c
                        break
                else:
                    continue
                break
        if not os.path.exists(bank_path):
            print(f"[Language] bank not found at {bank_path}")
            return None
        try:
            with open(bank_path, "r", encoding="utf-8") as f:
                bank = json.load(f)
        except Exception as e:
            print(f"[Language] failed to load bank: {e}")
            return None
        pool = bank.get(self.task_name, [])
        if not pool:
            return None
        instruction = str(np.random.choice(pool))
        self.set_instruction(instruction=instruction)
        if not isinstance(getattr(self, "info", None), dict):
            self.info = {}
        self.info["language_perturbation"] = {
            "instruction": instruction,
            "bank": bank_path,
        }
        print(f"[Language] {self.task_name} -> '{instruction[:60]}'")
        return instruction

    def get_cluttered_surfaces(self):
        pass
    
    def clutter_surface_split(self, xlim, ylim, zlim, prohibited_area, obstacle_count, cluttered_item_info, obj_names_short, obj_names_tall):
        """
        Produce clutter on a given surface from 2 object name pools
        """
        # # for viewing area estimation
        # for area in prohibited_area:
        #     x_min = area[0]
        #     x_max = area[2]
        #     y_min = area[1]
        #     y_max = area[3]
        #     half_size = [(x_max-x_min)/2, (y_max-y_min)/2, 0.0005]
        #     target = create_box(
        #         scene=self,
        #         pose=sapien.Pose([x_min+half_size[0], y_min+half_size[1], 0.74], [1,0,0,0]),
        #         half_size=half_size,
        #         color=(1, 0, 0),
        #         name=f"_collision",
        #         is_static=True,
        #     )
        # record cluttered objects
        self.record_cluttered_objects = []
        self.size_dict = []

        if np.random.rand() < self.clean_background_rate:
            return

        success_count = 0
        # allow_duplicate_clutter: place the same (object, model id) repeatedly and
        # give a much larger retry budget so `obstacle_count` is actually reached
        # (needed when the pool has few distinct ids, e.g. handcrafted clutter).
        allow_dup = getattr(self, "allow_duplicate_clutter", False)
        max_try = (obstacle_count * 20) if allow_dup else 50
        trys = 0

        # Track which specific model ids have been placed per object name
        placed_objects = {name: [] for name in cluttered_item_info.keys()}

        # Precompute desired counts by group (may not be reached if placement fails)
        short_target = int(round(0.3 * obstacle_count))
        tall_target = obstacle_count - short_target

        short_count = 0
        tall_count = 0

        # Build flat lists for sampling indices within each group
        obj_names_short = list(obj_names_short)
        obj_names_tall = list(obj_names_tall)

        # If one group is empty, fall back to the other
        if not obj_names_short and not obj_names_tall:
            return

        while success_count < obstacle_count and trys < max_try:
            # Decide which group to sample from for this attempt
            if not obj_names_short:
                group = "tall"
            elif not obj_names_tall:
                group = "short"
            else:
                # Prefer to fill up to targets with 30% short / 70% tall
                if short_count < short_target and tall_count < tall_target:
                    # sample with 0.3 / 0.7 probability
                    group = "short" if np.random.rand() < 0.3 else "tall"
                elif short_count < short_target:
                    group = "short"
                elif tall_count < tall_target:
                    group = "tall"
                else:
                    # both groups reached their nominal target; continue with 0.3/0.7 split
                    group = "short" if np.random.rand() < 0.3 else "tall"

            if group == "short":
                obj_list = obj_names_short
            else:
                obj_list = obj_names_tall

            if not obj_list:
                break

            obj = np.random.randint(len(obj_list))
            obj_name = obj_list[obj]

            # Randomly choose an index within available ids for this object
            ids_for_obj = cluttered_item_info[obj_name]["ids"]

            rand_idx = np.random.randint(len(ids_for_obj))
            obj_idx = ids_for_obj[rand_idx]

            if (not allow_dup) and obj_idx in placed_objects.get(obj_name, []):
                trys += 1
                continue

            obj_radius = cluttered_item_info[obj_name]["params"][obj_idx]["radius"]
            obj_offset = cluttered_item_info[obj_name]["params"][obj_idx]["z_offset"]
            obj_maxz = cluttered_item_info[obj_name]["params"][obj_idx]["z_max"]
            scale = cluttered_item_info[obj_name]["params"][obj_idx]["scale"]

            success, self.cluttered_obj = rand_create_cluttered_actor(
                self.scene,
                xlim=xlim,
                ylim=ylim,
                zlim=zlim,
                modelname=obj_name,
                modelid=obj_idx,
                scale=scale,
                modeltype=cluttered_item_info[obj_name]["type"],
                rotate_rand=True,
                rotate_lim=[0, 0, math.pi],
                size_dict=self.size_dict,
                obj_radius=obj_radius,
                z_offset=obj_offset,
                z_max=obj_maxz,
                prohibited_area=prohibited_area,
                is_static=False,
                constrained=False,
            )
            if not success or self.cluttered_obj is None:
                trys += 1
                continue

            self.cluttered_obj.set_name(obj_name)

            # manage stability as distractors
            self.stabilize_object(self.cluttered_obj)

            self.cluttered_objs.append(self.cluttered_obj)
            pose = self.cluttered_obj.get_pose().p.tolist()
            pose.append(obj_radius)
            self.size_dict.append(pose)
            success_count += 1

            if group == "short":
                short_count += 1
            else:
                tall_count += 1

            self._record_clutter_item(
                self.cluttered_obj, obj_name, obj_idx,
                modeltype=cluttered_item_info[obj_name]["type"],
            )
            placed_objects[obj_name].append(obj_idx)

            # add to collision list
            if cluttered_item_info[obj_name]["type"] == "urdf":
                path = f"{os.environ['BENCH_ROOT']}/assets/objects/objaverse/{obj_name}/{obj_idx}/coacd_collision.obj"
            else:
                path = f"{os.environ['BENCH_ROOT']}/assets/objects/{obj_name}/collision/base{obj_idx}.glb"
            self.collision_list.append({
                "actor": self.cluttered_obj,
                "collision_path": path,
                "is_obstacle": True,
            })

            # # for viewing radius estimation
            # half_size = [obj_radius, obj_radius, 0.0005]
            # pose = self.cluttered_obj.get_pose()
            # pose.q = [1,0,0,0]
            # target = create_box(
            #     scene=self,
            #     pose=pose,
            #     half_size=half_size,
            #     color=(1, 0, 0),
            #     name=f"{obj_name}_collision",
            #     is_static=True,
            # )
        
        if success_count < obstacle_count:
            print(f"Warning: Only {success_count} cluttered objects are placed on the surface.")

        self.size_dict = None
        self.cluttered_objs = []

    def clutter_surface(self, xlim, ylim, zlim, prohibited_area, obstacle_count, cluttered_item_info, obj_names):
        """
        Produce clutter on a given surface
        """
        # # for viewing area estimation
        # for area in prohibited_area:
        #     x_min = area[0]
        #     x_max = area[2]
        #     y_min = area[1]
        #     y_max = area[3]
        #     half_size = [(x_max-x_min)/2, (y_max-y_min)/2, 0.0005]
        #     target = create_box(
        #         scene=self,
        #         pose=sapien.Pose([x_min+half_size[0], y_min+half_size[1], zlim[0]], [1,0,0,0]),
        #         half_size=half_size,
        #         color=(1, 0, 0),
        #         name=f"_collision",
        #         is_static=True,
        #     )

        # record cluttered objects
        self.record_cluttered_objects = []
        self.size_dict = []

        if np.random.rand() < self.clean_background_rate:
            return

        success_count = 0
        max_try = 50
        trys = 0

        # Track which specific model ids have been placed per object name
        placed_objects = {name: [] for name in cluttered_item_info.keys()}

        obj_names = list(obj_names)
        if not obj_names:
            return

        while success_count < obstacle_count and trys < max_try:
            obj = np.random.randint(len(obj_names))
            obj_name = obj_names[obj]

            # Randomly choose an index within available ids for this object
            ids_for_obj = cluttered_item_info[obj_name]["ids"]
            rand_idx = np.random.randint(len(ids_for_obj))
            obj_idx = ids_for_obj[rand_idx]

            if obj_idx in placed_objects.get(obj_name, []):
                trys += 1
                continue

            obj_radius = cluttered_item_info[obj_name]["params"][obj_idx]["radius"]
            obj_offset = cluttered_item_info[obj_name]["params"][obj_idx]["z_offset"]
            obj_maxz = cluttered_item_info[obj_name]["params"][obj_idx]["z_max"]
            scale = cluttered_item_info[obj_name]["params"][obj_idx]["scale"]

            success, self.cluttered_obj = rand_create_cluttered_actor(
                self.scene,
                xlim=xlim,
                ylim=ylim,
                zlim=zlim,
                modelname=obj_name,
                modelid=obj_idx,
                scale=scale,
                modeltype=cluttered_item_info[obj_name]["type"],
                rotate_rand=True,
                rotate_lim=[0, 0, math.pi],
                size_dict=self.size_dict,
                obj_radius=obj_radius,
                z_offset=obj_offset,
                z_max=obj_maxz,
                prohibited_area=prohibited_area,
                is_static=False,
                constrained=False,
            )
            if not success or self.cluttered_obj is None:
                trys += 1
                continue

            self.cluttered_obj.set_name(obj_name)

            # manage stability as distractors
            self.stabilize_object(self.cluttered_obj)

            self.cluttered_objs.append(self.cluttered_obj)
            pose = self.cluttered_obj.get_pose().p.tolist()
            pose.append(obj_radius)
            self.size_dict.append(pose)
            success_count += 1

            self._record_clutter_item(
                self.cluttered_obj, obj_name, obj_idx,
                modeltype=cluttered_item_info[obj_name]["type"],
            )
            placed_objects[obj_name].append(obj_idx)

            # add to collision list
            if cluttered_item_info[obj_name]["type"] == "urdf":
                path = f"{os.environ['BENCH_ROOT']}/assets/objects/objaverse/{obj_name}/{obj_idx}/coacd_collision.obj"
            else:
                path = f"{os.environ['BENCH_ROOT']}/assets/objects/{obj_name}/collision/base{obj_idx}.glb"
            self.collision_list.append({
                "actor": self.cluttered_obj,
                "collision_path": path,
                "is_obstacle": True,
            })

        if success_count < obstacle_count:
            print(f"Warning: Only {success_count} cluttered objects are placed on the surface.")

        self.size_dict = None
        self.cluttered_objs = []
    
    def stabilize_object(self, object):
        # object.set_mass(1)
        rb = object.actor.components[1]
        try:
            rb.set_linear_damping(5.0)
            rb.set_angular_damping(20.0)
        except:
            pass

    # =========================================================== Collision Metrics ===========================================================

    def _init_collision_metrics(self):
        """Reset collision tracking state. Call early in _init_task_env_ before load_actors()."""
        self.target_object_names: set[str] = set()
        # Destination object (the task's `des_obj` role, e.g. board/plate/can);
        # resolved lazily on the first check_collisions() call once des_obj exists.
        # Empty for pick / pose-destination tasks. Excluded from the contact flag
        # (placing onto the destination is intended, like grasping the target).
        self.destination_object_names: set[str] = set()
        # Actors the task deliberately touches (populated by grasp_actor via
        # _mark_intended_contact: grasp targets, drawer/appliance handles + their
        # articulation links). Excluded from the contact flag.
        self._intended_contact_names: set[str] = set()
        self.collision_metrics = {
            "robot_to_furniture": 0,
            "robot_to_static_object": 0,
            "target_to_static_object": 0,
            "intended_to_static_object": 0,  # RETIRED 2026-07-14 (kept for schema compat, always 0): actuated-body shoves are contact-only
        }
        # Displacement-driven static-collision counting state (ported from the
        # collison_free_data_gen branch): a collision IS a static object moving
        # past threshold from its episode-start pose; contacts only attribute
        # which body (robot / held target) last touched it.
        self._static_last_toucher: dict[int, str] = {}   # per_scene_id -> "robot"|"target"
        self._static_last_touch_step: dict[int, int] = {}  # per_scene_id -> _metric_step of last touch
        self._counted_displaced_ids: set[int] = set()    # counted once per displacement EVENT (re-armed when the object settles + re-baselines)
        self._displaced_categories: dict[int, str] = {}  # per_scene_id -> counted category
        # "Actively moving" detector state for the per-frame collision flag:
        # rolling reference pose (~settle-window old) per object, and the last
        # substep on which the object was observed moving.
        self._static_active_ref: dict[int, tuple] = {}     # per_scene_id -> (step, pos, quat)
        self._static_last_active_step: dict[int, int] = {} # per_scene_id -> _metric_step
        # Episode-counting watch: last substep the object was touched OR seen
        # moving while the watch was live. Started by a touch; extended by
        # motion; expired (> settle window with neither) ends the watch for good
        # until the next touch.
        self._static_watch_last_seen: dict[int, int] = {}  # per_scene_id -> _metric_step
        # Last touch pair label per static object ("name#id|name#id") — used to
        # attribute the delayed-crossing collision frame when the threshold is
        # passed after contact already ended (no live pair at that substep).
        self._static_last_touch_pair: dict[int, str] = {}
        self._counted_furniture_names: set[str] = set()
        self._hit_furniture_names: set[str] = set()
        self.filtered_contacts_for_log = []
        self.static_object_pose_prev: dict[int, tuple] = {}   # per_scene_id -> (pos, quat)
        self.static_object_pose_start: dict[int, tuple] = {}  # per_scene_id -> (pos, quat), captured once per episode
        # Optional per-step JSONL streams (opened via start_metric_streams): a
        # contacts stream (every robot/target<->static contact point, zero-impulse
        # included) and a collisions stream (displacement events). None = disabled;
        # the primary per-timestep record is the hdf5 contact/collision arrays below.
        self._contact_stream = None
        self._collision_stream = None
        self._metric_step = -1
        # Per-timestep window flags: OR-accumulated inside check_collisions() every
        # physics substep, flushed + reset by _take_picture() into each saved frame
        # (-> per-frame "contact"/"collision" uint8 arrays in the episode hdf5).
        self._win_contact = False
        self._win_collision = False
        # per-window max contact impulse (N·s) for the flagged contact / collision
        self._win_contact_impulse = 0.0
        self._win_collision_impulse = 0.0
        # per-window sets of "a|b" object-pair labels (flushed by _take_picture),
        # so each saved frame records WHICH bodies touched / collided that window.
        self._win_contact_pairs = set()
        self._win_collision_pairs = set()
        self._win_contact_pairs_seen = set()  # TSL_DEBUG_CONTACTS one-shot log

    def _get_target_object_names(self) -> set[str]:
        """Return the names of target objects for this task.
        Must be overridden by every concrete task subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override _get_target_object_names()"
        )

    def _build_collision_name_sets(self):
        """
        Build name sets for collision detection. Must be called after all actors
        (robot, furniture, target objects, static objects, clutter) are loaded.
        """
        self.robot_link_names = set(
            link.get_name() for link in
            self.robot.left_entity.get_links() + self.robot.right_entity.get_links()
        )
        self.furniture_names = set(self.FURNITURE_NAMES)
        self.target_object_names = self._get_target_object_names()

        # Key static objects by per_scene_id — unique per actor, stable across Python access.
        # Cache the entity handle too so the displacement sweep can look up only the
        # touched objects instead of re-scanning scene.get_all_actors() every step.
        self.static_object_ids: set[int] = set()
        self._static_id_to_name: dict[int, str] = {}
        self._static_id_to_entity: dict = {}
        for entity in self.scene.get_all_actors():
            n = entity.get_name()
            if not n or n in self.furniture_names or n in self.target_object_names:
                continue
            self.static_object_ids.add(entity.per_scene_id)
            self._static_id_to_name[entity.per_scene_id] = n
            self._static_id_to_entity[entity.per_scene_id] = entity

    def _static_object_has_significant_pose_change(self, actor_id: int, entity) -> bool:
        """Return True if cumulative displacement from episode start exceeds thresholds."""
        start = self.static_object_pose_start.get(actor_id)
        if start is None:
            # No baseline -> no displacement evidence -> NOT a collision. (The old
            # "return True" fallback produced phantom per-frame collision flags on
            # objects that never moved, with zero impulse, whenever a contact was
            # checked before the object's first pose snapshot.)
            return False
        start_p, start_q = start
        curr = entity.get_pose()
        curr_p = np.array(curr.p, dtype=np.float64)
        curr_q = np.array(curr.q, dtype=np.float64)
        cumul_pos = float(np.linalg.norm(curr_p - start_p))
        qdot = abs(float(np.dot(curr_q, start_q)))
        cumul_ang = 2 * np.arccos(min(1.0, qdot))
        return (
            cumul_pos >= self.STATIC_OBJECT_POSITION_THRESHOLD_M
            or cumul_ang >= self.STATIC_OBJECT_ORIENTATION_THRESHOLD_RAD
        )

    def _print_significant_collision(self, actor_id: int, entity, category: str) -> None:
        """Print displacement stats the first time a (object, category) pair is counted."""
        display_name = f"{self._static_id_to_name.get(actor_id, entity.get_name())}#{actor_id}"
        curr_p = np.array(entity.get_pose().p, dtype=np.float64)
        curr_q = np.array(entity.get_pose().q, dtype=np.float64)
        start = self.static_object_pose_start.get(actor_id)
        print(f"[Collision] [SIGNIFICANT] {category}: '{display_name}'")
        if start is not None:
            start_p, start_q = start
            cumul_pos = float(np.linalg.norm(curr_p - start_p))
            qdot = abs(float(np.dot(curr_q, start_q)))
            cumul_ang = 2 * np.arccos(min(1.0, qdot))
            print(f"  start_pos={np.round(start_p, 4)}  curr_pos={np.round(curr_p, 4)}")
            print(f"  cumul_delta={cumul_pos:.4f} m  cumul_ang={np.degrees(cumul_ang):.2f} deg")
           
        else:
            print(f"  curr_pos={np.round(curr_p, 4)}  (no start snapshot — collision before first step)")

    def _snapshot_static_object_poses(self):
        """Snapshot poses of all static objects before scene.step() for cumulative displacement tracking."""
        if not hasattr(self, 'static_object_ids'):
            return
        for actor_id, entity in self._static_id_to_entity.items():
            p = entity.get_pose()
            snapshot = (np.array(p.p, dtype=np.float64), np.array(p.q, dtype=np.float64))
            self.static_object_pose_prev[actor_id] = snapshot
            # Baseline per OBJECT at its first-seen snapshot (not only on the first
            # snapshot CALL of the episode): the old first_call gate left any id
            # missed by that single call permanently baseline-less, silently
            # disabling its collision detection (observed: a bottle tipped 67 deg
            # by a carried book registered no collision). First-seen == episode-
            # start pose, since snapshots precede every physics step.
            if actor_id not in self.static_object_pose_start:
                self.static_object_pose_start[actor_id] = snapshot

    def check_collisions(self):
        """
        Query PhysX contacts after scene.step() and accumulate collision counts.
        Categories:
          - robot_to_furniture:      robot link <-> furniture (table, wall, shelf, ground)
          - robot_to_static_object:  robot link <-> movable static objects (screen, clutter, etc.)
          - target_to_static_object: target object <-> movable static objects (held obj bumping things)
        Furniture: only count contacts with impulse above threshold.
        Static objects: only count when static object has significant pose change from previous step (e.g. knocked over, fallen).
        Populates self.filtered_contacts_for_log with contact points that passed filters (for debug/logging).
        """
        contacts = self.scene.get_contacts()
        self.filtered_contacts_for_log = []
        self._metric_step = getattr(self, "_metric_step", -1) + 1
        _stream_pairs = []   # [toucher, "obj#id", impulse, [x,y,z]] per contact point

        # Lazily resolve destination object names once they exist (des_obj* attrs
        # are set during setup, before motion). Multi-destination tasks (e.g.
        # move_items_around's des_obj_1/2/3) are all excluded. Cached once found.
        if not self.destination_object_names:
            _des_names = set()
            for _attr in list(vars(self)):
                if _attr.startswith("des_obj"):
                    _d = getattr(self, _attr, None)
                    if _d is None:
                        continue
                    try:
                        _des_names.add(_d.get_name())
                    except Exception:
                        pass
            if _des_names:
                self.destination_object_names = _des_names

        # --- "actively moving" detector (feeds the per-frame collision flag) --
        # A displaced object only produces collision FRAMES while its pose is
        # still changing; once it rests (even in a displaced pose, even still in
        # contact) the frames go back to contact-only. Two tiers: per-substep
        # delta vs the pre-step snapshot catches fast motion with zero latency;
        # a rolling ~settle-window reference catches slow real motion that is
        # invisible substep-to-substep. Costs one get_pose per static per substep
        # (same as _snapshot_static_object_poses already pays).
        for _sid, _ent in getattr(self, "_static_id_to_entity", {}).items():
            try:
                _pose = _ent.get_pose()
            except Exception:
                continue  # despawned actor -> stale cached handle
            _p = np.asarray(_pose.p, dtype=np.float64)
            _q = np.asarray(_pose.q, dtype=np.float64)
            _prev = self.static_object_pose_prev.get(_sid)
            if _prev is not None:
                _dp = _dist3(_p, _prev[0])
                _da = _quat_angle(_q, _prev[1])
                if (_dp >= self.STATIC_ACTIVE_MOVE_EPS_NOW_M
                        or _da >= self.STATIC_ACTIVE_MOVE_EPS_NOW_RAD):
                    self._static_last_active_step[_sid] = self._metric_step
            _ref = self._static_active_ref.get(_sid)
            if _ref is None:
                self._static_active_ref[_sid] = (self._metric_step, _p, _q)
            elif self._metric_step - _ref[0] >= self.STATIC_SETTLE_WINDOW_STEPS:
                _dp = _dist3(_p, _ref[1])
                _da = _quat_angle(_q, _ref[2])
                if (_dp >= self.STATIC_ACTIVE_MOVE_EPS_WIN_M
                        or _da >= self.STATIC_ACTIVE_MOVE_EPS_WIN_RAD):
                    self._static_last_active_step[_sid] = self._metric_step
                self._static_active_ref[_sid] = (self._metric_step, _p, _q)
            _last_act = self._static_last_active_step.get(_sid)
            # Extend the episode-counting watch while motion continues (a slow
            # topple keeps itself watched all the way down). Only a LIVE watch
            # extends — after a full watch window with neither forceful touch
            # nor motion it dies, and only a new forceful touch restarts it.
            if _last_act == self._metric_step:
                _seen = self._static_watch_last_seen.get(_sid)
                if (_seen is not None
                        and self._metric_step - _seen <= self.STATIC_WATCH_WINDOW_STEPS):
                    self._static_watch_last_seen[_sid] = self._metric_step
            # Settled re-baseline: one-shot (== not >=) the substep an object
            # completes STATIC_BASELINE_RESET_STEPS of stillness after having
            # moved; re-arms if it becomes active again. Objects that never
            # moved have no last-active entry and keep their episode-start
            # baseline. The settle also ENDS the displacement event: the object
            # becomes countable again, so knock -> settle -> knock again logs
            # two collisions (counting is per displacement EVENT, not per
            # object per episode).
            if (_last_act is not None
                    and self._metric_step - _last_act == self.STATIC_BASELINE_RESET_STEPS
                    and _sid in self.static_object_pose_start):
                self.static_object_pose_start[_sid] = (_p, _q)
                self._counted_displaced_ids.discard(_sid)

        for contact in contacts:
            entity0 = contact.bodies[0].entity
            entity1 = contact.bodies[1].entity
            name0 = entity0.name
            name1 = entity1.name

            # Total impulse transferred by this contact PAIR this substep: vector
            # sum over the manifold points (N*s). Point-wise max/any under-reports
            # spread contacts badly — a 12 N press split across 4 points at 3 N
            # each never tripped the 10 N furniture gate and read ~4x too small.
            _pair_impulse = _pair_impulse_of(contact.points)
            has_impulse = _pair_impulse > self.collision_impulse_threshold

            is_robot_0    = name0 in self.robot_link_names
            is_robot_1    = name1 in self.robot_link_names
            is_gripper_0  = name0 in self.GRIPPER_LINK_NAMES
            is_gripper_1  = name1 in self.GRIPPER_LINK_NAMES
            is_furniture_0 = name0 in self.furniture_names
            is_furniture_1 = name1 in self.furniture_names
            is_target_0   = name0 in self.target_object_names
            is_target_1   = name1 in self.target_object_names
            is_static_0   = entity0.per_scene_id in self.static_object_ids
            is_static_1   = entity1.per_scene_id in self.static_object_ids
            # Destination boxes: some tasks put their des_obj* in
            # _get_target_object_names (so the box itself is never treated as
            # displaceable clutter) — but as a TOUCHER a destination is passive
            # scenery: an object knocked against it and resting there is not
            # being touched BY the task. Excluded from the "target" toucher role.
            is_dest_0     = name0 in self.destination_object_names
            is_dest_1     = name1 in self.destination_object_names
            # robot-ACTUATED bodies (grasped objects / operated articulation links,
            # e.g. a drawer being closed) — contacts they cause are robot-caused
            is_intended_0 = name0 in self._intended_contact_names
            is_intended_1 = name1 in self._intended_contact_names

            count_furniture = False
            count_static = False
            count_target_static = False
            count_intended_static = False


            # Furniture: require impulse (actual force exchange); exclude gripper links (expected contact);
            # count each furniture piece at most once per episode.
            if ((is_robot_0 and is_furniture_1 and not is_gripper_0) or (is_robot_1 and is_furniture_0 and not is_gripper_1)):
                if has_impulse:
                    robot_link = name0 if is_robot_0 else name1
                    furniture_name = name1 if is_furniture_1 else name0
                    self._hit_furniture_names.add(furniture_name)
                    if furniture_name not in self._counted_furniture_names:
                        # print(f"[Collision] robot_to_furniture: {robot_link} -> {furniture_name}")
                        self.collision_metrics["robot_to_furniture"] += 1
                        self._counted_furniture_names.add(furniture_name)
                    count_furniture = True

            # Static objects: contacts only RECORD the most recent toucher —
            # counting is displacement-driven (sweep after the loop). Gripper
            # links COUNT here: targets are excluded from static_object_ids, so a
            # gripper<->static contact is a bump, never an expected grasp.
            # WATCH ARMING requires actual force exchange (same principle as the
            # contact flag): a zero-impulse margin "contact" cannot displace
            # anything, so it earns no causal credit for later motion. The
            # count_* booleans stay ungated — they mark the touch itself.
            _forceful = _pair_impulse > self.CONTACT_MIN_IMPULSE_NS
            if (is_robot_0 and is_static_1) or (is_robot_1 and is_static_0):
                static_entity = entity1 if is_static_1 else entity0
                if _forceful:
                    self._static_last_toucher[static_entity.per_scene_id] = "robot"
                    self._static_last_touch_step[static_entity.per_scene_id] = self._metric_step
                    self._static_watch_last_seen[static_entity.per_scene_id] = self._metric_step
                count_static = True

            if ((is_target_0 and not is_dest_0) and is_static_1) or \
                    ((is_target_1 and not is_dest_1) and is_static_0):
                static_entity = entity1 if is_static_1 else entity0
                if _forceful:
                    self._static_last_toucher[static_entity.per_scene_id] = "target"
                    self._static_last_touch_step[static_entity.per_scene_id] = self._metric_step
                    self._static_watch_last_seen[static_entity.per_scene_id] = self._metric_step
                count_target_static = True

            # Robot-actuated body (e.g. the drawer being closed) hitting a static
            # object: classified so the CONTACT flag records it (robot-caused
            # through the operated body) — but NOT a collision (2026-07-14):
            # no watch arming, no toucher attribution, no collision frames.
            # grasp_actor never runs in pure policy EVAL, so the intended set
            # is empty there; excluding it from collision on the collection
            # side keeps eval collision numbers exactly comparable.
            if ((is_intended_0 and is_static_1 and not is_intended_1)
                    or (is_intended_1 and is_static_0 and not is_intended_0)):
                count_intended_static = True

            # Optional dataset stream: every robot/target<->static contact POINT,
            # zero-impulse resting pairs included ("force" is a label, not a filter).
            if self._contact_stream is not None and (count_static or count_target_static or count_intended_static):
                static_entity = entity1 if is_static_1 else entity0
                toucher_name  = name0 if (is_robot_0 or is_target_0) else name1
                obj_key = f"{static_entity.get_name()}#{static_entity.per_scene_id}"
                for pt in contact.points:
                    _stream_pairs.append([
                        toucher_name, obj_key,
                        round(float(np.linalg.norm(pt.impulse)), 6),
                        [round(float(x), 4) for x in pt.position],
                    ])

            # --- Per-timestep window flags (un-deduped, hdf5 record) ----------
            # Independent of the episode counting above: give every SAVED frame a
            # contact/collision boolean + object pairs.  contact: any robot<->world
            # contact minus self/wheel/target/destination.  collision: impulse
            # furniture hit, or a static object being touched while past the
            # displacement threshold AND still actively moving — the shove is
            # flagged, resting displaced afterwards is not.  Flushed by
            # _take_picture().
            # Stable "a|b" pair label. Non-robot bodies carry their exact
            # instance as "name#per_scene_id" (two clutter twins of the same
            # model are different objects — a plain name is ambiguous for any
            # offline consumer). Robot link names are unique already.
            _id0 = getattr(entity0, "per_scene_id", None)
            _id1 = getattr(entity1, "per_scene_id", None)
            _lbl0 = name0 if (is_robot_0 or _id0 is None) else f"{name0}#{_id0}"
            _lbl1 = name1 if (is_robot_1 or _id1 is None) else f"{name1}#{_id1}"
            _pair = "|".join(sorted((_lbl0, _lbl1)))
            _cimp = _pair_impulse  # total pair impulse (N*s) this substep
            if is_robot_0 != is_robot_1:  # robot <-> world only (self-contacts excluded)
                _robot_side = name0 if is_robot_0 else name1
                _other_name = name1 if is_robot_0 else name0
                _other_is_target = is_target_1 if is_robot_0 else is_target_0
                _other_is_dest = _other_name in self.destination_object_names
                _other_is_intended = _other_name in self._intended_contact_names
                # exclude base wheels (ground support, zero-impulse) and the
                # TARGET / DESTINATION / INTENDED objects (grasping the target,
                # placing on the destination, operating a drawer/appliance handle
                # passed to grasp_actor — all deliberate manipulation, not contact)
                if ("wheel" not in _robot_side and not _other_is_target
                        and not _other_is_dest and not _other_is_intended
                        and _cimp > self.CONTACT_MIN_IMPULSE_NS):
                    self._win_contact = True
                    self._win_contact_pairs.add(_pair)
                    self._win_contact_impulse = max(self._win_contact_impulse, _cimp)
                    if os.environ.get("TSL_DEBUG_CONTACTS") and _pair not in self._win_contact_pairs_seen:
                        self._win_contact_pairs_seen.add(_pair)
                        print(f"[TSL contact pair] {_pair}")
            # held/TARGET object or robot-ACTUATED body (drawer being closed, ...)
            # bumping a static (clutter) object is a contact too: the robot causes
            # it through the grasped/operated body. Destination / intended objects
            # excluded on the static side as above.
            if count_target_static or count_intended_static:
                _stat_name = (entity1 if is_static_1 else entity0).get_name()
                if (_stat_name not in self.destination_object_names
                        and _stat_name not in self._intended_contact_names
                        and _cimp > self.CONTACT_MIN_IMPULSE_NS):
                    self._win_contact = True
                    self._win_contact_pairs.add(_pair)
                    self._win_contact_impulse = max(self._win_contact_impulse, _cimp)
            if count_furniture:  # impulse-gated furniture hit this step
                self._win_collision = True
                self._win_collision_pairs.add(_pair)
                self._win_collision_impulse = max(self._win_collision_impulse, _cimp)
            # Displacement collision frames require ALL THREE simultaneously:
            #   FORCEFUL contact (impulse > gate — a forceless graze on an
            #     object moving for other reasons, or sliding back along the
            #     arm after an already-counted knock, earns no frames)
            #   + displaced >=1 cm / 0.1 rad from the reference pose
            #   + IN MOTION (see detector above).
            # A knock whose threshold crossing happens AFTER the contact is
            # handled by the watch-gated sweep below: exactly one frame — the
            # crossing — gets flagged. Actuated-body (drawer) shoves are
            # contact-only, never collision.
            if count_static or count_target_static:  # robot / held-target touching a static
                _step_static = entity1 if is_static_1 else entity0
                _ssid = _step_static.per_scene_id
                if _forceful:
                    self._static_last_touch_pair[_ssid] = _pair
                if (_forceful
                        and self._static_object_has_significant_pose_change(_ssid, _step_static)
                        and self._metric_step - self._static_last_active_step.get(_ssid, -(10 ** 9))
                            <= self.STATIC_SETTLE_WINDOW_STEPS):
                    self._win_collision = True
                    self._win_collision_pairs.add(_pair)
                    self._win_collision_impulse = max(self._win_collision_impulse, _cimp)

            if count_furniture or count_static or count_target_static or count_intended_static:
                for pt in contact.points:
                    impulse = float(np.linalg.norm(pt.impulse))
                    # Log furniture contacts by impulse; log static contacts regardless
                    if count_furniture and impulse > self.collision_impulse_threshold:
                        self.filtered_contacts_for_log.append({
                            "body0": name0,
                            "body1": name1,
                            "impulse": impulse,
                            "position": [float(x) for x in pt.position],
                        })
                    elif (count_static or count_target_static or count_intended_static) and impulse > 0:
                        self.filtered_contacts_for_log.append({
                            "body0": name0,
                            "body1": name1,
                            "impulse": impulse,
                            "position": [float(x) for x in pt.position],
                        })

        if self._contact_stream is not None and _stream_pairs:
            _max_imp = max(p[2] for p in _stream_pairs)
            self._contact_stream.write(json.dumps({
                "t": self._metric_step,
                "ta": int(getattr(self, "take_action_cnt", -1)),
                "pairs": _stream_pairs,
                "max_impulse": _max_imp,
                "force": _max_imp > 1e-3,
            }) + "\n")

        # Displacement-driven counting: a static object whose cumulative pose
        # change from its baseline crosses the thresholds is counted as one
        # collision — whether or not anything touches it at that instant (slow
        # topples finish after contact ends). The WATCH starts at a FORCEFUL
        # robot / target / actuated-body touch and stays live while the object
        # is touched-with-force OR still moving (detector above extends it),
        # so a slow topple is followed all the way down; it expires after a
        # full watch window (90 substeps) with neither, and expired watches
        # are not revived (object→object chains and settling creep stay
        # unattributed). Checks only watched objects via cached handles.
        for sid, toucher in self._static_last_toucher.items():
            if (sid in self._counted_displaced_ids
                    or self.static_object_pose_start.get(sid) is None
                    or self._metric_step - self._static_watch_last_seen.get(sid, -(10 ** 9))
                       > self.STATIC_WATCH_WINDOW_STEPS):
                continue
            entity = self._static_id_to_entity.get(sid)
            if entity is None:
                continue
            if self._static_object_has_significant_pose_change(sid, entity):
                category = {"robot": "robot_to_static_object",
                            "target": "target_to_static_object"}[toucher]
                self.collision_metrics[category] = self.collision_metrics.get(category, 0) + 1
                self._counted_displaced_ids.add(sid)
                self._displaced_categories[sid] = category
                # Delayed crossing (e.g. tap -> contact ends -> object topples
                # past threshold a few substeps later): the simultaneous-touch
                # path above never fires, so flag THIS frame — the moment the
                # displacement happened — once, attributed to the last toucher.
                self._win_collision = True
                _lp = self._static_last_touch_pair.get(sid)
                if _lp:
                    self._win_collision_pairs.add(_lp)
                self._print_significant_collision(sid, entity, category)
                if self._collision_stream is not None:
                    start_p, start_q = self.static_object_pose_start[sid]
                    curr = entity.get_pose()
                    cum_p = float(np.linalg.norm(np.asarray(curr.p) - start_p))
                    qdot  = abs(float(np.dot(np.asarray(curr.q), start_q)))
                    cum_a = float(np.degrees(2 * np.arccos(min(1.0, qdot))))
                    self._collision_stream.write(json.dumps({
                        "t": self._metric_step,
                        "ta": int(getattr(self, "take_action_cnt", -1)),
                        "category": category,
                        "object": f"{self._static_id_to_name.get(sid, '?')}#{sid}",
                        "cumul_delta_m": round(cum_p, 4),
                        "cumul_ang_deg": round(cum_a, 2),
                        "last_toucher": toucher,
                    }) + "\n")

    def start_metric_streams(self, contacts_path, collisions_path):
        """Open per-episode jsonl streams: contacts (every robot/target<->static
        contact point per step) and collisions (displacement events)."""
        self.stop_metric_streams()
        self._contact_stream = open(contacts_path, "w", encoding="utf-8")
        self._collision_stream = open(collisions_path, "w", encoding="utf-8")
        # NOTE: no _metric_step reset here — _init_collision_metrics owns it.
        # Resetting mid-episode would desync the touch/active-step dicts
        # (negative deltas read as "recently active" -> spurious flags).

    def stop_metric_streams(self):
        for attr in ("_contact_stream", "_collision_stream"):
            fh = getattr(self, attr, None)
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass
                setattr(self, attr, None)

    def get_collision_metrics(self):
        """Return a copy of current collision metrics dict."""
        total = sum(self.collision_metrics.values())
        return {
            **self.collision_metrics,
            "is_collision": total > 0,
            "total_collision_count": total,
            "robot_to_furniture_names": sorted(self._hit_furniture_names),
            **{f"{cat}_names": sorted(
                   f"{getattr(self, '_static_id_to_name', {}).get(i, str(i))}#{i}"
                   for i, c in self._displaced_categories.items() if c == cat)
               for cat in ("robot_to_static_object", "target_to_static_object",
                           "intended_to_static_object")},
        }

    # =========================================================== Proximity Tracking ===========================================================

    def _init_proximity_tracking(self, config):
        """
        Build the scaled-trimesh cache used by export_scene to sample each
        obstacle's surface. Call after collision_list is fully populated.
        Skips ArticulationActor entries (export_scene boxes them via AABB).
        No per-step distance is computed at collection time — link/sphere
        proximity is a post-hoc relabel from scene.npz + per-frame qpos.
        """
        from envs.utils.actor_utils import ArticulationActor

        self._held_actors = {"left": None, "right": None}
        self._proximity_enabled = True

        self._proximity_mesh_cache: dict = {}  # actor_name -> scaled trimesh.Trimesh

        for entry in self.collision_list:
            actor = entry["actor"]
            if isinstance(actor, ArticulationActor):
                continue

            collision_path = entry["collision_path"]
            actor_name = actor.get_name()
            scale = actor.scale if actor.scale is not None else [1.0, 1.0, 1.0]

            try:
                if os.path.isdir(collision_path):
                    files_override = entry.get("files")
                    if files_override:
                        obj_files = [
                            Path(collision_path) / f
                            for f in files_override
                            if (Path(collision_path) / f).is_file()
                        ]
                    else:
                        obj_files = sorted(Path(collision_path).glob("*.obj"))
                    meshes = []
                    for p in obj_files:
                        try:
                            m = trimesh.load(str(p), force="mesh", process=False)
                            if isinstance(m, trimesh.Scene):
                                if m.geometry:
                                    m = trimesh.util.concatenate(list(m.geometry.values()))
                                else:
                                    continue
                            if len(getattr(m, "vertices", [])) > 0 and len(getattr(m, "faces", [])) > 0:
                                meshes.append(m)
                        except Exception:
                            continue
                    if not meshes:
                        continue
                    base_mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
                else:
                    base_mesh = trimesh.load(collision_path, force="mesh", process=False)
                    if isinstance(base_mesh, trimesh.Scene):
                        if base_mesh.geometry:
                            base_mesh = trimesh.util.concatenate(list(base_mesh.geometry.values()))
                        else:
                            continue
                    if len(getattr(base_mesh, "vertices", [])) == 0 or len(getattr(base_mesh, "faces", [])) == 0:
                        continue

                scaled_mesh = base_mesh.copy()
                scaled_mesh.apply_scale(scale)

                # Pre-warm BVH to avoid first-step latency spike
                trimesh.proximity.closest_point(scaled_mesh, np.zeros((1, 3)))

                self._proximity_mesh_cache[actor_name] = scaled_mesh
            except Exception as e:
                print(f"[Proximity] failed to load mesh for {actor_name}: {e}")

        print(f"[Proximity] scene mesh cache ready: "
              f"{sorted(self._proximity_mesh_cache.keys())}")

    def load_camera(self, **kwags):
        """
        Add cameras and set camera parameters
            - Including four cameras: left, right, front, head.
        """

        self.cameras = Camera(
            bias=self.table_z_bias,
            random_head_camera_dis=self.random_head_camera_dis,
            **kwags,
        )
        self.add_extra_cameras() # extra cameras specific to task env
        self.cameras.load_camera(self.scene)
        self.scene.step()  # run a physical step
        self.scene.update_render()  # sync pose from SAPIEN to renderer
    
    def add_extra_cameras(self):
        pass

    # =========================================================== Basic APIs ===========================================================

    def add_prohibit_area(
        self,
        actor: Actor | sapien.Entity | sapien.Pose | list | np.ndarray,
        padding=0.01,
        area="table"
    ):
        if (isinstance(actor, sapien.Pose) or isinstance(actor, list) or isinstance(actor, np.ndarray)):
            actor_pose = transforms._toPose(actor)
            actor_data = {}
            scale = 1
        else:
            actor_pose = actor.get_pose()
            if isinstance(actor, Actor):
                actor_data = actor.config
            else:
                actor_data = {}
            scale = actor.scale
            
        origin_bounding_size = (np.array(actor_data.get("extents", [0.1, 0.1, 0.1])) * scale / 2)
        origin_bounding_pts = (np.array([
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ]) * origin_bounding_size)

        if isinstance(padding,float) or isinstance(padding,int):
            padding = [padding, padding]

        actor_matrix = actor_pose.to_transformation_matrix()
        trans_bounding_pts = actor_matrix[:3, :3] @ origin_bounding_pts.T + actor_matrix[:3, 3].reshape(3, 1)
        x_min = np.min(trans_bounding_pts[0]) - padding[0]
        x_max = np.max(trans_bounding_pts[0]) + padding[0]
        y_min = np.min(trans_bounding_pts[1]) - padding[1]
        y_max = np.max(trans_bounding_pts[1]) + padding[1]
        # add_robot_visual_box(self, [x_min, y_min, actor_matrix[3, 3]])
        # add_robot_visual_box(self, [x_max, y_max, actor_matrix[3, 3]])
        self.prohibited_area[area].append([x_min, y_min, x_max, y_max])

    def choose_grasp_pose(
        self,
        actor: Actor,
        arm_tag: ArmTag,
        pre_dis=0.1,
        target_dis=0,
        contact_point_id: list | float = None,
    ) -> list:
        """
        Test the grasp pose function.
        - actor: The actor to be grasped.
        - arm_tag: The arm to be used for grasping, either "left" or "right".
        - pre_dis: The distance in front of the grasp point, default is 0.1.
        """
        if not self.plan_success:
            return
        res_pre_top_down_pose = None
        res_top_down_pose = None
        res_id_top_down = None
        dis_top_down = 1e9
        res_pre_side_pose = None
        res_side_pose = None
        res_id_side = None
        dis_side = 1e9
        res_pre_pose = None
        res_pose = None
        res_id = None
        dis = 1e9

        pref_direction = self.robot.get_grasp_perfect_direction(arm_tag)

        def get_grasp_pose(pre_grasp_pose, pre_grasp_dis):
            grasp_pose = deepcopy(pre_grasp_pose)
            grasp_pose = np.array(grasp_pose)
            direction_mat = t3d.quaternions.quat2mat(grasp_pose[-4:])
            grasp_pose[:3] += [pre_grasp_dis, 0, 0] @ np.linalg.inv(direction_mat)
            grasp_pose = grasp_pose.tolist()
            return grasp_pose

        def check_pose(pre_pose, pose, arm_tag):
            if arm_tag == "left":
                plan_func = self.robot.left_plan_path
            else:
                plan_func = self.robot.right_plan_path
            pre_path = plan_func(pre_pose)
            if pre_path["status"] != "Success":
                return False
            pre_qpos = pre_path["position"][-1]
            return plan_func(pose)["status"] == "Success"

        if contact_point_id is not None:
            if type(contact_point_id) != list:
                contact_point_id = [contact_point_id]
            contact_point_id = [(i, None) for i in contact_point_id]
        else:
            contact_point_id = actor.iter_contact_points()
        for i, _ in contact_point_id:
            pre_pose = self.get_grasp_pose(actor, arm_tag, contact_point_id=i, pre_dis=pre_dis)
            if pre_pose is None:
                continue
            pose = get_grasp_pose(pre_pose, pre_dis - target_dis)
            now_dis_top_down = cal_quat_dis(
                pose[-4:],
                GRASP_DIRECTION_DIC[("top_down_little_left" if arm_tag == "right" else "top_down_little_right")],
            )
            now_dis_side = cal_quat_dis(pose[-4:], GRASP_DIRECTION_DIC[pref_direction])

            if res_pre_top_down_pose is None or now_dis_top_down < dis_top_down:
                res_pre_top_down_pose = pre_pose
                res_top_down_pose = pose
                res_id_top_down = i
                dis_top_down = now_dis_top_down

            if res_pre_side_pose is None or now_dis_side < dis_side:
                res_pre_side_pose = pre_pose
                res_side_pose = pose
                res_id_side = i
                dis_side = now_dis_side

            now_dis = 0.7 * now_dis_top_down + 0.3 * now_dis_side
            if res_pre_pose is None or now_dis < dis:
                res_pre_pose = pre_pose
                res_pose = pose
                res_id = i
                dis = now_dis
                
        if dis_top_down < 0.15:
            # print(f"choose_grasp_pose: selected contact_point_id={res_id_top_down} (top_down)")
            return res_pre_top_down_pose, res_top_down_pose
        if dis_side < 0.15:
            # print(f"choose_grasp_pose: selected contact_point_id={res_id_side} (side)")
            return res_pre_side_pose, res_side_pose
        # print(f"choose_grasp_pose: selected contact_point_id={res_id} (combined)")
        return res_pre_pose, res_pose

    def _mark_intended_contact(self, actor):
        """Record an actor the task deliberately touches (grasp target, drawer
        handle, appliance door, ...) so the per-frame `contact` flag can exclude
        it — generic: fires for whatever the task passes to grasp_actor."""
        if not hasattr(self, "_intended_contact_names"):
            self._intended_contact_names = set()
        for cand in (actor, getattr(actor, "actor", None), getattr(actor, "entity", None)):
            if cand is None:
                continue
            try:
                self._intended_contact_names.add(cand.get_name())
            except Exception:
                pass
            if hasattr(cand, "get_links"):  # articulation: include all link names
                try:
                    for link in cand.get_links():
                        self._intended_contact_names.add(link.get_name())
                except Exception:
                    pass

    def grasp_actor(
        self,
        actor: Actor,
        arm_tag: ArmTag,
        pre_grasp_dis=0.1,
        grasp_dis=0,
        gripper_pos=0.0,
        contact_point_id: list | float = None,
    ):
        self._mark_intended_contact(actor)
        if not self.plan_success:
            return None, []
        if self.need_plan == False:
            if pre_grasp_dis == grasp_dis:
                return arm_tag, [
                    Action(arm_tag, "move", target_pose=[0, 0, 0, 0, 0, 0, 0]),
                    Action(arm_tag, "close", target_gripper_pos=gripper_pos),
                ]
            else:
                return arm_tag, [
                    Action(arm_tag, "move", target_pose=[0, 0, 0, 0, 0, 0, 0]),
                    Action(
                        arm_tag,
                        "move",
                        target_pose=[0, 0, 0, 0, 0, 0, 0],
                        constraint_pose=[1, 1, 1, 0, 0, 0],
                    ),
                    Action(arm_tag, "close", target_gripper_pos=gripper_pos),
                ]

        pre_grasp_pose, grasp_pose = self.choose_grasp_pose(
            actor,
            arm_tag=arm_tag,
            pre_dis=pre_grasp_dis,
            target_dis=grasp_dis,
            contact_point_id=contact_point_id,
        )

        if pre_grasp_pose is None:
            print("[ERROR] can't find a valid pre_grasp_pose")
            self.plan_success = False
            return None, []

        if pre_grasp_pose == grasp_pose:
            return arm_tag, [
                Action(arm_tag, "move", target_pose=pre_grasp_pose),
                Action(arm_tag, "close", target_gripper_pos=gripper_pos),
            ]
        else:
            return arm_tag, [
                Action(arm_tag, "move", target_pose=pre_grasp_pose),
                Action(
                    arm_tag,
                    "move",
                    target_pose=grasp_pose,
                    constraint_pose=[1, 1, 1, 0, 0, 0],
                ),
                Action(arm_tag, "close", target_gripper_pos=gripper_pos),
            ]

    def get_curobo_target(self):
        """
        Return (left_ee_target, right_ee_target) for CuRobo escape planning.

        Generic default: finds the primary target object via _get_target_object_names(),
        then computes a pre-grasp pose for each arm that still has an open gripper
        (i.e. hasn't grasped anything yet). This covers the most common collision
        scenario — arm hitting clutter while reaching toward the target object —
        without any per-task code.

        For the grasped/placement phase (both grippers closed) returns (None, None),
        which causes the branch to be skipped. Tasks can override for finer control.

        Each target is a pose accepted by robot.left_plan_path / robot.right_plan_path,
        or None to skip planning that arm.
        """
        try:
            target_names = self._get_target_object_names()
            # Use wrapped Actor objects from task instance vars (self.bottle, self.cup, etc.)
            # self.scene.get_all_actors() returns raw SAPIEN Entity objects which lack
            # iter_contact_points() needed by choose_grasp_pose().
            target_actors = [
                v for v in self.__dict__.values()
                if hasattr(v, 'iter_contact_points') and hasattr(v, 'get_name')
                and v.get_name() in target_names
            ]
            if not target_actors:
                return None, None

            target_actor = target_actors[0]
            left_target = right_target = None

            # Pick only the arm closest to the target object.
            # Planning for all open-gripper arms causes both arms to move on
            # single-arm tasks where both grippers happen to be open.
            target_pos  = np.array(target_actor.get_pose().p)
            left_ee_pos = np.array(self.robot.get_left_ee_pose()[:3])
            right_ee_pos= np.array(self.robot.get_right_ee_pose()[:3])
            dist_left   = np.linalg.norm(left_ee_pos  - target_pos)
            dist_right  = np.linalg.norm(right_ee_pos - target_pos)

            # Use get_grasp_pose (pure geometry, no CuRobo calls) rather than
            # choose_grasp_pose. choose_grasp_pose internally calls right_plan_path
            # for every contact point via check_pose — after N×2 CuRobo calls, the
            # GPU numerical state is exhausted and the explicit plan_path call in
            # _curobo_escape fails with IK_FAIL even for the same pose.
            # get_grasp_pose computes the pose geometrically; _curobo_escape makes
            # the single CuRobo planning call.
            contact_points = list(target_actor.iter_contact_points())
            if not contact_points:
                return None, None
            first_cp = contact_points[0][0]

            if dist_left <= dist_right and self.robot.is_left_gripper_open():
                left_target = self.get_grasp_pose(target_actor, arm_tag=ArmTag("left"),
                                                   contact_point_id=first_cp, pre_dis=0.07)
            elif self.robot.is_right_gripper_open():
                right_target = self.get_grasp_pose(target_actor, arm_tag=ArmTag("right"),
                                                    contact_point_id=first_cp, pre_dis=0.07)

            return left_target, right_target
        except Exception as e:
            print(f"[curobo] get_curobo_target failed: {e}")
            return None, None

    def get_place_pose(
        self,
        actor: Actor,
        arm_tag: ArmTag,
        target_pose: list | np.ndarray,
        constrain: Literal["free", "align", "auto"] = "auto",
        align_axis: list[np.ndarray] | np.ndarray | list = None,
        actor_axis: np.ndarray | list = [1, 0, 0],
        actor_axis_type: Literal["actor", "world"] = "actor",
        local_up_axis: np.ndarray | list | None = None,
        functional_point_id: int = None,
        pre_dis: float = 0.1,
        pre_dis_axis: Literal["grasp", "fp"] | np.ndarray | list = "grasp",
    ):

        if not self.plan_success:
            return [-1, -1, -1, -1, -1, -1, -1]

        actor_matrix = actor.get_pose().to_transformation_matrix()
        if functional_point_id is not None:
            place_start_pose = actor.get_functional_point(functional_point_id, "pose")
            z_transform = False
        else:
            place_start_pose = actor.get_pose()
            z_transform = True

        end_effector_pose = (self.robot.get_left_ee_pose() if arm_tag == "left" else self.robot.get_right_ee_pose())

        if constrain == "auto":
            grasp_direct_vec = place_start_pose.p - end_effector_pose[:3]
            if np.abs(np.dot(grasp_direct_vec, [0, 0, 1])) <= 0.1:
                place_pose = get_place_pose(
                    place_start_pose,
                    target_pose,
                    constrain="align",
                    actor_axis=grasp_direct_vec,
                    actor_axis_type="world",
                    align_axis=[1, 1, 0] if arm_tag == "left" else [-1, 1, 0],
                    z_transform=z_transform,
                    local_up_axis=local_up_axis,
                )
            else:
                camera_vec = transforms._toPose(end_effector_pose).to_transformation_matrix()[:3, 2]
                place_pose = get_place_pose(
                    place_start_pose,
                    target_pose,
                    constrain="align",
                    actor_axis=camera_vec,
                    actor_axis_type="world",
                    align_axis=[0, 1, 0],
                    z_transform=z_transform,
                    local_up_axis=local_up_axis,
                )
        else:
            place_pose = get_place_pose(
                place_start_pose,
                target_pose,
                constrain=constrain,
                actor_axis=actor_axis,
                actor_axis_type=actor_axis_type,
                align_axis=align_axis,
                z_transform=z_transform,
                local_up_axis=local_up_axis,
            )
        start2target = (transforms._toPose(place_pose).to_transformation_matrix()[:3, :3]
                        @ place_start_pose.to_transformation_matrix()[:3, :3].T)
        target_point = (start2target @ (actor_matrix[:3, 3] - place_start_pose.p).reshape(3, 1)).reshape(3) + np.array(
            place_pose[:3])

        ee_pose_matrix = t3d.quaternions.quat2mat(end_effector_pose[-4:])
        target_grasp_matrix = start2target @ ee_pose_matrix

        res_matrix = np.eye(4)
        res_matrix[:3, 3] = actor_matrix[:3, 3] - end_effector_pose[:3]
        res_matrix[:3, 3] = np.linalg.inv(ee_pose_matrix) @ res_matrix[:3, 3]
        target_grasp_qpose = t3d.quaternions.mat2quat(target_grasp_matrix)

        grasp_bias = target_grasp_matrix @ res_matrix[:3, 3]
        if pre_dis_axis == "grasp":
            target_dis_vec = target_grasp_matrix @ res_matrix[:3, 3]
            target_dis_vec /= np.linalg.norm(target_dis_vec)
        else:
            target_pose_mat = transforms._toPose(target_pose).to_transformation_matrix()
            if pre_dis_axis == "fp":
                pre_dis_axis = [0.0, 0.0, 1.0]
            pre_dis_axis = np.array(pre_dis_axis)
            pre_dis_axis /= np.linalg.norm(pre_dis_axis)
            target_dis_vec = (target_pose_mat[:3, :3] @ np.array(pre_dis_axis).reshape(3, 1)).reshape(3)
            target_dis_vec /= np.linalg.norm(target_dis_vec)
        res_pose = (target_point - grasp_bias - pre_dis * target_dis_vec).tolist() + target_grasp_qpose.tolist()
        return res_pose

    def take_action(self, action, action_type:Literal['qpos', 'ee']='qpos'):  # action_type: qpos or ee
        if self.take_action_cnt == self.step_lim or self.eval_success:
            return

        eval_video_freq = 1  # fixed
        if (self.eval_video_path is not None and self.take_action_cnt % eval_video_freq == 0):
            obs = self.now_obs.get("observation", {})
            for _cam in self.EVAL_VIDEO_CAMERAS:
                if _cam in obs:
                    self.eval_video_ffmpeg.stdin.write(obs[_cam]["rgb"].tobytes())
                    break

        self.take_action_cnt += 1
        print(f"step: \033[92m{self.take_action_cnt} / {self.step_lim}\033[0m", end="\r")

        self._update_render()
        if self.render_freq:
            self.viewer.render()

        actions = np.array([action])
        left_jointstate = self.robot.get_left_arm_jointState()
        right_jointstate = self.robot.get_right_arm_jointState()
        left_arm_dim = len(left_jointstate) - 1 if action_type == 'qpos' else 7
        right_arm_dim = len(right_jointstate) - 1 if action_type == 'qpos' else 7
        current_jointstate = np.array(left_jointstate + right_jointstate)

        left_arm_actions, left_gripper_actions, left_current_qpos, left_path = (
            [],
            [],
            [],
            [],
        )
        right_arm_actions, right_gripper_actions, right_current_qpos, right_path = (
            [],
            [],
            [],
            [],
        )

        left_arm_actions, left_gripper_actions = (
            actions[:, :left_arm_dim],
            actions[:, left_arm_dim],
        )
        right_arm_actions, right_gripper_actions = (
            actions[:, left_arm_dim + 1:left_arm_dim + right_arm_dim + 1],
            actions[:, left_arm_dim + right_arm_dim + 1],
        )
        left_current_gripper, right_current_gripper = (
            self.robot.get_left_gripper_val(),
            self.robot.get_right_gripper_val(),
        )

        left_gripper_path = np.hstack((left_current_gripper, left_gripper_actions))
        right_gripper_path = np.hstack((right_current_gripper, right_gripper_actions))

        if action_type == 'qpos':
            left_current_qpos, right_current_qpos = (
                current_jointstate[:left_arm_dim],
                current_jointstate[left_arm_dim + 1:left_arm_dim + right_arm_dim + 1],
            )
            left_path = np.vstack((left_current_qpos, left_arm_actions))
            right_path = np.vstack((right_current_qpos, right_arm_actions))

            # ========== TOPP ==========
            # TODO
            topp_left_flag, topp_right_flag = True, True

            try:
                times, left_pos, left_vel, acc, duration = (self.robot.left_mplib_planner.TOPP(left_path,
                                                                                            1 / 250,
                                                                                            verbose=True))
                left_result = dict()
                left_result["position"], left_result["velocity"] = left_pos, left_vel
                left_n_step = left_result["position"].shape[0]
            except Exception as e:
                # print("left arm TOPP error: ", e)
                topp_left_flag = False
                left_n_step = 50  # fixed

            if left_n_step == 0:
                topp_left_flag = False
                left_n_step = 50  # fixed

            try:
                times, right_pos, right_vel, acc, duration = (self.robot.right_mplib_planner.TOPP(right_path,
                                                                                                1 / 250,
                                                                                                verbose=True))
                right_result = dict()
                right_result["position"], right_result["velocity"] = right_pos, right_vel
                right_n_step = right_result["position"].shape[0]
            except Exception as e:
                # print("right arm TOPP error: ", e)
                topp_right_flag = False
                right_n_step = 50  # fixed

            if right_n_step == 0:
                topp_right_flag = False
                right_n_step = 50  # fixed
        
        elif action_type == 'ee':

            left_result = self.robot.left_plan_path(left_arm_actions[0])
            right_result = self.robot.right_plan_path(right_arm_actions[0])
            if left_result["status"] != "Success":
                left_n_step = 50
                topp_left_flag = False
                # print("left fail")
            else: 
                left_n_step = left_result["position"].shape[0]
                topp_left_flag = True
            
            if right_result["status"] != "Success":
                right_n_step = 50
                topp_right_flag = False
                # print("right fail")
            else:
                right_n_step = right_result["position"].shape[0]
                topp_right_flag = True

        # ========== Gripper ==========

        left_mod_num = left_n_step % len(left_gripper_actions)
        right_mod_num = right_n_step % len(right_gripper_actions)
        left_gripper_step = [0] + [
            left_n_step // len(left_gripper_actions) + (1 if i < left_mod_num else 0)
            for i in range(len(left_gripper_actions))
        ]
        right_gripper_step = [0] + [
            right_n_step // len(right_gripper_actions) + (1 if i < right_mod_num else 0)
            for i in range(len(right_gripper_actions))
        ]

        left_gripper = []
        for gripper_step in range(1, left_gripper_path.shape[0]):
            region_left_gripper = np.linspace(
                left_gripper_path[gripper_step - 1],
                left_gripper_path[gripper_step],
                left_gripper_step[gripper_step] + 1,
            )[1:]
            left_gripper = left_gripper + region_left_gripper.tolist()
        left_gripper = np.array(left_gripper)

        right_gripper = []
        for gripper_step in range(1, right_gripper_path.shape[0]):
            region_right_gripper = np.linspace(
                right_gripper_path[gripper_step - 1],
                right_gripper_path[gripper_step],
                right_gripper_step[gripper_step] + 1,
            )[1:]
            right_gripper = right_gripper + region_right_gripper.tolist()
        right_gripper = np.array(right_gripper)

        now_left_id, now_right_id = 0, 0
        # Frame capture for POLICY-ROLLOUT data collection (save_data on; never in
        # eval-client mode). Default: ONE frame per policy command, captured before
        # execution — so hdf5 rows align 1:1 with policy commands and next-frame
        # actions keep the training-time step scale (fine-tunable data). Contact/
        # collision events still accumulate across the whole command and flush into
        # its frame at the next capture, so no events are lost — labels are just
        # per-command instead of per-0.06s. Set ROLLOUT_DENSE_RECORD=1 to restore
        # the save_freq-substep cadence (frame-parity with take_dense_action, for
        # per-timestep-label datasets mixing rollout + CuRobo frames).
        _rec = bool(getattr(self, "save_data", False)) and getattr(self, "save_freq", None)
        _dense = os.environ.get("ROLLOUT_DENSE_RECORD", "0") == "1"
        if _rec and not _dense:
            self._take_picture()  # state[t]: pre-execution state of command t
        _ctrl_i = 0

        # ========== Control Loop ==========
        while now_left_id < left_n_step or now_right_id < right_n_step:

            if (now_left_id < left_n_step and now_left_id / left_n_step <= now_right_id / right_n_step):
                if topp_left_flag:
                    self.robot.set_arm_joints(
                        left_result["position"][now_left_id],
                        left_result["velocity"][now_left_id],
                        "left",
                    )
                self.robot.set_gripper(left_gripper[now_left_id], "left")

                now_left_id += 1

            if (now_right_id < right_n_step and now_right_id / right_n_step <= now_left_id / left_n_step):
                if topp_right_flag:
                    self.robot.set_arm_joints(
                        right_result["position"][now_right_id],
                        right_result["velocity"][now_right_id],
                        "right",
                    )
                self.robot.set_gripper(right_gripper[now_right_id], "right")

                now_right_id += 1

            if getattr(self, 'enable_collision_metrics', False) and hasattr(self, 'robot_link_names'):
                self._snapshot_static_object_poses()
            self.scene.step()
            self._update_render()

            # picture BEFORE check_collisions — same substep/frame boundary as
            # take_dense_action, so policy-rollout labels line up with CuRobo
            # data at the frame level (a boundary event lands on the same frame)
            if _rec and _dense and _ctrl_i % self.save_freq == 0:
                self._take_picture()
            _ctrl_i += 1

            if getattr(self, 'enable_collision_metrics', False) and hasattr(self, 'robot_link_names'):
                self.check_collisions()

            if self.check_success():
                self.eval_success = True
                self.get_obs() # update obs
                if _rec:
                    self._take_picture()  # final frame at task success
                if (self.eval_video_path is not None):
                    # Same camera preference as the per-step frames, otherwise
                    # the success frame jumps to a different viewpoint.
                    _obs = self.now_obs.get("observation", {})
                    for _cam in self.EVAL_VIDEO_CAMERAS:
                        if _cam in _obs:
                            self.eval_video_ffmpeg.stdin.write(_obs[_cam]["rgb"].tobytes())
                            break
                return

        self._update_render()
        if self.render_freq:  # UI
            self.viewer.render()
    
    def take_dense_action(self, control_seq, save_freq=-1):
        """Extends base take_dense_action with per-step collision detection."""
        left_arm     = control_seq["left_arm"]
        left_gripper = control_seq["left_gripper"]
        right_arm    = control_seq["right_arm"]
        right_gripper= control_seq["right_gripper"]

        save_freq = self.save_freq if save_freq == -1 else save_freq
        if save_freq is not None:
            self._take_picture()

        max_control_len = 0
        if left_arm    is not None: max_control_len = max(max_control_len, left_arm["position"].shape[0])
        if left_gripper is not None: max_control_len = max(max_control_len, left_gripper["num_step"])
        if right_arm   is not None: max_control_len = max(max_control_len, right_arm["position"].shape[0])
        if right_gripper is not None: max_control_len = max(max_control_len, right_gripper["num_step"])

        _collision_active = getattr(self, 'enable_collision_metrics', False) and hasattr(self, 'robot_link_names')

        for control_idx in range(max_control_len):
            if left_arm is not None and control_idx < left_arm["position"].shape[0]:
                self.robot.set_arm_joints(left_arm["position"][control_idx],
                                          left_arm["velocity"][control_idx], "left")
            if left_gripper is not None and control_idx < left_gripper["num_step"]:
                self.robot.set_gripper(left_gripper["result"][control_idx], "left",
                                       left_gripper["per_step"])
            if right_arm is not None and control_idx < right_arm["position"].shape[0]:
                self.robot.set_arm_joints(right_arm["position"][control_idx],
                                          right_arm["velocity"][control_idx], "right")
            if right_gripper is not None and control_idx < right_gripper["num_step"]:
                self.robot.set_gripper(right_gripper["result"][control_idx], "right",
                                       right_gripper["per_step"])

            if _collision_active:
                self._snapshot_static_object_poses()
            self.scene.step()

            if self.render_freq and control_idx % self.render_freq == 0:
                self._update_render()
                self.viewer.render()
            if save_freq is not None and control_idx % save_freq == 0:
                self._update_render()
                self._take_picture()
            if _collision_active:
                self.check_collisions()

        if save_freq is not None:
            self._take_picture()
        return True

    # =========================================================== Extra Curobo Utils ===========================================================

    def update_world(self, exclude_obstacles: bool = None):
        """Updates CuRobo Collision World Model with new collision objects.

        exclude_obstacles=None (the default) resolves from the run's planner
        regime (planner_exclude_obstacles, legacy-coupled to
        enable_collision_metrics) so the 19 bare mid-task update_world() calls
        in task files keep the configured blindness instead of silently
        re-including clutter. Pass an explicit bool to override.
        """
        if exclude_obstacles is None:
            exclude_obstacles = getattr(self, "planner_exclude_obstacles", None)
            if exclude_obstacles is None:
                exclude_obstacles = bool(getattr(self, "enable_collision_metrics", False))
        collision_dict = {"mesh": {}, "cuboid": {}}
        if self.collision_list:
            for info in self.collision_list:
                if exclude_obstacles and info.get("is_obstacle", False):
                    continue
                actor = info["actor"]
                collision_path = info["collision_path"]
                if os.path.isdir(collision_path): # if actor is made from multiple obj files
                    name_prefix = actor.get_name()
                    if "link" in info:
                        if isinstance(info["link"], list):
                            pose = sapien.Pose()
                            pose.p = actor.get_link_pose(info["link"][0]).p
                            pose.q = actor.get_link_pose(info["link"][1]).q
                        else:
                            pose = actor.get_link_pose(info["link"])
                    elif "pose" in info:
                        pose = info["pose"]
                    else:
                        pose = actor.get_pose()
                    np_pose = np.concatenate([pose.p, pose.q]).tolist()
                    convex_collision_dict = self.collision_dict_from_convex_obj_dir(
                        collision_path,
                        pose=np_pose,
                        scale=actor.scale,
                        name_prefix = name_prefix,
                        files = info.get("files", None)
                    )
                    collision_dict["mesh"] = (
                        collision_dict["mesh"] | convex_collision_dict["mesh"]
                    )
                else:
                    if "pose" in info:
                        pose = info["pose"]
                    else:
                        pose = actor.get_pose()
                    np_pose = np.concatenate([pose.p, pose.q]).tolist()
                    collision_dict["mesh"][f"{actor.get_name()}_{np_pose}_{self._collision_cache_id()}"] = {
                            "file_path": collision_path,
                            "pose": np_pose,
                            "scale": actor.scale,
                        }

        if self.cuboid_collision_list:
            for info in self.cuboid_collision_list:
                name = info["name"]
                dims = info["dims"]
                pose = info["pose"]
                collision_dict["cuboid"][f"{name}_{pose}_{self._collision_cache_id()}"] = {
                    "dims": dims,
                    "pose": pose,
                }
        self.robot.update_world(collision_dict)
    
    def collision_dict_from_convex_obj_dir(
        self,
        obj_dir: str | Path,
        *,
        name_prefix: str = "shelf_part",
        pose: tuple[float, float, float, float, float, float, float],  # [x,y,z,qw,qx,qy,qz]
        scale: tuple[float, float, float],  # e.g. (0.6, 0.8, 0.4)
        glob_pattern: str = "*.obj",
        files: list[str] = None,
        recursive: bool = False,
    ) -> dict:
        """
        Used to convert a directory of obj files into a dict of collision objects for curobo planner.
        Returns collision_dict in the form:
        collision_dict["mesh"][<name>] = {"file_path": ..., "pose": ..., "scale": ...}

        One entry per OBJ file (skips invalid/empty OBJs).
        """
        obj_dir = Path(obj_dir)
        if not obj_dir.exists() or not obj_dir.is_dir():
            raise FileNotFoundError(f"OBJ directory not found or not a directory: {obj_dir}")

        if files is not None:
            obj_files = []
            for file_name in files:
                p = obj_dir / file_name
                if p.is_file():
                    obj_files.append(p)
            obj_files = sorted(obj_files)
        else:
            it = obj_dir.rglob(glob_pattern) if recursive else obj_dir.glob(glob_pattern)
            obj_files = sorted([p for p in it if p.is_file()])

        if not obj_files:
            if files is not None:
                raise FileNotFoundError(
                    f"No requested OBJ files found in {obj_dir}. Requested files: {files}"
                )
            raise FileNotFoundError(
                f"No OBJ files found in {obj_dir} with pattern '{glob_pattern}' (recursive={recursive})"
            )

        collision_dict = {"mesh": {}}

        for i, p in enumerate(obj_files):
            # Validate OBJ so cuRobo/trimesh won't crash later
            try:
                m = trimesh.load(str(p), force="mesh", process=False)
            except Exception: # means the obj file is invalid
                continue

            if isinstance(m, trimesh.Scene):
                if len(m.geometry) == 0:
                    continue
                # concatenate ensures vertices/faces exist
                m = trimesh.util.concatenate(tuple(m.geometry.values()))

            if getattr(m, "vertices", None) is None or len(m.vertices) == 0:
                continue
            if getattr(m, "faces", None) is None or len(m.faces) == 0:
                continue

            if getattr(self, "_eval_env", None):
                part_name = f"{p.name}_{self._collision_cache_id()}"
            else:
                part_name = f"{p}_{self.seed}"
            collision_dict["mesh"][part_name] = {
                "file_path": str(p),
                "pose": list(pose),
                "scale": list(scale),
            }
        
        if not collision_dict["mesh"]:
            raise ValueError(
                f"No valid mesh files were added from directory: {obj_dir} "
            )

        return collision_dict
        
    def attach_object(self, actor, file_path, arms_tag: str):
        """
        Attach a held object to the robot in Curobo Planning. Currently supports Actor or ArticulationActor.
        """
        pose = actor.get_pose()
        np_pose = np.concatenate([pose.p, pose.q]).tolist()
        object = {
            "name": actor.get_name(),
            "pose": np_pose,
            "file_path": file_path,
            "scale": actor.scale,
        }
        self.robot.attach_object(object, arms_tag=arms_tag)
        if hasattr(self, "_held_actors"):  # exclude the grasped object from proximity
            self._held_actors[arms_tag] = actor.get_name()

    def detach_object(self, arms_tag: str):
        """
        Detach the attached objects from the robot in Curobo Planning.
        """
        self.robot.detach_object(arms_tag=arms_tag)
        if hasattr(self, "_held_actors"):
            self._held_actors[arms_tag] = None

    def enable_obstacle(self, enable: bool, mesh_names: list[str] = [], obb_names: list[str] = []):
        self.robot.enable_obstacle(enable, mesh_names=mesh_names, obb_names=obb_names)

    def add_gripper_operating_area(self):
        # prohibit the area under the gripper start state so there are no initial collisions with obstacles
        if "table" not in self.prohibited_area:
            return
        x_half_width = 0.075
        ymax = -0.18
        ymin = -0.26
        self.prohibited_area["table"].append([-0.3-x_half_width, ymin, -0.3+x_half_width, ymax])
        self.prohibited_area["table"].append([0.3-x_half_width, ymin, 0.3+x_half_width, ymax])
    
    def add_operating_area(self, pose, width = 0.07, length = 0.28, direction = "forward"):
        if "table" not in self.prohibited_area:
            return
        # add a prohibited area in the space where the arm approaches a grasp or place. For horizontal movement.
        if direction == "forward": # from -y to +y
            xmin = pose[0] - width/2
            xmax = pose[0] + width/2
            ymin = pose[1] - length
            ymax = pose[1]
        elif direction == "right": # from -x to +x
            xmin = pose[0] - length
            xmax = pose[0]
            ymin = pose[1] - width/2
            ymax = pose[1] + width/2
        elif direction == "left": # from +x to -x
            xmin = pose[0]
            xmax = pose[0] + length
            ymin = pose[1] - width/2
            ymax = pose[1] + width/2
        self.prohibited_area["table"].append([xmin, ymin, xmax, ymax])
