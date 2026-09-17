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
import xml.etree.ElementTree as ET

from bench_envs._bench_base_task import Bench_base_task
from envs.utils import *
from bench_envs.utils import *
import math
from envs.robot import Robot
from envs.camera import Camera
from envs.utils.actor_utils import Actor, ArticulationActor

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


class Kitchen_base_large(Bench_base_task):

    FURNITURE_NAMES = {
        "table", "wall", "ground",
        # structural container on the countertop (create_actor static, always present)
        "basket_right",
        # filler box between cabinet URDF and wall (create_box static)
        "cabinet_wall_filler",
        # fridge / microwave / cabinet are URDF articulations — their links do NOT
        # appear in scene.get_all_actors(), so no entry needed here.
    }

    def __init__(self):
        pass

    def _extract_intrinsic_scale(self, model_data: dict) -> float:
        """
        Extract a single (uniform) intrinsic scale value from a model_data dict.

        The assets store scale either as a scalar or as a 3-vector; we use the
        first component since kitchen furniture scaling is treated as uniform.
        """
        base = model_data.get("scale", 1.0)
        if isinstance(base, (list, tuple)) and len(base) > 0:
            return float(base[0])
        return float(base)

    def _get_asset_model_scale_create_actor(self, modelname: str, model_id: int = 0) -> float:
        """
        Intrinsic scale from `assets/objects/<modelname>/model_data<model_id>.json`.

        `create_actor.py` treats the `scale=` argument as absolute.
        The kitchen env historically used `*_left_scale` as a multiplier, so we
        multiply by this intrinsic scale to preserve the intended sizes.
        """
        modeldir = Path(os.environ["ASSETS_ROOT"]) / "objects" / modelname
        json_file = modeldir / f"model_data{model_id}.json"
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                model_data = json.load(f)
        except Exception:
            return 1.0

        return self._extract_intrinsic_scale(model_data)

    def _get_asset_model_scale_sapien_urdf(self, modelname: str, modelid: int = 0) -> float:
        """
        Intrinsic scale from `assets/objects/<modelname>/<variant>/model_data.json`.

        Matches the variant selection logic in `create_sapien_urdf_obj`.
        """
        modeldir = Path(os.environ["ASSETS_ROOT"]) / "objects" / modelname
        try:
            model_list = [m for m in modeldir.iterdir() if m.is_dir() and m.name != "visual"]
            def _variant_sort_key(p: Path) -> int:
                m = re.search(r"\d+", p.name)
                return int(m.group()) if m else 0
            model_list = sorted(
                model_list,
                key=_variant_sort_key,
            )
            if not model_list:
                return 1.0
            idx = int(modelid)
            idx = max(0, min(idx, len(model_list) - 1))
            chosen = model_list[idx]
            json_file = chosen / "model_data.json"
            with open(json_file, "r", encoding="utf-8") as f:
                model_data = json.load(f)
        except Exception:
            return 1.0

        return self._extract_intrinsic_scale(model_data)

    def apply_srdf_collisions(self, articulation, srdf_path: Path) -> None:
        """
        Apply SRDF <disable_collisions link1="..." link2="..."/> by directly
        editing PhysX collision-group bitmasks on collision shapes.

        This does not rely on URDFLoader's srdf kwargs (which may not exist
        in the local SAPIEN build).
        """
        if srdf_path is None:
            return
        if not srdf_path.exists():
            return

        try:
            tree = ET.parse(str(srdf_path))
            root = tree.getroot()
        except Exception as e:
            return

        disable_tags = root.findall(".//disable_collisions")
        parsed_pairs: list[tuple[str, str, str]] = []
        for tag in disable_tags:
            link1_name = tag.get("link1")
            link2_name = tag.get("link2")
            reason = tag.get("reason", "")
            if link1_name and link2_name:
                parsed_pairs.append((link1_name, link2_name, reason))

        if not parsed_pairs:
            return

        # Map SRDF link names to actual SAPIEN articulation link objects.
        link_map = {}
        try:
            for link in articulation.get_links():
                link_map[link.get_name()] = link
        except Exception as e:
            return

        # Only touch links that appear in at least one SRDF pair.
        involved = set()
        for l1, l2, _ in parsed_pairs:
            involved.add(l1)
            involved.add(l2)

        # Cache collision shapes per link name.
        link_shapes: dict[str, list] = {}
        for link_name in sorted(involved):
            link_obj = link_map.get(link_name)
            if link_obj is None:
                continue

            shapes = None
            try:
                if hasattr(link_obj, "get_collision_shapes") and callable(getattr(link_obj, "get_collision_shapes")):
                    shapes = link_obj.get_collision_shapes()
                elif hasattr(link_obj, "collision_shapes"):
                    cs = getattr(link_obj, "collision_shapes")
                    shapes = cs() if callable(cs) else cs
            except Exception as e:
                shapes = None

            if shapes is None:
                continue
            link_shapes[link_name] = list(shapes)

        if not link_shapes:
            return

        # Collision groups doc (SAPIEN/PhysX wrapper):
        #   collide iff (g0 & other.g1) or (g1 & other.g0) AND NOT ((g2 & other.g2) and (g3 lower16 equal))
        # So we:
        # 1) set g3 lower16 to a constant for all shapes in this articulation (so ignore-id can match)
        # 2) for each disabled link pair, set a unique bit in g2 on shapes of both links.
        srdf_collision_id = 0xBEEF  # lower16 bits

        def _apply_group_bit(shape, bit: int) -> None:
            groups = shape.get_collision_groups()
            if groups is None or len(groups) != 4:
                raise RuntimeError(f"Unexpected collision groups format: {groups}")
            g0, g1, g2, g3 = groups
            g3_new = (int(g3) & 0xFFFF0000) | srdf_collision_id
            g2_new = int(g2) | bit
            shape.set_collision_groups([int(g0), int(g1), int(g2_new), int(g3_new)])

        # First, normalize g3 lower16 for all touched shapes so that ignore checks can match.
        for link_name, shapes in link_shapes.items():
            for shape in shapes:
                try:
                    groups = shape.get_collision_groups()
                    if groups is None or len(groups) != 4:
                        continue
                    g0, g1, g2, g3 = groups
                    g3_new = (int(g3) & 0xFFFF0000) | srdf_collision_id
                    shape.set_collision_groups([int(g0), int(g1), int(g2), int(g3_new)])
                except Exception:
                    # Don't fail the whole SRDF application for one bad shape.
                    continue

        # Apply unique ignore bits per disable_collisions pair.
        # Note: if there are more pairs than we can represent safely, reuse may cause over-ignore.
        for i, (link1_name, link2_name, reason) in enumerate(parsed_pairs):
            if i >= 31:
                return
            ignore_bit = 1 << i
            l1_shapes = link_shapes.get(link1_name, [])
            l2_shapes = link_shapes.get(link2_name, [])
            if not l1_shapes or not l2_shapes:
                continue
            try:
                for shape in l1_shapes:
                    _apply_group_bit(shape, ignore_bit)
                for shape in l2_shapes:
                    _apply_group_bit(shape, ignore_bit)
            except Exception as e:
                continue

    def _init_task_env_(self, table_xy_bias=[0, 0], table_height_bias=0, **kwags):
        """
        Initialization TODO
        - `self.FRAME_IDX`: The index of the file saved for the current scene.
        - `self.fcitx5-configtool`: Left gripper pose (close <=0, open >=0.4).
        - `self.ep_num`: Episode ID.
        - `self.task_name`: Task name.
        - `self.save_dir`: Save path.`
        - `self.left_original_pose`: Left arm original pose.
        - `self.right_original_pose`: Right arm original pose.
        - `self.left_arm_joint_id`: [6,14,18,22,26,30].
        - `self.right_arm_joint_id`: [7,15,19,23,27,31].
        - `self.render_fre`: Render frequency.
        """
        super().__init__() # need to fix this. right now this does nothing. it should be calling gym.Env.__init__()
        ta.setup_logging("CRITICAL")  # hide logging
        np.random.seed(kwags.get("seed", 0))
        torch.manual_seed(kwags.get("seed", 0))
        # random.seed(kwags.get('seed', 0))
        self.seed = kwags.get("seed", 0)
        print_c(f"#### Seed value {self.seed} ####", "YELLOW")

        self.FRAME_IDX = 0
        self.task_name = kwags.get("task_name")
        self.save_dir = kwags.get("save_path") or os.environ.get("DATA_ROOT", "data")
        self.ep_num = kwags.get("now_ep_num", 0)
        self.render_freq = kwags.get("render_freq", 10)
        self.data_type = kwags.get("data_type", None)
        self.save_data = kwags.get("save_data", False)
        self.dual_arm = kwags.get("dual_arm", True)
        self.eval_mode = kwags.get("eval_mode", False)
        self.sample_d = kwags.get("sample_d", "objects")
        self.enable_collision_metrics = kwags.get("enable_collision_metrics", False)
        # None -> legacy behavior (planner blind iff metrics on); explicit true/false
        # decouples planner obstacle-awareness from metric recording (negative samples)
        self.planner_exclude_obstacles = kwags.get("planner_exclude_obstacles", None)

        self.cuboid_collision_list = [] # list of cuboid collision objects for curobo planner
        self.cluttered_objs = list()

        self.need_topp = True  # TODO

        # Random
        random_setting = kwags.get("domain_randomization")
        self.random_background = random_setting.get("random_background", False)
        self.clean_background_rate = random_setting.get("clean_background_rate", 1)
        self.random_head_camera_dis = random_setting.get("random_head_camera_dis", 0)
        self.random_table_height = random_setting.get("random_table_height", 0)
        self.random_light = random_setting.get("random_light", False)
        self.crazy_random_light_rate = random_setting.get("crazy_random_light_rate", 0)
        self.crazy_random_light = (0 if not self.random_light else np.random.rand() < self.crazy_random_light_rate)
        self.random_embodiment = random_setting.get("random_embodiment", False)  # TODO
        self.cluttered_table = random_setting.get("cluttered_table", False)
        self.obstacle_height = random_setting.get("obstacle_height", "short")
        self.obstacle_density =   random_setting.get("obstacle_density", 3)

        self._parse_perturbations(random_setting)

        self.file_path = []
        self.plan_success = True
        self.step_lim = None
        self.fix_gripper = False
        self.setup_scene()

        self.left_js = None
        self.right_js = None
        self.raw_head_pcl = None
        self.real_head_pcl = None
        self.real_head_pcl_color = None

        self.now_obs = {}
        self.take_action_cnt = 0
        self.eval_video_path = kwags.get("eval_video_save_dir", None)
        self.incl_collision = kwags.get("include_collision", False)
        self.jitter_basket = kwags.get("jitter_basket", True)
        self.save_freq = kwags.get("save_freq")
        self.video_fps = kwags.get("video_fps", 30)
        self.world_pcd = None

        # table: main countertop; shelf0/1: pantry rack shelves; fridge/cabinet: internal storage volumes
        self.prohibited_area = {
            "table": [],
            "shelf0": [],
            "shelf1": [],
            "fridge": [],
            "cabinet": [],
        }
        # Base env currently does not spawn distractors; keep this for compatibility.
        self.record_cluttered_objects = []

        self.eval_success = False
        self.table_z_bias = (np.random.uniform(low=-self.random_table_height, high=0) + table_height_bias)  # TODO
        self.need_plan = kwags.get("need_plan", True)
        self.left_joint_path = kwags.get("left_joint_path", [])
        self.right_joint_path = kwags.get("right_joint_path", [])
        self.left_cnt = 0
        self.right_cnt = 0

        # Fridge, microwave, and table-side container (`basket_right`) on the table front edge.
        # Rotations in degrees (roll, pitch, yaw) and uniform scales.
        # Fixed at the base env level (shared by all tasks).
        self.fridge_left_rot = [-90.0, 0.0, 90.0]
        self.fridge_left_scale = 0.5

        self.microwave_left_rot = [-90.0, 180.0, 0.0]
        self.microwave_left_scale = 1.4

        self.basket_right_rot = [0.0, 0.0, 90.0]
        self.basket_right_scale = 1.4
        # World-frame additive jitter on nominal basket pose (table_xy_bias frame).
        self.basket_right_position_jitter_x = (-0.04, 0.04)
        self.basket_right_position_jitter_y = (-0.04, 0.04)
        # Table-side container (tasks use `basket_right` as the reference actor).
        self.basket_right_modelname = "063_tabletrashbin"
        self.basket_right_model_id = 6

        self.scene_id = kwags.get("scene_id") if kwags.get("scene_id") is not None else np.random.randint(0,3)  # for furniture arrangement
        print_c(f"Scene {self.scene_id} is selected", "YELLOW")

        # Cabinet scale: currently only uniform scaling is supported by SAPIEN's URDF loader.
        # This parameter allows you to uniformly resize the cabinet; to truly scale only height,
        # the underlying meshes/URDF would need to encode per-axis scale.
        self.cabinet_scale = 0.5

        self.instruction = None  # for Eval

        self.collision_list = [] # list of collision objects for curobo planner
        self._init_collision_metrics()

        # Map semantic appliance roles to underlying assets
        self.kitchen_appliance_assets = {
            "fridge": {"modelname": "036_cabinet", "default_modelid": 46653},
            "cabinet": {"modelname": "036_cabinet", "default_modelid": 46653},
            "drawer": {"modelname": "036_cabinet", "default_modelid": 46653},
        }

        self.load_robot(**kwags)
        self.create_static_elements(table_xy_bias=table_xy_bias, table_height=0.74)
        self.load_camera(**kwags)
        self.robot.move_to_homestate()

        render_freq = self.render_freq
        self.render_freq = 0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

        self.robot.set_origin_endpose()

        self.load_actors()
        # self.load_basic_kitchen_items()
        if self.cluttered_table:
            self.get_cluttered_surfaces()

        self._apply_specular_ood()
        self._furniture_texture_targets = [
            ("fridge_left", "fridge"),
            ("microwave_left", "microwave"),
            ("basket_right", "basket"),
            ("cabinet", "shelf"),
        ]
        self._apply_furniture_texture_ood()

        if self.enable_collision_metrics:
            self._build_collision_name_sets()

        # per-step proximity (nearest-obstacle clearance) tracking
        if kwags.get("data_type", {}).get("proximity", True):
            self._init_proximity_tracking(kwags.get("proximity_tracking", {}))

        # Even for a minimal scene, ensure that articulated objects like the drawer
        # are placed in a stable configuration.
        is_stable, unstable_list = self.check_stable()
        if not is_stable:
            raise UnStableError(
                f'Objects is unstable in seed({kwags.get("seed", 0)}), unstable objects: {", ".join(unstable_list)}')

        exclude_obs = self.planner_exclude_obstacles
        if exclude_obs is None:
            exclude_obs = self.enable_collision_metrics  # legacy coupling
        if not exclude_obs:
            self.update_world()
            print(f"\033[93m{self.task_name} curobo planner consider obstacles\033[0m")
        else:
            self.update_world(exclude_obstacles=True)  # planner blind to clutter (eval / negative samples)
            print(f"\033[38;5;208m{self.task_name} curobo planner skips clutter obstacles\033[0m")

        if self.eval_mode:
            with open(os.path.join(os.environ["BENCH_ROOT"], "bench_task_config", "_bench_eval_step_limit.yml"), "r") as f:
                try:
                    data = yaml.safe_load(f)
                    self.step_lim = data[self.task_name]
                except:
                    print(f"{self.task_name} not in step limit file, set to 1000")
                    self.step_lim = 1000

        # info
        self.info = dict()
        self.info["cluttered_table_info"] = self.record_cluttered_objects
        self.info["texture_info"] = {
            "wall_texture": self.wall_texture,
            "table_texture": self.table_texture,
            "floor_texture": self.floor_texture,
        }
        self.info["info"] = {}

        self.stage_success_tag = False

    def create_static_elements(self, table_xy_bias=[0, 0], table_height=0.74):
        # create static furniture elements
        self.table_xy_bias = table_xy_bias
        wall_texture, table_texture, floor_texture = None, None, None
        table_height += self.table_z_bias

        if self.random_background:
            texture_type = "seen" if not self.eval_mode else "unseen"
            directory_path = f"{os.environ['ASSETS_ROOT']}/background_texture/{texture_type}"
            file_count = len(
                [name for name in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, name))])

            # wall_texture, table_texture = np.random.randint(0, file_count), np.random.randint(0, file_count)
            wall_texture = 43
            table_texture = 141
            floor_texture = 38

            self.wall_texture = f"seen/{wall_texture}"
            self.table_texture = f"seen/{table_texture}"
            self.floor_texture = f"seen/{floor_texture}"
            if np.random.rand() <= self.clean_background_rate:
                self.wall_texture = None
            if np.random.rand() <= self.clean_background_rate:
                self.table_texture = None
            if np.random.rand() <= self.clean_background_rate:
                self.floor_texture = None
        else:
            self.wall_texture, self.table_texture, self.floor_texture = None, None, None
        
        # floor--------------------------------------------------------
        self.floor_parts = []

        positions = [
            [1, 1, 0],    # top-right
            [-1, 1, 0],   # top-left
            [1, -1, 0],   # bottom-right
            [-1, -1, 0],  # bottom-left
        ]

        for i, pos in enumerate(positions):
            floor_part = create_visual_textured_box(
                self.scene,
                sapien.Pose(p=pos),
                half_size=[1, 1, 0.005],
                color=(0.85, 0.85, 0.85),
                name=f"floor_{i}",
                texture_id=self.floor_texture,
            )
            self.floor_parts.append(floor_part)
        
        # wall--------------------------------------------------------
        self.wall = create_box(
            self.scene,
            sapien.Pose(p=[0, 1, 1.5]),
            half_size=[3, 0.6, 1.5],
            color=(1, 0.9, 0.9),
            name="wall",
            texture_id=self.wall_texture,
            is_static=True,
        )

        # Unified countertop (single static body).
        # Keep name `table` and place so the top surface is at `table_height`,
        # matching the previous create_table(...) semantics.
        counter_length = 1.2
        counter_width = 0.7
        counter_thickness = 0.05
        tabletop_pose = sapien.Pose([0.0, 0.0, -counter_thickness / 2])
        tabletop_half_size = [counter_length / 2, counter_width / 2, counter_thickness / 2]

        # Add a recessed base volume under the top so it reads like a real
        # countertop/cabinet block, while leaving a front "toe-kick" gap so
        # the robot can get closer.
        front_recess = 0.12
        base_height = max(0.0, table_height - counter_thickness)
        base_depth = max(0.0, counter_width - front_recess)
        base_half_size = [counter_length / 2, base_depth / 2, base_height / 2]
        # Local frame: actor origin is at the top surface (world z = table_height).
        # Shift the base backward (+y) so the front edge is recessed.
        base_pose = sapien.Pose(
            [0.0, front_recess / 2, -(counter_thickness / 2 + base_height / 2)]
        )

        counter_builder = self.scene.create_actor_builder()
        counter_builder.set_physx_body_type("static")
        counter_builder.add_box_collision(
            pose=tabletop_pose,
            half_size=tabletop_half_size,
            material=self.scene.default_physical_material,
        )
        if base_height > 1e-6 and base_depth > 1e-6:
            counter_builder.add_box_collision(
                pose=base_pose,
                half_size=base_half_size,
                material=self.scene.default_physical_material,
            )
        if self.table_texture is not None:
            texturepath = f"{os.environ['ASSETS_ROOT']}/background_texture/{self.table_texture}.png"
            texture2d = sapien.render.RenderTexture2D(texturepath)
            material = sapien.render.RenderMaterial()
            material.set_base_color_texture(texture2d)
            material.base_color = [1, 1, 1, 1]
            material.metallic = 0.1
            material.roughness = 0.3
            counter_builder.add_box_visual(pose=tabletop_pose, half_size=tabletop_half_size, material=material)
            # Slightly different roughness for the base so it doesn't look like a floating slab.
            base_material = sapien.render.RenderMaterial(base_color=[0.55, 0.47, 0.38, 1])
            base_material.metallic = 0.0
            base_material.roughness = 0.8
            if base_height > 1e-6 and base_depth > 1e-6:
                counter_builder.add_box_visual(pose=base_pose, half_size=base_half_size, material=base_material)
        else:
            counter_builder.add_box_visual(pose=tabletop_pose, half_size=tabletop_half_size, material=(1, 1, 1))
            if base_height > 1e-6 and base_depth > 1e-6:
                counter_builder.add_box_visual(pose=base_pose, half_size=base_half_size, material=(0.55, 0.47, 0.38))

        counter_builder.set_initial_pose(sapien.Pose(p=[table_xy_bias[0], table_xy_bias[1], table_height]))
        self.table = counter_builder.build(name="table")

        # Place static appliances on the table front edge.
        self._load_fridge_on_table(table_height, table_xy_bias)
        self._load_microwave_on_table(table_height, table_xy_bias)
        self._load_basket_on_table(table_height, table_xy_bias)
        self._load_cabinet_on_table(table_height, table_xy_bias)

        self._add_cabinet_wall_filler()
        if self.incl_collision:
            self.add_collision()

        # Additional kitchen appliances (wall cabinets, pantry rack, etc.)
        # can be re-enabled later via _load_kitchen_appliances if needed.

    def _load_fridge_on_table(self, table_height: float, table_xy_bias):
        """Place the static fridge on the right front edge of the table."""
        y_front = table_xy_bias[1] + 0.30
        x_fridge = table_xy_bias[0] + 0.40
        z_fridge = table_height + 0.24

        fx_roll_deg, fx_pitch_deg, fx_yaw_deg = self.fridge_left_rot
        fx_ax = math.radians(fx_roll_deg)
        fx_ay = math.radians(fx_pitch_deg)
        fx_az = math.radians(fx_yaw_deg)
        fqx, fqy, fqz, fqw = t3d.euler.euler2quat(fx_ax, fx_ay, fx_az)
        fridge_quat = [fqw, fqx, fqy, fqz]

        pose_fridge = sapien.Pose([x_fridge, y_front, z_fridge], fridge_quat)
        self.fridge_left = self._create_bench_cabinet(
            asset_dir_name="124_fridge_hivvdf",
            pose=pose_fridge,
            fix_root_link=True,
            extra_scale=self.fridge_left_scale,
        )
        if self.fridge_left is not None:
            self.fridge_left.set_name("fridge_left")
            self.add_prohibit_area(self.fridge_left, padding=0.05, area="table")

    def _get_scene_obj_locations(self, object_name="microwave"):
        if self.scene_id == 0: 
            microwave_location = [0.0, 0.30]
            basket_location = [-0.37, 0.12, 0]
        elif self.scene_id == 1:
            microwave_location = [-0.4, 0.30]
            basket_location = [0, 0.15, 0]
        elif self.scene_id == 2:
            microwave_location = [-0.4, 0.10]
            # self.microwave_left_rot = [0.0, 180.0, 0.0]
            basket_location = [-0.4, 0.07, 0.927]
        else:
            raise ValueError(f"Invalid scene_id {self.scene_id}")
        if object_name == "microwave":
            return microwave_location
        elif object_name == "basket":
            return basket_location
        raise ValueError(f"Object name {object_name} is not supported")

    def _load_microwave_on_table(self, table_height: float, table_xy_bias):
        """Place the static microwave in the middle of the front edge of the table."""
        
        x_microwave, y_front = self._get_scene_obj_locations()
        # x_microwave = table_xy_bias[0] + 0.0
        z_microwave = table_height + 0.02

        mw_roll_deg, mw_pitch_deg, mw_yaw_deg = self.microwave_left_rot
        mw_ax = math.radians(mw_roll_deg)
        mw_ay = math.radians(mw_pitch_deg)
        mw_az = math.radians(mw_yaw_deg)
        mw_qx, mw_qy, mw_qz, mw_qw = t3d.euler.euler2quat(mw_ax, mw_ay, mw_az)
        microwave_quat = [mw_qw, mw_qx, mw_qy, mw_qz]

        pose_microwave = sapien.Pose([x_microwave, y_front, z_microwave], microwave_quat)
        try:
            intrinsic_scale = self._get_asset_model_scale_sapien_urdf(
                modelname="044_microwave",
                modelid=0,
            )
            final_scale = intrinsic_scale * float(self.microwave_left_scale)
            microwave_actor = create_sapien_urdf_obj(
                scene=self,
                pose=pose_microwave,
                modelname="044_microwave",
                scale=final_scale,
                modelid=0,
                fix_root_link=True,
            )
        except Exception as e:
            print(f"[Kitchen_base_large] failed to load microwave URDF: {e}")
            microwave_actor = None

        if microwave_actor is not None:
            self.microwave_left = microwave_actor
            # Ensure cached config scaling matches the physical scaling we applied.
            if isinstance(self.microwave_left.config, dict):
                self.microwave_left.config["scale"] = float(final_scale)
            self.microwave_left.set_name("microwave_center")
            self.add_prohibit_area(self.microwave_left, padding=[0.2, 0.05], area="table")
            self.collision_list.append({
                "actor": self.microwave_left,
                "collision_path": f"{os.environ['ASSETS_ROOT']}/objects/044_microwave/visual/base0.glb",
            })
    def _load_basket_on_table(self, table_height: float, table_xy_bias):
        """Place the static table-side container (`063_tabletrashbin`) on the left front edge of the table."""
        jx = float(np.random.uniform(self.basket_right_position_jitter_x[0], self.basket_right_position_jitter_x[1]))
        jy = float(np.random.uniform(self.basket_right_position_jitter_y[0], self.basket_right_position_jitter_y[1]))
        x_right, y_front, z_basket = self._get_scene_obj_locations(object_name="basket")
        if self.jitter_basket:
            y_front += jy
            x_right += jx
        z_basket = table_height + 0.02 if z_basket == 0 else z_basket

        br_roll_deg, br_pitch_deg, br_yaw_deg = self.basket_right_rot
        br_ax = math.radians(br_roll_deg)
        br_ay = math.radians(br_pitch_deg)
        br_az = math.radians(br_yaw_deg)
        bqx, bqy, bqz, bqw = t3d.euler.euler2quat(br_ax, br_ay, br_az)
        basket_quat = [bqw, bqx, bqy, bqz]

        pose_basket = sapien.Pose([x_right, y_front, z_basket], basket_quat)
        modelname = str(self.basket_right_modelname)
        model_id = int(self.basket_right_model_id)
        intrinsic_scale = self._get_asset_model_scale_create_actor(modelname=modelname, model_id=model_id)
        final_scale = intrinsic_scale * float(self.basket_right_scale)
        basket_actor = create_actor(
            scene=self.scene,
            pose=pose_basket,
            modelname=modelname,
            # `create_actor.py` treats `scale=` as absolute, so multiply
            # by intrinsic model_data scale to keep historical multiplier semantics.
            scale=final_scale,
            is_static=True,
            convex=False,
            model_id=model_id,
        )
        if basket_actor is not None:
            self.basket_right = basket_actor
            # Ensure cached config scaling matches the physical scaling we applied.
            if isinstance(self.basket_right.config, dict):
                # For create_actor, model_data["scale"] is usually [sx, sy, sz].
                self.basket_right.config["scale"] = [float(final_scale)] * 3
            self.basket_right.set_name("basket_right")
            self.add_prohibit_area(self.basket_right, padding=0.05, area="table")
    def add_collision(self, objects=("basket")):
        print_c("Furniture collisions added","YELLOW")
        if "basket" in objects:
            self.collision_list.append({
                "actor": self.basket_right,
                "collision_path": f"{os.environ['ASSETS_ROOT']}/objects/063_tabletrashbin/collision/base6.glb",
            })
        if "fridge" in objects:
            self.collision_list.append({
                    "actor": self.fridge_left,
                    "collision_path": f"{os.environ['ASSETS_ROOT']}/objects/124_fridge_hivvdf/blender_public/links/",
                    "pose": self.fridge_left.get_link_pose("base_link"), 
                    "files": ["base_link_collision.glb", "link_0_collision.glb"],
                })
        if "cabinet" in objects:
            self.collision_list.append({
                    "actor": self.cabinet,
                    "collision_path": f"{os.environ['ASSETS_ROOT']}/objects/125_cabinet_tynnnw/blender_public/links/",
                    "pose": self.cabinet.get_link_pose("base_link"),
                    "files": ["base_link_collision.glb", "left_door_collision.glb", "right_door_collision.glb"],
                })
    def _load_cabinet_on_table(self, table_height: float, table_xy_bias):
        """Place the chosen articulated cabinet asset on the opposite end of the table from the drawer."""
        # Mirror the drawer position across the table center in x.
        x_center = table_xy_bias[0]
        y_center = table_xy_bias[1] + 0.12
        z_center = table_height + 0.7

        # Apply 90° rotation around all three axes (roll, pitch, yaw).
        ax = math.radians(90)  # roll (x)
        ay = math.radians(180)  # pitch (y)
        az = math.radians(-90)  # yaw (z)
        # transforms3d.euler.euler2quat expects (ax, ay, az) in radians (x, y, z axes) and returns (qx, qy, qz, qw).
        qx, qy, qz, qw = t3d.euler.euler2quat(ax, ay, az)
        base_pose = sapien.Pose([x_center, y_center, z_center], [qw, qx, qy, qz])
        # Use the generic bench cabinet loader.
        self.cabinet = self._create_bench_cabinet(
            asset_dir_name="125_cabinet_tynnnw",
            pose=base_pose,
            fix_root_link=True,
            extra_scale=self.cabinet_scale,
        )
        if self.cabinet is not None:
            self.cabinet.set_name("cabinet")
            self.add_prohibit_area(self.cabinet, padding=0.02, area="cabinet")
            self._init_cabinet_states()
    def _entity_aabb(self, entity):
        # Actor path: reuse existing utility.
        if hasattr(entity, "get_components"):
            return get_actor_boundingbox(entity)

        # Articulation path: aggregate all link collision-shape vertices.
        if not hasattr(entity, "get_links"):
            return None, None

        all_points = []
        for link in entity.get_links():
            try:
                link_pose = link.pose
            except Exception:
                link_pose = link.get_pose()
            link_mat = link_pose.to_transformation_matrix()

            try:
                shapes = link.get_collision_shapes()
            except Exception:
                shapes = []

            for shape in list(shapes):
                try:
                    local_v = np.array(shape.get_vertices(), dtype=float)
                except Exception:
                    try:
                        hs = np.array(shape.half_size, dtype=float)
                    except Exception:
                        continue
                    local_v = np.array(
                        [[x, y, z] for x in (-hs[0], hs[0]) for y in (-hs[1], hs[1]) for z in (-hs[2], hs[2])],
                        dtype=float,
                    )

                try:
                    local_v = local_v * np.array(shape.scale, dtype=float)
                except Exception:
                    pass

                try:
                    shape_mat = shape.get_local_pose().to_transformation_matrix()
                except Exception:
                    shape_mat = np.eye(4, dtype=float)

                world_mat = link_mat @ shape_mat
                homo_v = np.pad(local_v, ((0, 0), (0, 1)), constant_values=1.0)
                world_v = (world_mat @ homo_v.T).T[:, :3]
                all_points.append(world_v)

        if not all_points:
            return None, None
        points_cloud = np.vstack(all_points)
        return points_cloud.min(axis=0), points_cloud.max(axis=0)

    def _add_cabinet_wall_filler(self):
        """
        Fill space behind cabinet with a simple gray static box.
        """
        if getattr(self, "cabinet", None) is None:
            return

        cab_pose = np.array(self.cabinet.get_pose().p, dtype=float)
        scale_ratio = float(self.cabinet_scale) / 0.5

        # User-requested simple placement: smaller dimensions + positive y offset.
        x_half = 0.30 * scale_ratio
        y_half = 0.11 * scale_ratio
        z_half = 0.22 * scale_ratio

        x_center = float(cab_pose[0])
        y_center = float(cab_pose[1] + 0.16 * scale_ratio)
        z_center = float(cab_pose[2])

        self.cabinet_wall_filler = create_box(
            self.scene,
            sapien.Pose(p=[x_center, y_center, z_center]),
            half_size=[x_half, y_half, z_half],
            color=(0.32, 0.32, 0.32),
            name="cabinet_wall_filler",
            is_static=True,
        )

    # -----------------------
    # Cabinet articulation state helpers
    # -----------------------
    def _get_cabinet_right_joint_index(self) -> int:
        """
        Return the right-door joint index for the cabinet articulation.

        Convention for this asset: joint 0 is left door, joint 1 is right door.
        Fall back to the last DOF if a different articulation shape is loaded.
        """
        if not hasattr(self, "cabinet") or self.cabinet is None:
            return -1
        try:
            qpos = np.array(self.cabinet.get_qpos(), dtype=float)
        except Exception:
            return -1
        if qpos.shape[0] <= 0:
            return -1
        if qpos.shape[0] >= 2:
            return 1
        return qpos.shape[0] - 1

    def _init_cabinet_states(self):
        """Initialize canonical closed/open states for right-door-only cabinet control."""
        if not hasattr(self, "cabinet") or self.cabinet is None:
            self.cabinet_closed_qpos = None
            self.cabinet_open_qpos = None
            self.cabinet_right_joint_idx = -1
            return

        qpos = np.array(self.cabinet.get_qpos(), dtype=float)
        self.cabinet_right_joint_idx = self._get_cabinet_right_joint_index()
        self.cabinet_closed_qpos = qpos.copy()

        try:
            qlimits = np.array(self.cabinet.get_qlimits(), dtype=float)
        except Exception:
            qlimits = None

        open_qpos = qpos.copy()
        idx = self.cabinet_right_joint_idx
        if (
            idx >= 0
            and qlimits is not None
            and qlimits.shape[0] > idx
        ):
            low, high = qlimits[idx]
            if np.isfinite(low) and np.isfinite(high) and high > low:
                open_qpos[idx] = high
        self.cabinet_open_qpos = open_qpos

    def set_cabinet_closed(self):
        """Reset the cabinet right door to its canonical closed configuration."""
        if getattr(self, "cabinet_closed_qpos", None) is None:
            return
        if not hasattr(self, "cabinet") or self.cabinet is None:
            return
        self.cabinet.set_qpos(np.array(self.cabinet_closed_qpos, dtype=float))

    def set_cabinet_open(self):
        """Set the cabinet right door to its canonical fully-open configuration."""
        if getattr(self, "cabinet_open_qpos", None) is None:
            return
        if not hasattr(self, "cabinet") or self.cabinet is None:
            return
        self.cabinet.set_qpos(np.array(self.cabinet_open_qpos, dtype=float))

    def is_cabinet_open(self, threshold: float = 0.02) -> bool:
        """Return True if the right cabinet door is open beyond threshold."""
        if not hasattr(self, "cabinet") or self.cabinet is None:
            return False
        if getattr(self, "cabinet_closed_qpos", None) is None:
            return False
        idx = getattr(self, "cabinet_right_joint_idx", -1)
        if idx < 0:
            return False
        current = np.array(self.cabinet.get_qpos(), dtype=float)
        closed = np.array(self.cabinet_closed_qpos, dtype=float)
        if current.shape != closed.shape or current.shape[0] <= idx:
            return False
        return abs(float(current[idx] - closed[idx])) > float(threshold)

    def is_cabinet_closed(self, threshold: float = 0.02) -> bool:
        """Return True if the right cabinet door is effectively closed."""
        if not hasattr(self, "cabinet") or self.cabinet is None:
            return False
        if getattr(self, "cabinet_closed_qpos", None) is None:
            return False
        idx = getattr(self, "cabinet_right_joint_idx", -1)
        if idx < 0:
            return False
        current = np.array(self.cabinet.get_qpos(), dtype=float)
        closed = np.array(self.cabinet_closed_qpos, dtype=float)
        if current.shape != closed.shape or current.shape[0] <= idx:
            return False
        return abs(float(current[idx] - closed[idx])) <= float(threshold)

    # -----------------------
    # Fridge articulation state helpers
    # -----------------------
    def _init_fridge_states(self):
        """Initialize canonical closed and open configurations for the fridge articulation."""
        if not hasattr(self, "fridge_left") or self.fridge_left is None:
            self.fridge_closed_qpos = None
            self.fridge_open_qpos = None
            return

        # Closed configuration = current qpos
        qpos = np.array(self.fridge_left.get_qpos(), dtype=float)
        self.fridge_closed_qpos = qpos.copy()

        # Use articulation qlimits to define an "open" configuration:
        # move each finite-interval joint to its upper limit so the door is open.
        qlimits = np.array(self.fridge_left.get_qlimits(), dtype=float)  # shape (dof, 2)
        open_qpos = qpos.copy()

        if len(open_qpos) > 0 and qlimits.shape[0] >= open_qpos.shape[0]:
            for i in range(open_qpos.shape[0]):
                low, high = qlimits[i]
                if np.isfinite(low) and np.isfinite(high) and high > low:
                    open_qpos[i] = high

        self.fridge_open_qpos = open_qpos

    def set_fridge_closed(self):
        """Reset the fridge to its canonical closed configuration."""
        if getattr(self, "fridge_closed_qpos", None) is None:
            return
        if not hasattr(self, "fridge_left") or self.fridge_left is None:
            return
        self.fridge_left.set_qpos(self.fridge_closed_qpos)

    def set_fridge_open(self):
        """Set the fridge to its canonical open configuration."""
        if getattr(self, "fridge_open_qpos", None) is None:
            return
        if not hasattr(self, "fridge_left") or self.fridge_left is None:
            return
        self.fridge_left.set_qpos(self.fridge_open_qpos)

    def set_fridge_open_angle_deg(self, angle_deg: float, open_span_deg: float = 90.0) -> None:
        """
        Set the fridge door to a target opening angle.

        This linearly interpolates articulation `qpos` between the canonical
        closed state (`fridge_closed_qpos`) and the canonical fully-open state
        (`fridge_open_qpos`), assuming the fully-open pose corresponds to
        `open_span_deg` degrees of rotation.
        """
        if getattr(self, "fridge_closed_qpos", None) is None:
            return
        if getattr(self, "fridge_open_qpos", None) is None:
            return
        if not hasattr(self, "fridge_left") or self.fridge_left is None:
            return
        if open_span_deg <= 0:
            return

        angle_deg = float(angle_deg)
        open_span_deg = float(open_span_deg)
        angle_deg = max(0.0, min(angle_deg, open_span_deg))
        ratio = angle_deg / open_span_deg

        closed = np.asarray(self.fridge_closed_qpos, dtype=float)
        open_qpos = np.asarray(self.fridge_open_qpos, dtype=float)

        if closed.shape != open_qpos.shape:
            # Shape mismatch shouldn't happen, but keep a safe fallback.
            self.fridge_left.set_qpos(open_qpos if ratio >= 0.5 else closed)
            return

        delta = open_qpos - closed
        # Only interpolate joints that actually move between closed/open.
        movable = np.abs(delta) > 1e-6
        new_qpos = closed.copy()
        new_qpos[movable] = closed[movable] + ratio * delta[movable]
        self.fridge_left.set_qpos(new_qpos)

    def set_fridge_open_random_angle_between(
        self,
        min_angle_deg: float = 45.0,
        max_angle_deg: float = 90.0,
        open_span_deg: float = 90.0,
    ) -> float:
        """Randomly set fridge door angle between `min_angle_deg` and `max_angle_deg`."""
        min_angle_deg = float(min_angle_deg)
        max_angle_deg = float(max_angle_deg)
        if max_angle_deg < min_angle_deg:
            min_angle_deg, max_angle_deg = max_angle_deg, min_angle_deg

        angle_deg = float(np.random.uniform(min_angle_deg, max_angle_deg))
        self.set_fridge_open_angle_deg(angle_deg, open_span_deg=open_span_deg)
        return angle_deg

    def is_fridge_open(self, threshold: float = 0.01) -> bool:
        """Return True if the fridge has moved significantly away from the closed configuration."""
        if getattr(self, "fridge_closed_qpos", None) is None:
            return False
        if not hasattr(self, "fridge_left") or self.fridge_left is None:
            return False

        current = np.array(self.fridge_left.get_qpos(), dtype=float)
        diff = np.abs(current - self.fridge_closed_qpos)
        return np.max(diff) > threshold

    def is_fridge_closed(self, threshold: float = 0.01) -> bool:
        """Return True if the fridge is effectively in the closed configuration."""
        if getattr(self, "fridge_closed_qpos", None) is None:
            return False
        if not hasattr(self, "fridge_left") or self.fridge_left is None:
            return False
        current = np.array(self.fridge_left.get_qpos(), dtype=float)
        diff = np.abs(current - self.fridge_closed_qpos)
        return float(np.max(diff)) <= float(threshold)

    def is_fridge_fully_open(self, threshold: float = 0.01) -> bool:
        """
        Return True if the fridge is effectively in the canonical "fully open"
        configuration (computed from articulation qlimits).
        """
        if getattr(self, "fridge_open_qpos", None) is None:
            return False
        if not hasattr(self, "fridge_left") or self.fridge_left is None:
            return False
        current = np.array(self.fridge_left.get_qpos(), dtype=float)
        diff = np.abs(current - self.fridge_open_qpos)
        return float(np.max(diff)) <= float(threshold)

    def _create_bench_cabinet(
        self,
        asset_dir_name: str,
        pose: sapien.Pose,
        fix_root_link: bool = True,
        extra_scale: float = 1.0,
    ) -> ArticulationActor | None:
        """
        Generic helper to load an articulated cabinet defined under assets/objects/<asset_dir_name>
        as an ArticulationActor, using its local model_data.json for scale and transform.
        """
        modeldir = Path(os.environ["ASSETS_ROOT"]) / "objects" / asset_dir_name
        # Prefer mobility.urdf if present, otherwise fall back to a SAPIEN-exported URDF name.
        urdf_path = modeldir / "mobility.urdf"
        if not urdf_path.exists():
            # Common pattern for exported URDFs (e.g., sapien_urdf/nkrgez.urdf)
            sapien_urdf_dir = modeldir / "sapien_urdf"
            if sapien_urdf_dir.is_dir():
                # Pick the first .urdf file in sapien_urdf
                urdfs = list(sapien_urdf_dir.glob("*.urdf"))
                if urdfs:
                    urdf_path = urdfs[0]

        json_file = modeldir / "model_data.json"

        if not urdf_path.exists():
            print(f"[Kitchen_base_large] cabinet URDF not found: {urdf_path}")
            return None

        # If this URDF has an accompanying SRDF, use it to disable internal
        # collisions between specified link pairs (e.g., fridge door vs frame).
        srdf_path: Path | None = None
        candidate_srdf = urdf_path.with_suffix(".srdf")
        alt_candidate_srdf = None
        if candidate_srdf.exists():
            srdf_path = candidate_srdf
        else:
            # Fallback: <stem>.srdf in the same folder.
            alt_candidate_srdf = urdf_path.parent / f"{urdf_path.stem}.srdf"
            if alt_candidate_srdf.exists():
                srdf_path = alt_candidate_srdf

        if json_file.exists():
            with open(json_file, "r", encoding="utf-8") as file:
                model_data = json.load(file)
            raw_scale = model_data.get("scale", 1.0)
            extra_scale_f = float(extra_scale)

            # Keep scaling semantics consistent with the rest of the codebase:
            # - `loader.scale` drives the *physical* scaling of URDF geometry
            # - `model_data["scale"]` drives how we scale cached points/extents
            #   (e.g., via `add_prohibit_area`)
            raw_scale_vec = np.array(raw_scale, dtype=float).reshape(-1)
            if raw_scale_vec.size == 1:
                raw_scale_vec = np.array([raw_scale_vec[0], raw_scale_vec[0], raw_scale_vec[0]], dtype=float)
            elif raw_scale_vec.size >= 3:
                raw_scale_vec = raw_scale_vec[:3]
            else:
                raw_scale_vec = np.array([1.0, 1.0, 1.0], dtype=float)

            scaled_scale_vec = raw_scale_vec * extra_scale_f
            model_data["scale"] = scaled_scale_vec.tolist()

            # URDFLoader expects a uniform scalar scale.
            scale_scalar = float(scaled_scale_vec[0])
            trans_mat = np.array(model_data.get("transform_matrix", np.eye(4)))
        else:
            # Provide a minimal config so downstream code (e.g. add_prohibit_area) always
            # sees a dict instead of None.
            model_data = {"scale": [1.0, 1.0, 1.0]}
            scale_scalar = float(extra_scale)
            trans_mat = np.eye(4)

        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        # URDFLoader expects a uniform scalar scale.
        loader.scale = float(scale_scalar)
        loader.fix_root_link = fix_root_link
        loader.load_multiple_collisions_from_file = True

        try:
            articulation = loader.load_multiple(str(urdf_path))[0][0]

            if srdf_path is not None:
                self.apply_srdf_collisions(articulation, srdf_path)
        except Exception as e:
            print(f"[Kitchen_base_large] failed to load cabinet URDF {urdf_path}: {e}")
            return None

        pose_mat = pose.to_transformation_matrix()
        pose_with_offset = sapien.Pose(
            p=pose_mat[:3, 3] + trans_mat[:3, 3],
            q=t3d.quaternions.mat2quat(trans_mat[:3, :3] @ pose_mat[:3, :3]),
        )
        articulation.set_pose(pose_with_offset)

        if model_data is not None:
            init_qpos = model_data.get("init_qpos")
            if init_qpos is not None and len(init_qpos) > 0:
                articulation.set_qpos(np.array(init_qpos, dtype=float))

        # Match drive properties to other articulated objects
        for joint in articulation.get_joints():
            # Keep stiffness at 0 (position is handled via qpos), but avoid excessive
            # damping that can cause jitter/instability for articulated doors.
            joint.set_drive_properties(damping=10.0, stiffness=0)

        articulation.set_name(asset_dir_name)
        return ArticulationActor(articulation, model_data, scale=model_data.get("scale"))

    # Drawer-related APIs are deprecated and no longer used now that the
    # base kitchen does not spawn any drawer units. They are retained only
    # so older code that imports them does not break.
    def _init_drawer_states(self):
        self.drawer_closed_qpos = None
        self.drawer_open_qpos = None

    def set_drawer_closed(self):
        """Deprecated: drawers are no longer part of the base kitchen."""
        return

    def set_drawer_open(self):
        """Deprecated: drawers are no longer part of the base kitchen."""
        return

    def is_drawer_open(self, threshold: float = 0.01) -> bool:
        """Deprecated: always returns False because no drawers exist."""
        return False
    
    def _load_kitchen_appliances(self, table_height: float):
        # Fridge on the left side of the counter
        if "fridge" in self.kitchen_appliance_assets:
            cfg = self.kitchen_appliance_assets["fridge"]
            fridge_id = cfg["default_modelid"]
            self.fridge = rand_create_sapien_urdf_obj(
                scene=self.scene,
                modelname=cfg["modelname"],
                modelid=fridge_id,
                xlim=[-0.5, -0.5],
                ylim=[0.2, 0.2],
                rotate_rand=False,
                rotate_lim=[0, 0, 0],
                qpos=[1, 0, 0, 1],
                fix_root_link=True,
            )
            if self.fridge is not None:
                self.fridge.set_name("fridge")
                self.add_prohibit_area(self.fridge, padding=0.01, area="fridge")

        # Wall cabinet on the right, slightly above the counter
        if "cabinet" in self.kitchen_appliance_assets:
            cfg = self.kitchen_appliance_assets["cabinet"]
            cabinet_id = cfg["default_modelid"]
            self.cabinet = rand_create_sapien_urdf_obj(
                scene=self.scene,
                modelname=cfg["modelname"],
                modelid=cabinet_id,
                xlim=[0.55, 0.55],
                ylim=[0.2, 0.2],
                rotate_rand=False,
                rotate_lim=[0, 0, 0],
                qpos=[1, 0, 0, 1],
                fix_root_link=True,
            )
            if self.cabinet is not None:
                self.cabinet.set_name("cabinet")
                self.add_prohibit_area(self.cabinet, padding=0.02, area="cabinet")

        # Legacy "drawer" appliance has been removed from the base kitchen layout.

    def _get_available_model_ids(self, modelname: str):
        asset_path = os.path.join(os.environ["BENCH_ROOT"], "assets/objects", modelname)
        json_files = glob.glob(os.path.join(asset_path, "model_data*.json"))
        available_ids = []
        for file in json_files:
            base = os.path.basename(file)
            try:
                idx = int(base.replace("model_data", "").replace(".json", ""))
                available_ids.append(idx)
            except ValueError:
                continue
        return available_ids

    def _sample_model_id(self, modelname: str, fallback: int = 0) -> int:
        ids = self._get_available_model_ids(modelname)
        if not ids:
            return fallback
        return int(np.random.choice(ids))

    def get_cluttered_surfaces(self):
        # clutter surfaces with additional random obstacles
        # table ------------------------------------------------------
        table_bb = get_actor_boundingbox(self.table)
        xlim = [table_bb[0][0], table_bb[1][0]]
        ylim = [table_bb[0][1], table_bb[1][1]]
        zlim = [table_bb[1][2]]
        xlim[0] += self.table_xy_bias[0]
        xlim[1] += self.table_xy_bias[0]
        ylim[0] += self.table_xy_bias[1]
        ylim[1] += self.table_xy_bias[1]
        zlim = np.array(zlim) + self.table_z_bias
        
        # collect objects already in the scene
        task_objects_list = []
        # print_c("articulations in the scene: ", "YELLOW")

        # print([o.get_name() for o in self.scene.get_all_articulations()])
        for entity in self.scene.get_all_actors()+ self.scene.get_all_articulations():
            actor_name = entity.get_name()
            if actor_name == "":
                continue
            if actor_name in self.FURNITURE_NAMES:
                continue
            task_objects_list.append(actor_name)
        # print_c(f"Existing objects in the scene: {task_objects_list}", "YELLOW")
        cluttered_item_info, obj_names_short, obj_names_tall = get_obstacle_objects_subset(
            "kitchenl", self.obstacle_distribution, task_objects_list
        )
        self.clutter_surface_split(xlim, ylim, zlim, self.prohibited_area["table"], self.obstacle_density, cluttered_item_info, obj_names_short, obj_names_tall)

    def add_extra_cameras(self):
        self.cameras.add_extra_cameras(f"{os.environ['ASSETS_ROOT']}/embodiments/kitchen_l_config.yml")