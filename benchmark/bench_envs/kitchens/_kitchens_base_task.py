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
from transforms3d.euler import euler2quat
from collections import OrderedDict
import torch, random

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


class KitchenS_base_task(Bench_base_task):

    FURNITURE_NAMES = {
        "table", "wall", "ground",
        # sink basin (has box collision) + chrome rim pieces (create_box, static)
        "sink",
        "sink_rim_front", "sink_rim_back", "sink_rim_left", "sink_rim_right",
        # dish rack body (non-convex collision mesh = exact rack geometry)
        "dishrack",
    }

    def __init__(self):
        pass

    def _init_task_env_(self, table_xy_bias=[0, 0], table_height_bias=0, **kwags):
        super().__init__()
        ta.setup_logging("CRITICAL")

        loading_eval_env = self._bind_eval_env(kwags)
        if not loading_eval_env:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            random.seed(self.seed)
        print_c(f"#### Seed value {self.seed} ####", "YELLOW")

        # A task may restrict which KitchenS scenes it runs in by setting the
        # class attribute `allowed_scene_ids` (e.g. microwave-cavity tasks skip
        # scene 2, where the microwave sits center and the cavity is out of the
        # arm's reach). Default behaviour is unchanged (uniform over 0,1,2).
        if loading_eval_env and self._eval_scene_spec.get("scene_id") is not None:
            self.scene_id = int(self._eval_scene_spec["scene_id"])
        elif kwags.get("scene_id") is not None:
            self.scene_id = kwags.get("scene_id")
        else:
            _allowed = getattr(self, "allowed_scene_ids", None)
            if _allowed:
                self.scene_id = int(np.random.choice(list(_allowed)))
            else:
                self.scene_id = np.random.randint(0, 3)
        print_c(f"KitchenS scene {self.scene_id} selected", "YELLOW")
        
        self.FRAME_IDX = 0
        self.task_name = kwags.get("task_name")
        self.save_dir = kwags.get("save_path", "data")
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

        self.need_topp = True

        # Random
        random_setting = kwags.get("domain_randomization")
        self.random_background = random_setting.get("random_background", False)
        self.cluttered_table = random_setting.get("cluttered_table", False)
        self.clean_background_rate = random_setting.get("clean_background_rate", 1)
        self.random_head_camera_dis = random_setting.get("random_head_camera_dis", 0)
        self.random_table_height = random_setting.get("random_table_height", 0)
        self.random_light = random_setting.get("random_light", False)
        self.crazy_random_light_rate = random_setting.get("crazy_random_light_rate", 0)
        if loading_eval_env:
            self.crazy_random_light = bool(self._eval_scene_spec.get("crazy_random_light", False))
        else:
            self.crazy_random_light = (0 if not self.random_light else np.random.rand() < self.crazy_random_light_rate)
        self.random_embodiment = random_setting.get("random_embodiment", False)
        self.obstacle_height = random_setting.get("obstacle_height", "short")
        self.obstacle_density = random_setting.get("obstacle_density", 3)

        self._parse_perturbations(random_setting)

        self.file_path = []
        self.plan_success = True
        self.step_lim = None
        self.fix_gripper = False
        self.setup_scene(**kwags)

        self.left_js = None
        self.right_js = None
        self.raw_head_pcl = None
        self.real_head_pcl = None
        self.real_head_pcl_color = None

        self.now_obs = {}
        self.take_action_cnt = 0
        self.eval_video_path = kwags.get("eval_video_save_dir", None)
        self.incl_collision = kwags.get("include_collision", False)

        self.save_freq = kwags.get("save_freq")
        self.video_fps = kwags.get("video_fps", 30)
        self.world_pcd = None

        self.key_objects = []
        self.size_dict = list()
        self.cluttered_objs = list()
        self.prohibited_area = {"table": []}
        self.record_cluttered_objects = list()
        self.cluttered_objects_info = get_cluttered_objects_info()

        self.eval_success = False
        if loading_eval_env:
            self.table_z_bias = float(self._eval_scene_spec.get("table_z_bias", 0.0))
        else:
            self.table_z_bias = 0



        self.kitchens_info = {
            "table_height": 0.74,
            "table_area": [1.2, 0.7],
            "table_lims": [],
        }

        self.item_info = get_task_objects_config()
        self.target_objects_info = get_target_objects_subset("kitchens", self.sample_d)
        if getattr(self, "unseen_targets", False):
            self._merge_ood_target_info("kitchens")

        self.instruction = None
        self.collision_list = []
        self.cuboid_collision_list = []
        self._init_collision_metrics()

        self.need_plan = kwags.get("need_plan", True)
        self.left_joint_path = kwags.get("left_joint_path", [])
        self.right_joint_path = kwags.get("right_joint_path", [])
        self.left_cnt = 0
        self.right_cnt = 0

        self.load_robot(**kwags)
        self.create_static_elements(table_xy_bias=table_xy_bias)
        self.load_camera(**kwags)
        if loading_eval_env:
            self.apply_camera_spec(self._eval_scene_spec.get("cameras") or [])
        self.robot.move_to_homestate()

        render_freq = self.render_freq
        self.render_freq = 0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

        self.robot.set_origin_endpose()
        if loading_eval_env:
            self.load_actors_from_spec(self._eval_task_spec, self._eval_init_state)
        else:
            self.load_actors()
        self.add_gripper_operating_area()

        if loading_eval_env:
            self.spawn_clutter_from_spec(self._eval_scene_spec.get("clutter") or [])
        elif self.cluttered_table:
            self.get_cluttered_surfaces()

        self._apply_specular_ood()
        self._furniture_texture_targets = [
            ("microwave", "microwave"),
            ("sink", "sink"),
        ]
        self._apply_furniture_texture_ood()

        if self.enable_collision_metrics:
            self._build_collision_name_sets()

        # per-step proximity (nearest-obstacle clearance) tracking
        if kwags.get("data_type", {}).get("proximity", True):
            self._init_proximity_tracking(kwags.get("proximity_tracking", {}))

        if loading_eval_env:
            self._finish_eval_env_restore()
        else:
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

        self.info = dict()
        self.info["cluttered_table_info"] = self.record_cluttered_objects
        self.info["texture_info"] = {
            "wall_texture": self.wall_texture,
            "table_texture": self.table_texture,
            "floor_texture": self.floor_texture,
        }
        self.info["info"] = {}

        self.stage_success_tag = False

    # ------------------------------------------------------------------
    # Scene layout helpers
    # ------------------------------------------------------------------

    def _get_scene_obj_locations(self, object_name="microwave"):
        """Return [x, y] positions for kitchen-S fixtures based on scene_id.

        scene_0: MW left, Dishrack center, Sink right
        scene_1: MW left, Sink center, Dishrack right
        scene_2: MW center, Dishrack left, Sink right
        """
        # Dishrack y was originally 0.25 but the aloha arm IK envelope cannot
        # reach the elevated rack (z ≈ 0.96) at that depth. Pulled forward to
        # 0.18 so the plate target (rack_y - 0.09 = 0.09) is at the same
        # comfortable depth as the sink basin (y = 0.08).
        if self.scene_id == 0:
            locations = {
                "microwave": [-0.32, 0.18],
                "dishrack": [0.05, 0.17],
                "sink": [0.42, 0.08],
            }
        elif self.scene_id == 1:
            locations = {
                "microwave": [-0.32, 0.18],
                "dishrack": [0.42, 0.17],
                "sink": [0.10, 0.08],
            }
        elif self.scene_id == 2:
            locations = {
                "microwave": [0.10, 0.18],
                "dishrack": [-0.32, 0.17],
                "sink": [0.42, 0.08],
            }
        else:
            raise ValueError(f"Invalid scene_id {self.scene_id}")

        if object_name not in locations:
            raise ValueError(f"Unknown object_name '{object_name}', expected one of {list(locations.keys())}")
        return locations[object_name]

    # ------------------------------------------------------------------
    # Spawn helper with prohibited-area rejection
    # ------------------------------------------------------------------

    def rand_pose_on_counter(
        self,
        xlim,
        ylim,
        zlim=None,
        qpos=(1, 0, 0, 0),
        rotate_rand=False,
        rotate_lim=(0, 0, 0),
        ylim_prop=False,
        obj_padding=0.02,
        attempts=80,
        allow_sink=False,
        allow_microwave_front=False,
    ):
        """Sample a pose in ``xlim × ylim`` that avoids every box already in
        ``self.prohibited_area["table"]``. If the requested range has no clear
        pose, fail the seed instead of silently returning a prohibited pose.

        ``obj_padding`` is the half-extent used to treat the sampled point as
        a footprint (so the footprint, not just the center, must clear the
        existing prohibited boxes).

        ``allow_sink`` skips the basin keep-out (static boards/bins may sit
        over the middle sink if they rest stably). Microwave, rack, and
        other object boxes still apply.

        ``allow_microwave_front`` skips only the extra door-swing keepout
        (``microwave_front_keepout``). Use it for objects that leave that
        zone before the door closes — the microwave body AABB still applies.
        """
        from envs.utils import rand_pose

        if zlim is None:
            zlim = [self.kitchens_info["table_height"] + self.table_z_bias + 0.001]

        sink_xy = None
        if allow_sink and getattr(self, "sink", None) is not None:
            sp = self.sink.get_pose().p
            sink_xy = (float(sp[0]), float(sp[1]))
        mw_front = getattr(self, "_microwave_front_keepout_box", None)

        pose = None
        for _ in range(attempts):
            pose = rand_pose(
                xlim=xlim,
                ylim=ylim,
                zlim=zlim,
                qpos=list(qpos),
                rotate_rand=rotate_rand,
                rotate_lim=list(rotate_lim),
                ylim_prop=ylim_prop,
            )
            x, y = float(pose.p[0]), float(pose.p[1])
            fx0, fx1 = x - obj_padding, x + obj_padding
            fy0, fy1 = y - obj_padding, y + obj_padding
            blocked = False
            for box in self.prohibited_area.get("table", []):
                if allow_microwave_front and mw_front is not None and box is mw_front:
                    continue
                x0, y0, x1, y1 = box
                if sink_xy is not None and x0 <= sink_xy[0] <= x1 and y0 <= sink_xy[1] <= y1:
                    continue
                if fx1 >= x0 and fx0 <= x1 and fy1 >= y0 and fy0 <= y1:
                    blocked = True
                    break
            if not blocked:
                return pose
        raise RuntimeError(
            f"[KitchenS] rand_pose_on_counter found no valid pose after {attempts} "
            f"attempts in xlim={list(xlim)}, ylim={list(ylim)}, "
            f"obj_padding={obj_padding}; refusing prohibited fallback at "
            f"({pose.p[0]:.3f}, {pose.p[1]:.3f})"
        )

    # ------------------------------------------------------------------
    # Static scene construction
    # ------------------------------------------------------------------

    def create_static_elements(self, table_xy_bias=[0, 0]):
        self.table_xy_bias = table_xy_bias
        table_height = self.kitchens_info["table_height"] + self.table_z_bias

        # Textures -------------------------------------------------------
        if self._apply_eval_env_textures():
            pass
        elif self.random_background:
            texture_type = "seen" if not self.eval_mode else "unseen"
            directory_path = f"{os.environ['ASSETS_ROOT']}/background_texture/{texture_type}"
            file_count = len(
                [name for name in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, name))])

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

        # Floor ----------------------------------------------------------
        self.floor_parts = []
        positions = [
            [1, 1, 0],
            [-1, 1, 0],
            [1, -1, 0],
            [-1, -1, 0],
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

        # Wall -----------------------------------------------------------
        self.wall = create_box(
            self.scene,
            sapien.Pose(p=[0, 1, 1.5]),
            half_size=[3, 0.6, 1.5],
            color=(1, 0.9, 0.9),
            name="wall",
            texture_id=self.wall_texture,
            is_static=True,
        )

        # Countertop geometry (shared with fixture loaders) --------------
        counter_length = self.kitchens_info["table_area"][0]
        counter_width = self.kitchens_info["table_area"][1]
        counter_thickness = 0.04
        self.kitchens_info["counter_thickness"] = counter_thickness

        # Sink position & hole dims — computed here so the counter can be
        # built with a matching through-hole around the basin.
        sink_rel_x, sink_rel_y = self._get_scene_obj_locations("sink")
        self.kitchens_info["sink_geom"] = {
            "rel_p": [sink_rel_x, sink_rel_y],
            "hole_hx": 0.13,
            "hole_hy": 0.20,
            "depth": 0.09,
            "inner_hx": 0.12,
            "inner_hy": 0.19,
        }

        # Backsplash (visual, behind the counter) ------------------------
        self._create_backsplash(counter_length, counter_width, table_height, table_xy_bias)

        # Countertop — single static actor composed of 4 pieces around
        # the sink opening so objects can fall into the basin.
        self._create_counter_with_sink_hole(
            counter_length, counter_width, counter_thickness,
            table_height, table_xy_bias,
        )

        # Base cabinets below the counter (visual only) ------------------
        self._create_base_cabinets(counter_length, counter_width, table_height, counter_thickness, table_xy_bias)

        # Thin dark trim along the front edge of the counter ------------
        self._create_counter_edge_trim(counter_length, counter_width, table_height, counter_thickness, table_xy_bias)

        # Upper decorative open shelves (left & right) -------------------
        self._create_upper_shelves(table_height, table_xy_bias)

        self.kitchens_info["table_lims"] = [
            -counter_length / 2, -counter_width / 2,
            counter_length / 2, counter_width / 2,
        ]
        self.cuboid_collision_list.append({
            "name": "table",
            "dims": [counter_length, counter_width, 0.002],
            "pose": [0, 0, table_height, 1, 0, 0, 0],
        })

        # Kitchen-S fixtures ---------------------------------------------
        self._load_microwave(table_height, table_xy_bias)
        self._load_dishrack(table_height, table_xy_bias)
        self._load_sink(table_height, table_xy_bias)

        if self.incl_collision:
            self.add_collision()

    # ------------------------------------------------------------------
    # Counter & decorative elements
    # ------------------------------------------------------------------

    def _create_counter_with_sink_hole(
        self,
        counter_length,
        counter_width,
        counter_thickness,
        table_height,
        table_xy_bias,
    ):
        """Build a single static "table" actor composed of 4 counter pieces
        arranged around the sink opening, so objects can physically fall
        into the basin.
        """
        sink_geom = self.kitchens_info["sink_geom"]
        sink_rel_x, sink_rel_y = sink_geom["rel_p"]
        sink_hx = sink_geom["hole_hx"]
        sink_hy = sink_geom["hole_hy"]

        th = counter_thickness / 2
        counter_top_z = table_height - th

        cl, cr = -counter_length / 2, counter_length / 2
        cf, cb = -counter_width / 2, counter_width / 2
        hl = sink_rel_x - sink_hx
        hr = sink_rel_x + sink_hx
        hf = sink_rel_y - sink_hy
        hb = sink_rel_y + sink_hy

        if self.table_texture is not None:
            texture_path = f"{os.environ['ASSETS_ROOT']}/background_texture/{self.table_texture}.png"
            texture2d = sapien.render.RenderTexture2D(texture_path)
            counter_mat = sapien.render.RenderMaterial()
            counter_mat.set_base_color_texture(texture2d)
            counter_mat.base_color = [1, 1, 1, 1]
            counter_mat.metallic = 0.1
            counter_mat.roughness = 0.3
        else:
            counter_mat = sapien.render.RenderMaterial(base_color=[0.28, 0.27, 0.26, 1])
            counter_mat.metallic = 0.12
            counter_mat.roughness = 0.22

        counter_pieces = [
            ("right", hr, cr, cf, cb),
            ("left",  cl, hl, cf, cb),
            ("front", hl, hr, cf, hf),
            ("back",  hl, hr, hb, cb),
        ]

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        for _, x1, x2, y1, y2 in counter_pieces:
            hx = (x2 - x1) / 2
            hy = (y2 - y1) / 2
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            if hx <= 1e-6 or hy <= 1e-6:
                continue
            piece_pose = sapien.Pose([cx, cy, 0])
            builder.add_box_collision(
                pose=piece_pose,
                half_size=[hx, hy, th],
                material=self.scene.default_physical_material,
            )
            builder.add_box_visual(pose=piece_pose, half_size=[hx, hy, th], material=counter_mat)

        builder.set_initial_pose(
            sapien.Pose(p=[table_xy_bias[0], table_xy_bias[1], counter_top_z])
        )
        self.table = builder.build(name="table")

    def _create_backsplash(self, counter_length, counter_width, table_height, table_xy_bias):
        backsplash_z = table_height + 0.22
        create_visual_textured_box(
            self.scene,
            sapien.Pose(p=[
                table_xy_bias[0],
                table_xy_bias[1] + counter_width / 2 + 0.005,
                backsplash_z,
            ]),
            half_size=[counter_length / 2 + 0.02, 0.006, 0.22],
            color=(0.92, 0.94, 0.96),
            name="backsplash",
        )

    def _create_base_cabinets(
        self,
        counter_length,
        counter_width,
        table_height,
        counter_thickness,
        table_xy_bias,
    ):
        cabinet_h = (table_height - counter_thickness) / 2
        cabinet_z = cabinet_h
        cx = table_xy_bias[0]
        cy = table_xy_bias[1]

        create_visual_textured_box(
            self.scene,
            sapien.Pose(p=[cx, cy - counter_width / 2 + 0.01, cabinet_z]),
            half_size=[counter_length / 2, 0.01, cabinet_h],
            color=(0.75, 0.60, 0.42),
            name="cabinet_front",
        )
        for dx in [-0.23, 0.0, 0.23]:
            create_visual_textured_box(
                self.scene,
                sapien.Pose(p=[cx + dx, cy - counter_width / 2 + 0.005, cabinet_z]),
                half_size=[0.003, 0.003, cabinet_h - 0.02],
                color=(0.55, 0.42, 0.28),
                name=f"cabinet_divider_{dx}",
            )
        for dx in [-0.35, -0.12, 0.12, 0.35]:
            handle_z = cabinet_z + 0.05
            create_visual_textured_box(
                self.scene,
                sapien.Pose(p=[cx + dx, cy - counter_width / 2 - 0.005, handle_z]),
                half_size=[0.025, 0.005, 0.004],
                color=(0.65, 0.65, 0.68),
                name=f"cabinet_handle_{dx}",
            )
        create_visual_textured_box(
            self.scene,
            sapien.Pose(p=[cx - counter_length / 2 + 0.01, cy, cabinet_z]),
            half_size=[0.01, counter_width / 2, cabinet_h],
            color=(0.70, 0.56, 0.40),
            name="cabinet_left",
        )
        create_visual_textured_box(
            self.scene,
            sapien.Pose(p=[cx + counter_length / 2 - 0.01, cy, cabinet_z]),
            half_size=[0.01, counter_width / 2, cabinet_h],
            color=(0.70, 0.56, 0.40),
            name="cabinet_right",
        )
        create_visual_textured_box(
            self.scene,
            sapien.Pose(p=[cx, cy + counter_width / 2 - 0.01, cabinet_z]),
            half_size=[counter_length / 2, 0.01, cabinet_h],
            color=(0.65, 0.52, 0.38),
            name="cabinet_back",
        )

    def _create_counter_edge_trim(
        self, counter_length, counter_width, table_height, counter_thickness, table_xy_bias
    ):
        create_visual_textured_box(
            self.scene,
            sapien.Pose(p=[
                table_xy_bias[0],
                table_xy_bias[1] - counter_width / 2,
                table_height - counter_thickness / 2,
            ]),
            half_size=[counter_length / 2, 0.003, counter_thickness / 2 + 0.002],
            color=(0.22, 0.21, 0.20),
            name="counter_edge",
        )

    def _create_upper_shelves(self, table_height, table_xy_bias):
        counter_width = self.kitchens_info["table_area"][1]
        cx = table_xy_bias[0]
        cy = table_xy_bias[1]

        for shelf_side, sx, sw in [("right", 0.28, 0.48), ("left", -0.28, 0.40)]:
            shelf_base_z = table_height + 0.32
            shelf_total_h = 0.38
            shelf_depth = 0.10
            shelf_back_y = cy + counter_width / 2 - 0.01
            shelf_front_y = shelf_back_y - shelf_depth
            shelf_center_y = (shelf_back_y + shelf_front_y) / 2
            plank_thickness = 0.012

            create_visual_textured_box(
                self.scene,
                sapien.Pose(p=[cx + sx, shelf_back_y, shelf_base_z + shelf_total_h / 2]),
                half_size=[sw / 2, 0.004, shelf_total_h / 2],
                color=(0.68, 0.54, 0.38),
                name=f"shelf_{shelf_side}_back",
            )
            create_visual_textured_box(
                self.scene,
                sapien.Pose(p=[cx + sx - sw / 2, shelf_center_y, shelf_base_z + shelf_total_h / 2]),
                half_size=[plank_thickness / 2, shelf_depth / 2, shelf_total_h / 2],
                color=(0.75, 0.60, 0.42),
                name=f"shelf_{shelf_side}_left",
            )
            create_visual_textured_box(
                self.scene,
                sapien.Pose(p=[cx + sx + sw / 2, shelf_center_y, shelf_base_z + shelf_total_h / 2]),
                half_size=[plank_thickness / 2, shelf_depth / 2, shelf_total_h / 2],
                color=(0.75, 0.60, 0.42),
                name=f"shelf_{shelf_side}_right",
            )
            for i, frac in enumerate([0.0, 0.5, 1.0]):
                board_z = shelf_base_z + frac * shelf_total_h
                create_visual_textured_box(
                    self.scene,
                    sapien.Pose(p=[cx + sx, shelf_center_y, board_z]),
                    half_size=[sw / 2, shelf_depth / 2, plank_thickness / 2],
                    color=(0.75, 0.60, 0.42),
                    name=f"shelf_{shelf_side}_board_{i}",
                )

    # ------------------------------------------------------------------
    # Fixture loaders
    # ------------------------------------------------------------------

    def _load_microwave(self, table_height, table_xy_bias):
        x, y = self._get_scene_obj_locations("microwave")
        x += table_xy_bias[0]
        y += table_xy_bias[1]

        # 1.5× the default scale from model_data.json (0.15 → 0.225).
        # Geometry scales uniformly, so the base-below-root offset scales too.
        mw_scale_mult = 1.5
        mw_scale = 0.15 * mw_scale_mult
        z = table_height + 0.02 * mw_scale_mult

        quat = euler2quat(0, 0, np.pi / 2, axes='sxyz')
        pose = sapien.Pose([x, y, z], [quat[0], quat[1], quat[2], quat[3]])

        try:
            self.microwave = create_sapien_urdf_obj(
                scene=self,
                pose=pose,
                modelname="044_microwave",
                modelid=0,
                fix_root_link=True,
                scale=mw_scale,
            )
        except Exception as e:
            print(f"[KitchenS] failed to load microwave URDF: {e}")
            self.microwave = None
            return

        self.microwave.set_name("microwave")
        self.add_prohibit_area(self.microwave, padding=0.02, area="table")
        self.collision_list.append({
            "actor": self.microwave,
            "collision_path": f"{os.environ['ASSETS_ROOT']}/objects/044_microwave/visual/base0.glb",
        })

        # Tasks that close the microwave door (close_microwave_ks and the
        # heat/serve chains) must keep the door-swing zone in front of the
        # microwave clear — an object spawned there blocks the door from
        # closing. The microwave opening/door faces -y (the robot side), so
        # prohibit a strip spanning the microwave width and extending in front.
        if getattr(self, "microwave_front_keepout", False):
            # Same list object is stored so rand_pose_on_counter can skip
            # this box with allow_microwave_front without matching by value.
            box = [
                x - 0.25, y - 0.22,   # x0, y0  (-y = robot / door side)
                x + 0.25, y + 0.10,   # x1, y1
            ]
            self.prohibited_area["table"].append(box)
            self._microwave_front_keepout_box = box
        else:
            self._microwave_front_keepout_box = None

    def _load_dishrack(self, table_height, table_xy_bias):
        x, y = self._get_scene_obj_locations("dishrack")
        x += table_xy_bias[0]
        y += table_xy_bias[1]

        # Base orientation lays the rack flat on the counter (basin opens up)
        # by rotating +90° around the world x-axis.
        rack_q = np.array([0.707, 0.707, 0, 0])  # wxyz, +90° about x

        # Compute z offset from the actual visual mesh bounds (json extents do
        # not match the glb). After the +90° x-rotation, the mesh's original
        # +y axis becomes world +z, so world bottom = origin_z + y_min * scale.
        # 135_dish-rack is a benchmark-custom asset under assets/objects/
        # (lives under assets/objects/). create_actor is hardcoded to
        # assets/objects/, so the actor is built inline here.
        # rack_asset_dir = f"{os.environ['ASSETS_ROOT']}/objects/135_dish-rack"
        rack_asset_dir = f"{os.environ['ASSETS_ROOT']}/objects/135_dish-rack"
        with open(f"{rack_asset_dir}/model_data0.json") as _f:
            _rd = json.load(_f)
        # Default JSON scale (0.6435) puts the rack top at z ≈ 0.89, putting
        # the plate target z (rack_center + 0.06) at ≈ 0.96. The aloha arm
        # wrist target (plate_z + TCP_OFFSET) ≈ 1.08 is past the IK envelope
        # at the rack's depth on the counter. Scaling the rack to 0.4
        # lowers the plate target to ≈ 0.90 → wrist ≈ 1.02 (sink-equivalent).
        _rack_scale = _rd["scale"][0] * 1.1
        _rack_mesh = trimesh.load(f"{rack_asset_dir}/base0.glb", force="mesh")
        _y_min = float(_rack_mesh.bounds[0][1])
        rack_z = table_height - _y_min * _rack_scale

        _rack_scale_xyz = [_rack_scale, _rack_scale, _rack_scale]
        _rack_builder = self.scene.create_actor_builder()
        _rack_builder.set_physx_body_type("static")
        # The rack is static, so use the exact visual mesh as a non-convex
        # (triangle-mesh) collision. The shipped collision/base0.glb was a
        # mismatched, undersized convex blob; convex decomposition is also
        # wrong for a concave basin (it fills the opening). Non-convex of the
        # real mesh keeps the basin and matches the visual exactly.
        _rack_builder.add_nonconvex_collision_from_file(
            filename=f"{rack_asset_dir}/base0.glb", scale=_rack_scale_xyz)
        _rack_builder.add_visual_from_file(
            filename=f"{rack_asset_dir}/base0.glb", scale=_rack_scale_xyz)
        _rack_entity = _rack_builder.build()
        # Mesh origin is outside the mesh body (mesh-x bounds asym, mesh-z entirely
        # positive). After +90° x-rotation, mesh-x → world-x and mesh-z → world -y,
        # so without compensation the rack sits offset in world XY from the spawn
        # coord. Shift the entity pose by -(mesh_center_scaled_in_world) so the
        # rack's actual AABB center lands at (x, y).
        _mx_min, _my_min, _mz_min = _rack_mesh.bounds[0] * _rack_scale
        _mx_max, _my_max, _mz_max = _rack_mesh.bounds[1] * _rack_scale
        _cx_off = 0.5 * (_mx_min + _mx_max)         # world-x offset of mesh center
        _cy_off = -0.5 * (_mz_min + _mz_max)        # world-y offset (mesh-z → -world-y)
        rack_pose_x = x - _cx_off
        rack_pose_y = y - _cy_off
        _rack_entity.set_pose(sapien.Pose(p=[rack_pose_x, rack_pose_y, rack_z], q=rack_q.tolist()))
        _rack_entity.set_name("135_dish-rack")
        self.dishrack = Simple_Actor(_rack_entity, scale=_rack_scale_xyz)
        self.dishrack.set_name("dishrack")
        # True geometric center of the rack on the counter. The rack ENTITY
        # pose (dishrack.get_pose()) is deliberately offset from the body (the
        # mesh origin sits outside the mesh), so check_success must compare
        # placed objects against this real center, not the entity pose.
        self.dishrack_center = np.array([x, y, rack_z])
        # Prohibit-area footprint in world AABB, centered on the spawn coord now
        # that the rack pose is compensated.
        _rack_x0, _rack_x1 = _mx_min - _cx_off, _mx_max - _cx_off
        _rack_y0, _rack_y1 = -_mz_max - _cy_off, -_mz_min - _cy_off
        _rack_pad = 0.04
        self.prohibited_area["table"].append([
            x + _rack_x0 - _rack_pad,
            y + _rack_y0 - _rack_pad,
            x + _rack_x1 + _rack_pad,
            y + _rack_y1 + _rack_pad,
        ])
        self.collision_list.append({
            "actor": self.dishrack,
            "collision_path": f"{os.environ['ASSETS_ROOT']}/objects/135_dish-rack/base0.glb",
        })
        # NOTE: the old invisible box "containment tray" was removed. It sat a
        # solid floor at rack_top_z, so objects landed on top of the rack
        # instead of descending into the basin ("floating"). With the rack now
        # using an exact non-convex collision, objects settle into the real
        # basin geometry directly.

    def _load_sink(self, table_height, table_xy_bias):
        sink_geom = self.kitchens_info["sink_geom"]
        rel_x, rel_y = sink_geom["rel_p"]
        x = rel_x + table_xy_bias[0]
        y = rel_y + table_xy_bias[1]

        hole_hx = sink_geom["hole_hx"]
        hole_hy = sink_geom["hole_hy"]
        depth = sink_geom["depth"]
        inner_hx = sink_geom["inner_hx"]
        inner_hy = sink_geom["inner_hy"]

        sink_z = table_height
        wall_thickness = hole_hx - inner_hx

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")

        material = sapien.render.RenderMaterial(base_color=[0.75, 0.75, 0.78, 1.0])
        material.metallic = 0.6
        material.roughness = 0.3

        # Basin floor
        bottom_half = [inner_hx, inner_hy, wall_thickness / 2]
        bottom_pose = sapien.Pose([0, 0, -depth + wall_thickness / 2])
        builder.add_box_collision(pose=bottom_pose, half_size=bottom_half)
        builder.add_box_visual(pose=bottom_pose, half_size=bottom_half, material=material)

        # Four side walls
        walls = [
            ([hole_hx - wall_thickness / 2, 0, -depth / 2], [wall_thickness / 2, hole_hy, depth / 2]),  # +x
            ([-(hole_hx - wall_thickness / 2), 0, -depth / 2], [wall_thickness / 2, hole_hy, depth / 2]),  # -x
            ([0, hole_hy - wall_thickness / 2, -depth / 2], [inner_hx, wall_thickness / 2, depth / 2]),  # +y
            ([0, -(hole_hy - wall_thickness / 2), -depth / 2], [inner_hx, wall_thickness / 2, depth / 2]),  # -y
        ]
        for w_pos, w_half in walls:
            wp = sapien.Pose(w_pos)
            builder.add_box_collision(pose=wp, half_size=w_half)
            builder.add_box_visual(pose=wp, half_size=w_half, material=material)

        builder.set_initial_pose(sapien.Pose(p=[x, y, sink_z]))
        self.sink = builder.build(name="sink")

        # Thin chrome rim framing the hole (visual only)
        rim_t = 0.012
        rim_hz = 0.003
        rim_z = table_height + 0.003
        rim_color = (0.82, 0.82, 0.85)
        create_box(
            self.scene,
            sapien.Pose(p=[x, y - hole_hy - rim_t / 2, rim_z]),
            half_size=[hole_hx + rim_t, rim_t / 2, rim_hz],
            color=rim_color, name="sink_rim_front", is_static=True,
        )
        create_box(
            self.scene,
            sapien.Pose(p=[x, y + hole_hy + rim_t / 2, rim_z]),
            half_size=[hole_hx + rim_t, rim_t / 2, rim_hz],
            color=rim_color, name="sink_rim_back", is_static=True,
        )
        create_box(
            self.scene,
            sapien.Pose(p=[x - hole_hx - rim_t / 2, y, rim_z]),
            half_size=[rim_t / 2, hole_hy, rim_hz],
            color=rim_color, name="sink_rim_left", is_static=True,
        )
        create_box(
            self.scene,
            sapien.Pose(p=[x + hole_hx + rim_t / 2, y, rim_z]),
            half_size=[rim_t / 2, hole_hy, rim_hz],
            color=rim_color, name="sink_rim_right", is_static=True,
        )

        # Faucet — upright post, gooseneck spout, small side handle
        faucet_y = y + inner_hy + 0.02
        faucet_color = (0.82, 0.82, 0.85)
        create_visual_textured_box(
            self.scene,
            sapien.Pose(p=[x, faucet_y, table_height + 0.10]),
            half_size=[0.010, 0.010, 0.10],
            color=faucet_color,
            name="faucet_post",
        )
        create_visual_textured_box(
            self.scene,
            sapien.Pose(p=[x, faucet_y - 0.06, table_height + 0.19]),
            half_size=[0.008, 0.07, 0.008],
            color=faucet_color,
            name="faucet_spout",
        )
        create_visual_textured_box(
            self.scene,
            sapien.Pose(p=[x + 0.04, faucet_y, table_height + 0.06]),
            half_size=[0.012, 0.006, 0.015],
            color=faucet_color,
            name="faucet_handle",
        )

        # Prohibited zone covers basin + faucet footprint so task objects
        # never spawn on top of the open sink.
        sink_pad = 0.03
        self.prohibited_area["table"].append([
            x - hole_hx - sink_pad,
            y - hole_hy - sink_pad,
            x + hole_hx + sink_pad,
            faucet_y + sink_pad,
        ])

    # ------------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------------

    def add_extra_cameras(self):
        self.cameras.add_extra_cameras(f"{os.environ['ASSETS_ROOT']}/embodiments/kitchen_s_config.yml")

    # ------------------------------------------------------------------
    # Clutter
    # ------------------------------------------------------------------

    def get_cluttered_surfaces(self):
        xlim = [self.kitchens_info["table_lims"][0], self.kitchens_info["table_lims"][2]]
        ylim = [self.kitchens_info["table_lims"][1], self.kitchens_info["table_lims"][3]]
        zlim = [self.kitchens_info["table_height"]]
        xlim[0] += self.table_xy_bias[0]
        xlim[1] += self.table_xy_bias[0]
        ylim[0] += self.table_xy_bias[1]
        ylim[1] += self.table_xy_bias[1]
        zlim = np.array(zlim) + self.table_z_bias

        task_objects_list = []
        for entity in self.scene.get_all_actors():
            actor_name = entity.get_name()
            if actor_name == "" or actor_name in self.FURNITURE_NAMES:
                continue
            task_objects_list.append(actor_name)

        cluttered_item_info, obj_names_short, obj_names_tall = get_obstacle_objects_subset(
            "kitchens", self.obstacle_distribution, task_objects_list
        )

        self.clutter_surface_split(
            xlim, ylim, zlim,
            self.prohibited_area["table"],
            self.obstacle_density,
            cluttered_item_info,
            obj_names_short,
            obj_names_tall,
        )

    # ------------------------------------------------------------------
    # Collision helpers
    # ------------------------------------------------------------------

    def enable_table(self, enable: bool):
        names = [f"table_[0, 0, 0.74, 1, 0, 0, 0]_{self._collision_cache_id()}"]
        self.enable_obstacle(enable, obb_names=names)

    def grasp_actor_from_table(
        self,
        actor: Actor,
        arm_tag: ArmTag,
        pre_grasp_dis=0.1,
        grasp_dis=0,
        gripper_pos=0.0,
        contact_point_id: list | float = None,
    ):
        _, actions = self.grasp_actor(
            actor=actor, arm_tag=arm_tag,
            pre_grasp_dis=pre_grasp_dis, grasp_dis=grasp_dis,
            gripper_pos=gripper_pos, contact_point_id=contact_point_id,
        )
        # grasp_actor returns [] when choose_grasp_pose finds no valid
        # pre-grasp; degrade to a clean plan failure so the collector
        # retries with a new seed instead of crashing.
        if not actions:
            self.plan_success = False
            return
        self.move((arm_tag, [actions[0]]))
        self.enable_table(enable=False)
        self.move((arm_tag, actions[1:]))

    # ------------------------------------------------------------------
    # Sink query helpers (for task success checking)
    # ------------------------------------------------------------------

    def is_object_in_sink(self, actor, margin=0.01):
        sx, sy = self._get_scene_obj_locations("sink")
        sx += self.table_xy_bias[0]
        sy += self.table_xy_bias[1]
        hole_hx, hole_hy = 0.13, 0.20

        pos = actor.get_pose().p
        return (
            sx - hole_hx + margin < pos[0] < sx + hole_hx - margin
            and sy - hole_hy + margin < pos[1] < sy + hole_hy - margin
        )

    def is_object_on_dishrack(self, actor, margin=0.02):
        dx, dy = self._get_scene_obj_locations("dishrack")
        dx += self.table_xy_bias[0]
        dy += self.table_xy_bias[1]
        rack_hx, rack_hy = 0.10, 0.08

        pos = actor.get_pose().p
        return (
            dx - rack_hx - margin < pos[0] < dx + rack_hx + margin
            and dy - rack_hy - margin < pos[1] < dy + rack_hy + margin
            and pos[2] > self.kitchens_info["table_height"] + 0.01
        )
