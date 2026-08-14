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

from bench_envs._bench_base_task import Bench_base_task
from envs.utils import *
from bench_envs.utils import *
import math
from envs.robot import Robot
from envs.camera import Camera
from envs.utils.actor_utils import Actor, ArticulationActor
from bench_envs.utils.scene_gen_utils import get_actor_boundingbox
from bench_envs.utils.scene_gen_utils import print_c

from copy import deepcopy
import subprocess
from pathlib import Path
import trimesh
import imageio
import glob
import yaml
import importlib

from envs._GLOBAL_CONFIGS import *

from typing import Optional, Literal
from transforms3d.euler import euler2quat

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


class Study_base_task(Bench_base_task):
    # Study scene furniture: table, wall, floor, ground, bookcase, wooden_box
    FURNITURE_NAMES = {"table", "wall", "floor", "ground", "014_bookcase", "042_wooden_box"}

    def __init__(self):
        pass

    def _get_target_object_names(self) -> set[str]:
        """Default for tasks with single self.target_obj. Override for multi-target tasks."""
        return {self.target_obj.get_name()}

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
        self.scene_objs = []
        self.scene_obj_info = []

        self.need_topp = True  # TODO

        # Random
        random_setting = kwags.get("domain_randomization")
        self.random_background = random_setting.get("random_background", False)
        self.cluttered_table = random_setting.get("cluttered_table", False)
        self.clean_background_rate = random_setting.get("clean_background_rate", 1)
        self.random_head_camera_dis = random_setting.get("random_head_camera_dis", 0)
        self.random_table_height = random_setting.get("random_table_height", 0)
        self.random_light = random_setting.get("random_light", False)
        self.crazy_random_light_rate = random_setting.get("crazy_random_light_rate", 0)
        self.crazy_random_light = (0 if not self.random_light else np.random.rand() < self.crazy_random_light_rate)
        self.random_embodiment = random_setting.get("random_embodiment", False)  # TODO
        self.obstacle_height = random_setting.get("obstacle_height", "short")
        self.obstacle_density = random_setting.get("obstacle_density", 3)

        self._parse_perturbations(random_setting)

        self.col_temp = os.environ['ASSETS_ROOT'] + "/objects/{object}/collision/base{object_id}.glb"
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

        self.save_freq = kwags.get("save_freq")
        self.video_fps = kwags.get("video_fps", 30)
        self.world_pcd = None

        self.size_dict = list()
        self.cluttered_objs = list()
        self.prohibited_area = {"table": [], "shelf0": [], "shelf1": []} # shelf 0 for lower shelf, shelf 1 for upper shelf
        self.unstable_objects = []
        self.record_cluttered_objects = list()  # record cluttered objects info
        self.cluttered_objects_info = get_cluttered_objects_info()

        self.eval_success = False
        self.table_z_bias = (np.random.uniform(low=-self.random_table_height, high=0) + table_height_bias)  # TODO
        self.shelf_heights = [0.845, 1.265] # heights of the shelf levels
        self.need_plan = kwags.get("need_plan", True)
        self.left_joint_path = kwags.get("left_joint_path", [])
        self.right_joint_path = kwags.get("right_joint_path", [])
        self.left_cnt = 0
        self.right_cnt = 0
        self.scene_id =kwags.get("scene_id") if kwags.get("scene_id") is not None else np.random.randint(0,3)  # for furniture arrangement
        self.incl_collision = kwags.get("include_collison", True)
        self.instruction = None  # for Eval

        self.collision_list = [] # list of collision objects for curobo planner
        self.cuboid_collision_list = [] # list of cuboid collision objects for curobo planner
        self._init_collision_metrics()

        self.load_robot(**kwags)
        self.create_static_elements(table_xy_bias=table_xy_bias, table_height=0.74)
        self.load_camera(**kwags)
        self.robot.move_to_homestate()

        render_freq = self.render_freq
        self.render_freq = 0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq
        self.cuboid_collision_list = []
        self.robot.set_origin_endpose()
        self.load_actors()
        
        if self.cluttered_table:
            self.get_cluttered_surfaces()

        self._apply_specular_ood()
        self._furniture_texture_targets = [
            ("bookcase", "bookcase"),
            ("box", "box"),
        ]
        self._apply_furniture_texture_ood()

        if self.enable_collision_metrics:
            self._build_collision_name_sets()

        # per-step proximity (nearest-obstacle clearance) tracking
        if kwags.get("data_type", {}).get("proximity", True):
            self._init_proximity_tracking(kwags.get("proximity_tracking", {}))

        is_stable, unstable_list = self.check_stable()
        if not is_stable:
            raise UnStableError(
                f'Objects is unstable in seed({kwags.get("seed", 0)}), unstable objects: {", ".join(unstable_list)}')
            # print(f'Objects is unstable in seed({kwags.get("seed", 0)}), unstable objects: {", ".join(unstable_list)}')

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

    def add_collision(self):
        for obj in self.scene_obj_info:
                print_c(f"Adding collision for {obj[0]} with id {obj[2]}", "GREEN")
                self.collision_list.append({
                    "actor":obj[1],
                    "collision_path": self.col_temp.format(object=obj[0], object_id=obj[2])
            })
    def create_static_elements(self, table_xy_bias=[0, 0], table_height=0.74):
        self.table_xy_bias = table_xy_bias
        wall_texture, table_texture, floor_texture = None, None, None
        table_height += self.table_z_bias
        print_c(f"Scene {self.scene_id} is selected", "YELLOW")
        # Setup textures 
        if self.random_background:
            texture_type = "seen" if not self.eval_mode else "unseen"
            directory_path = f"{os.environ['ASSETS_ROOT']}/background_texture/{texture_type}"
            file_count = len(
                [name for name in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, name))])

            # wall_texture, table_texture = np.random.randint(0, file_count), np.random.randint(0, file_count)
            if texture_type == "seen":
                wall_texture = np.random.randint(0, file_count)
                table_texture = np.random.choice(
                    # simple textures, not distracting
                    [0,2,4,5,7,9,14,16,18,19]
                    )
                floor_texture = np.random.choice(
                    # simple textures, not distracting
                    [2,3,4,5,6,17,47,71,110]
                    )
            else:
                wall_texture = np.random.randint(0, file_count)
                table_texture = np.random.choice(
                    # simple textures, not distracting
                    [1,8,9,27,29,30,37,55]
                    )
                floor_texture = np.random.choice(
                    # simple textures, not distracting
                    [0,6,8,23,34,41,47,61,64]
                    )

            self.wall_texture = f"{texture_type}/{wall_texture}"
            self.table_texture = f"{texture_type}/{table_texture}"
            self.floor_texture = f"{texture_type}/{floor_texture}"
            if np.random.rand() <= self.clean_background_rate:
                self.wall_texture = None
            if np.random.rand() <= self.clean_background_rate:
                self.table_texture = None
            if np.random.rand() <= self.clean_background_rate:
                self.floor_texture = None
        else:
            self.wall_texture, self.table_texture, self.floor_texture = None, None, None

        # Create room
        self.floor = create_visual_textured_box(
            self.scene,
            sapien.Pose(p=[0, 0, 0]),
            half_size=[2, 2, 0.005],
            color=(0.85, 0.85, 0.85),
            name="floor",
            texture_id=self.floor_texture,
        )

        self.wall = create_box(
            self.scene,
            sapien.Pose(p=[0, 1, 1.5]),
            half_size=[3, 0.6, 1.5],
            color=(1, 0.9, 0.9),
            name="wall",
            texture_id=self.wall_texture,
            is_static=True,
        )

        # Create scene specific elements
        with open(f"{os.environ['BENCH_ROOT']}/bench_envs/scene_configs/study.yaml", "r") as f:
            obj_config = yaml.safe_load(f)
        for obj, param in obj_config[f"scene_{self.scene_id}"].items():
            pose = sapien.Pose(
                    p = param.get("p", [*table_xy_bias, table_height]),
                    q = euler2quat(*[np.deg2rad(d) for d in param.get("q", [0,0,0])])
            )
            if param.get("obj_type", "actor") == "actor":
                value = create_actor(
                    scene=self,
                    pose=pose,
                    modelname=obj,
                    convex=param.get("convex", True),
                    scale=param.get("scale"),
                    model_id= param.get('model_id', 0),
                    is_static=param.get('is_static', True)
                )
                self.scene_objs.append(value)
                self.scene_obj_info.append((obj,value, param.get('model_id', 0)))
                self.add_prohibit_area(value, padding=0.05, area="table")

            elif param.get("obj_type", "actor") == "table":
                value= create_table(
                        self.scene,
                        pose=pose,
                        length=param.get("length", 1.2),
                        width=param.get("width", 0.7),
                        height=param.get("height", table_height),
                        thickness=param.get("tickness", 0.05),
                        is_static=param.get('is_static', True),
                        texture_id=param.get('texture_id', self.table_texture),
                    )
            setattr(self, obj.split("_")[-1], value)
        if self.incl_collision:
            self.add_collision()
            
        box_bb = get_actor_boundingbox(self.box.actor)
        case_bb = get_actor_boundingbox(self.bookcase.actor)
        self.prohibited_area["table"].append([box_bb[0][0], case_bb[0][1],
                                                box_bb[1][0], case_bb[1][1]])

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
        
        # collect objects already on the scene
        task_objects_list = []
        for entity in self.scene.get_all_actors():
            actor_name = entity.get_name()
            if actor_name == "":
                continue
            if actor_name in ["table", "wall", "ground"]:
                continue
            task_objects_list.append(actor_name)

        cluttered_item_info, obj_names_short, obj_names_tall = get_obstacle_objects_subset(
            "study", self.obstacle_distribution, task_objects_list
        )
        self.clutter_surface_split(xlim, ylim, zlim, self.prohibited_area["table"], self.obstacle_density, cluttered_item_info, obj_names_short, obj_names_tall)

    def add_extra_cameras(self):
        self.cameras.add_extra_cameras(f"{os.environ['ASSETS_ROOT']}/embodiments/office_config.yml")