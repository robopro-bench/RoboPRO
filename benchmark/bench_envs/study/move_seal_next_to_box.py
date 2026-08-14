# from envs._base_task import Base_Task
import sapien
import math
import glob
import yaml
import os
import numpy as np
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
from bench_envs.study._study_base_task import Study_base_task
from envs.utils import *
from bench_envs.utils.scene_gen_utils import get_position_limits, get_actor_boundingbox, get_collison_with_objs
from bench_envs.utils.scene_gen_utils import print_c, place_actor, point_to_box_distance
from transforms3d.euler import euler2quat

class move_seal_next_to_box(Study_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r", encoding="utf-8") as f:
            task_objs = yaml.safe_load(f)
        xlim, ylim, self.side_to_place = get_position_limits(self.table,
                                      boundary_thr=0.10, side="left" if self.scene_id==0 else "right")
        
        object_bounds = [get_actor_boundingbox(o) for o in self.scene_objs]
        self.target_name = "100_seal"
        self.move_thr = 0.05

        xlim_m = [xlim[0] + self.move_thr + 0.1, xlim[1]] if self.scene_id == 0 \
        else [xlim[0], xlim[1] - self.move_thr - 0.1]
        self.target_obj, self.target_id, self.target_pose = \
        place_actor(self.target_name, self, col_thr=0.20, xlim=xlim_m, ylim=ylim, 
                    qpos=(90,0,0), object_bounds=object_bounds, task_objs=task_objs,
                     mass = 0.1, rotation=False)
        
        bb_box = get_actor_boundingbox(self.box.actor)        
        self.des_obj_pose = [bb_box[1][0] + self.move_thr if self.side_to_place == "left" else bb_box[0][0] - self.move_thr,
             np.random.uniform(low=bb_box[0][1]+0.05, high=bb_box[1][1]-0.05),
             bb_box[1][-1]] + [1,0,0,0]

        print_c(f"Placement destination pose {self.des_obj_pose}", "RED")

        self.add_prohibit_area(self.target_obj, padding=0.1, area="table")
        self.add_prohibit_area(self.des_obj_pose, padding=0.0, area="table")


    def play_once(self, z = 0.05, pre_dis= 0.07, dis=0.005, pre_grasp_dist=0.1):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag(self.side_to_place ) 
        # Grasp the mouse with the selected arm
        self.move(self.grasp_actor(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=pre_grasp_dist))

        # Lift the mouse upward by 0.1 meters in z-direction
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=z))

        self.attach_object(self.target_obj, f"{os.environ['ASSETS_ROOT']}/objects/{self.target_name}/collision/base{self.target_id}.glb", str(arm_tag))
        # Place the mouse at the target location with alignment constraint
        self.move(
            self.place_actor(
                self.target_obj,
                arm_tag=arm_tag,
                target_pose= self.des_obj_pose,
                constrain= "auto",
                pre_dis=pre_dis,
                dis=dis,
            ))

        # Record information about the objects and arm used in the task
        self.info["info"] = {
            "{A}": f"{self.target_name}/base{self.target_id}",
            "{B}": f"red",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        dist_thr = 0.15
        box_bb = get_actor_boundingbox(self.box.actor)
        dist_to_box = point_to_box_distance(self.target_obj.get_pose().p, box_bb[0], box_bb[1])

        return (dist_to_box < dist_thr
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())

