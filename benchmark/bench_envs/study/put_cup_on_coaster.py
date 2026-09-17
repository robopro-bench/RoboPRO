# from envs._base_task import Base_Task
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
from bench_envs.utils.scene_gen_utils import print_c, place_actor
from transforms3d.euler import euler2quat


class put_cup_on_coaster(Study_base_task):

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name(), self.des_obj.get_name()}

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        print_c(self.seed, "YELLOW")
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r", encoding="utf-8") as f:
            task_objs = yaml.safe_load(f)
        
        xlim, ylim, self.side_to_place = get_position_limits(self.table, 
                                                             boundary_thr=0.15, side="right" if self.scene_id == 0 else "left")
      
        object_bounds = [get_actor_boundingbox(o) for o in self.scene_objs]
        
        self.target_name = "021_cup"
        self.target_obj, self.target_id, self.target_pose = \
        place_actor(self.target_name, self, col_thr=0.10, xlim=xlim, ylim=ylim,
                    qpos= (90,0,0), object_bounds=object_bounds, task_objs=task_objs,
                    obj_id = None, mass = 0.1, rotation=False)
        tar_bb = get_actor_boundingbox(self.target_obj.actor)
        object_bounds.append(tar_bb)

        self.des_obj, self.des_obj_id, self.des_obj_pose = \
            place_actor("019_coaster", self, col_thr=0.15, xlim=xlim, ylim=ylim, qpos=(0,0,0),
                         object_bounds=object_bounds, task_objs=task_objs,  obj_id = "0", mass = 0.2,
                         rotation=False)
        
        # Get the placement pose
        p = self.des_obj.get_pose().p.tolist() 

        p[1] -= 0.05  # Coaster appears 5cm below in the simulator
        p[-1] = self.target_obj.get_pose().p[-1]
        
        self.des_obj_pose = p +  [1,0,0,0] 

        print_c(f"Placement destination pose {self.des_obj_pose}", "RED")


        self.add_prohibit_area(self.target_obj, padding=0.05, area="table")
        self.add_prohibit_area(self.des_obj, padding=0.1, area="table")

     
      
    def play_once(self, z = 0.1, pre_dis= 0.07, dis=0.005, pre_grasp_dist=0.1):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag(self.side_to_place ) #("right" if self.target_obj.get_pose().p[0] > 0 else "left")

        # Grasp the mouse with the selected arm
        self.move(self.grasp_actor(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=pre_grasp_dist))

        # Lift the mouse upward by 0.1 meters in z-direction
        # self.move(self.move_by_displacement(arm_tag=arm_tag, z=z))

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
        eps = 0.02
        return (np.all(np.abs(self.des_obj.get_pose().p[:2] - self.target_obj.get_pose().p[:2]) < [eps, eps])  
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
