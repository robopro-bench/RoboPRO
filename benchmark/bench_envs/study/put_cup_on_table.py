# from envs._base_task import Base_Task
import sapien
import math
import glob
import yaml
import os
import numpy as np

from bench_envs.study._study_base_task import Study_base_task
from envs.utils import *
from envs._GLOBAL_CONFIGS import *
from bench_envs.utils.scene_gen_utils import get_position_limits, get_actor_boundingbox
from bench_envs.utils.scene_gen_utils import print_c, place_actor,get_random_place_pose

class put_cup_on_table(Study_base_task):

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name(), self.des_obj.get_name()}

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r", encoding="utf-8") as f:
            task_objs = yaml.safe_load(f)
        
        xlim, ylim, self.arm_side = get_position_limits(self.table, boundary_thr=0.25)
       
        object_bounds = [get_actor_boundingbox(o) for o in self.scene_objs]

        self.des_obj_name = "043_book"
        self.des_obj, self.des_obj_id, self.des_obj_pose = \
        place_actor(self.des_obj_name, self, col_thr =0.15,
                     xlim=xlim, ylim=ylim, qpos=(90,0,90),
                     object_bounds = object_bounds, task_objs = task_objs,
                     obj_id = None, mass = 0.5, rotation=False)

        des_bb = get_actor_boundingbox(self.des_obj.actor)

        self.target_name = "021_cup"
        
        self.target_pose = self.des_obj_pose
        self.target_pose.set_p([*self.target_pose.p[:2], des_bb[1][-1]]) 

        self.target_obj, self.target_id, self.target_pose = \
            place_actor(self.target_name, self, task_objs = task_objs,
                    obj_pose=self.target_pose, mass = 0.2)
        
        self.cup_des_pose = get_random_place_pose(xlim = xlim, ylim=ylim,
                                             col_thr=0.15,zlim=[0.78],
                                             object_bounds=object_bounds)

        self.add_prohibit_area(self.target_obj, padding=0.0, area="table")
        self.add_prohibit_area(self.cup_des_pose, padding=0.0, area="table")
        
        self.cup_des_pose = self.cup_des_pose.p.tolist() + self.cup_des_pose.q.tolist()
        print_c(f"Cup destination {self.cup_des_pose}", "RED")
    def play_once(self,z = 0.05, pre_dis= 0.07, dis=0.005, pre_grasp_dist=0.1):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag(self.arm_side)
        # Grasp the mouse with the selected arm
        self.move(self.grasp_actor(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=pre_grasp_dist))

        # Lift the mouse upward by 0.1 meters in z-direction
        self.move(self.move_by_displacement(arm_tag=arm_tag,z=z))

        self.attach_object(self.target_obj, f"{os.environ['ASSETS_ROOT']}/objects/{self.target_name}/collision/base{self.target_id}.glb", str(arm_tag))
       
        self.move(
            self.place_actor(
                self.target_obj,
                arm_tag=arm_tag,
                target_pose= self.cup_des_pose,
                constrain="auto",
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
        eps = 0.01
        b_pose = self.target_obj.get_pose().p
        table_bb = get_actor_boundingbox(self.table)
        cup_on_table = np.all((table_bb[0][:2] <= b_pose[:2])  &  (b_pose[:2] <= table_bb[1][:2]))
        cup_on_table &= (b_pose[-1] - table_bb[1][-1]) < eps  
        return (cup_on_table 
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())