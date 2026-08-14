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
from bench_envs.utils.scene_gen_utils import print_c, place_actor
from transforms3d.euler import euler2quat


class put_pen_in_pencup(Study_base_task):

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name(), self.des_obj.get_name()}

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        print_c(self.seed, "YELLOW")
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r", encoding="utf-8") as f:
            task_objs = yaml.safe_load(f)

        object_bounds = [get_actor_boundingbox(o) for o in self.scene_objs]

        self.target_name = "058_markpen"

        xlim, ylim, self.side_to_place = get_position_limits(self.table,
                                         boundary_thr=[0.15, 0.25],
                                           side="right" if self.scene_id == 0 else "left")

        self.target_obj, self.target_id, self.target_pose = \
        place_actor(self.target_name, self, col_thr=0.15, xlim=[xlim[0], (xlim[0]+xlim[1])/2],
                    ylim=ylim, qpos=(90,0,90),object_bounds= object_bounds,
                     task_objs=task_objs,  obj_id = 0, mass = 0.1)


        tar_bb = get_actor_boundingbox(self.target_obj.actor)
        object_bounds.append(tar_bb)

        self.des_obj, self.des_obj_id, self.des_obj_pose = \
            place_actor("059_pencup", self, col_thr=0.15, xlim=[(xlim[0]+xlim[1])/2, xlim[1]],
                        ylim= ylim, qpos=(90,0,90),
                        object_bounds=object_bounds, task_objs=task_objs,
                        obj_id = 1, mass = 0.2, rotation=False) 
        self.collision_list.append({
                "actor":self.des_obj,
                "collision_path": self.col_temp.format(object="059_pencup",
                                                        object_id=1)
            })
        # Get the placement pose
        self.des_obj_pose = self.des_obj_pose.p.tolist() + euler2quat(0,0,np.deg2rad(90)).tolist()
        self.des_obj_pose[2] = 0.90

        print_c(f"Placement destination pose {self.des_obj_pose}", "RED")


        self.add_prohibit_area(self.target_obj, padding=0.10, area="table")
        self.add_prohibit_area(self.des_obj, padding=0.0, area="table")

    def play_once(self, z = 0.15, pre_dis= 0.01, dis=0.005, pre_grasp_dist=0.1):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag(self.side_to_place )
        # Grasp the mouse with the selected arm
        self.move(self.grasp_actor(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=pre_grasp_dist))
        self.attach_object(self.target_obj, f"{os.environ['ASSETS_ROOT']}/objects/{self.target_name}/collision/base{self.target_id}.glb", str(arm_tag))

        self.move(self.move_by_displacement(arm_tag=arm_tag, y= -z, z=z))
        self.move(
            self.place_actor(
                self.target_obj,
                arm_tag=arm_tag,
                target_pose= self.des_obj_pose,
                constrain= "free",
                pre_dis=pre_dis,
                dis=dis,
                actor_axis_type="world"
            ))

        # Record information about the objects and arm used in the task
        self.info["info"] = {
            "{A}": f"{self.target_name}/base{self.target_id}",
            "{B}": f"red",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        target_pose = self.target_obj.get_pose().p
        target_des_pos = self.des_obj.get_pose().p
        eps = 0.03

        return (np.all(abs(target_pose[:2] - target_des_pos[:2]) < np.array([eps, eps]))
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())