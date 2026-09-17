import os

import yaml

from bench_envs.kitchenl._kitchen_base_large import Kitchen_base_large
from bench_envs.utils.scene_gen_utils import get_actor_boundingbox, get_actor_boundingbox_urdf, get_random_place_pose, place_actor, point_to_box_distance, print_c
from bench_envs.utils.scene_gen_utils import get_actor_boundingbox_urdf
from envs.utils import *
import math
import numpy as np
import sapien
import transforms3d as t3d


class put_can_infront_of_microwave(Kitchen_base_large):
    MILK_BOX_MASS = 0.1
    MILK_BOX_SPAWN_Z_OFFSET = 0
    TABLE_WORLD_XY_JITTER = 0.05

    # Success region multiplier around basket bounds.
    BASKET_EXPANSION_RATIO = 1

    # Left-arm motion tuning.
    APPROACH_DELTA_1 = dict(x=-0.22, y=0.22, z=-0.14)
    RETREAT_DELTA = dict(y=-0.07, z=0.02)
    GRASP_PRE_DIS = 0.07
    GRASP_DIS = 0.01
    GRASP_CLOSE_POS = 0.0

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}  # place_actor("071_can") → name = "071_can"

    def setup_demo(self, is_test: bool = False, **kwargs):
        kwargs["include_collision"] = True
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)



    def load_actors(self):
        if self.scene_id == 0:
            xlim = [-0.45, -0.2]
        else:
            xlim = [0, 0.1]
        self.target_name = "071_can"
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r", encoding="utf-8") as f:
            task_objs = yaml.safe_load(f)

        self.target_obj, self.target_id, self.target_pose = \
        place_actor(self.target_name, self, col_thr=0.15, xlim=xlim, ylim=[-0.1], 
                    qpos=(90,0,0), object_bounds={}, task_objs=task_objs,
                     mass = 0.2, rotation=False, scene_name='kitchenl')
        
        self.add_prohibit_area(self.target_obj, padding=0.0, area="table")
        
        print_c(f"Can placement {self.target_pose}", "BLUE")



        self.micp = self.microwave_left.get_pose().p
        self.des_obj_pose = [np.random.uniform(self.micp[0]-0.1, self.micp[0]+0.1),
                             self.micp[1]-0.3,  0.78] + [1,0,0,0]
        
        self.add_prohibit_area(self.des_obj_pose, padding=0.0, area="table")

        print_c(f"Placement destination pose {self.des_obj_pose}", "RED")
        

    def play_once(self):
        arm_tag = ArmTag("left")
        self.move(
            self.grasp_actor(
                self.target_obj,
                arm_tag=arm_tag,
                pre_grasp_dis=self.GRASP_PRE_DIS,
                grasp_dis=self.GRASP_DIS
            )
        )
        self.attach_object(self.target_obj, f"{os.environ['ASSETS_ROOT']}/objects/{self.target_name}/collision/base{self.target_id}.glb", str(arm_tag))
        self.move(self.move_by_displacement(arm_tag, z = 0.04))

        self.move(
            self.place_actor(
                self.target_obj,
                arm_tag=arm_tag,
                target_pose= self.des_obj_pose,
                constrain= "auto",
                pre_dis=0.01,
                dis=0.002,
            ))
        self.move(self.move_by_displacement(arm_tag, z = 0.04))
        self.info["info"] = {
            "{A}": f"{self.target_name}/base{self.target_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        # was: unbounded half-plane (X within microwave width, Y anywhere in front, no Z) -> trivially
        # satisfiable and can pass at spawn. Ground on the intended 'in front' placement (des_obj_pose).
        tp = self.target_obj.get_pose().p
        dp = self.des_obj_pose
        eps_xy = 0.10  # NOTE(author): placement tolerance around the target; tune as needed
        return (abs(tp[0] - dp[0]) < eps_xy and abs(tp[1] - dp[1]) < eps_xy
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())

