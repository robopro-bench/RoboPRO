import os

import yaml

from bench_envs.kitchenl._kitchen_base_large import Kitchen_base_large
from bench_envs.utils.scene_gen_utils import get_actor_boundingbox_urdf, place_actor, print_c
from envs.utils import *
import math
import numpy as np
import sapien
import transforms3d as t3d


class put_bottle_in_fridge(Kitchen_base_large):
    BOTTLE_MASS = 0.2
    BOTTLE_SPAWN_Z_OFFSET = 0.04
    FRIDGE_X_BOUNDS = (-0.33, 0.16)
    FRIDGE_Y_BOUNDS = (-0.22, 0.22)
    FRIDGE_Z_BOUNDS = (-0.34, 0.24)
    # Placement target in fridge base-link local frame
    FRIDGE_PLACE_LOCAL = np.array([-0.10, 0.00, 0.05], dtype=float)

    def _get_target_object_names(self) -> set[str]:
        return {self.bottle.get_name()}

    def setup_demo(self, is_test: bool = False, **kwargs):
        kwargs["include_collision"] = False
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)
        self.set_fridge_open()

    def load_actors(self):
        if getattr(self, "fridge_closed_qpos", None) is None:
            self._init_fridge_states()

        self.set_fridge_open()    

        self.bottle_modelname = "001_bottle"

        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r", encoding="utf-8") as f:
            task_objs = yaml.safe_load(f)


        self.bottle, self.bottle_model_id, self.target_pose = \
        place_actor(self.bottle_modelname, self, col_thr=0.10, xlim=[0,0.35], ylim=[-0.15,-0.06], 
                    qpos=(90,0,0), object_bounds=[], task_objs=task_objs,
                     mass = 0.2, rotation=False, scene_name="kitchenl")

        self.add_prohibit_area(self.bottle, padding=0.04, area="table")


    def _is_bottle_inside_fridge(self) -> bool:
        if self.bottle is None or self.fridge_left is None:
            return False
        bottle_world = np.array(self.bottle.get_pose().p, dtype=float)
        base_pose = self.fridge_left.get_link_pose("base_link")
        base_tf = base_pose.to_transformation_matrix()
        inv_tf = np.linalg.inv(base_tf)
        bottle_local_h = inv_tf @ np.array([bottle_world[0], bottle_world[1], bottle_world[2], 1.0], dtype=float)
        x_l, y_l, z_l = bottle_local_h[:3]
        x_ok = (self.FRIDGE_X_BOUNDS[0] <= x_l <= self.FRIDGE_X_BOUNDS[1])
        y_ok = (self.FRIDGE_Y_BOUNDS[0] <= y_l <= self.FRIDGE_Y_BOUNDS[1])
        z_ok = (self.FRIDGE_Z_BOUNDS[0] <= z_l <= self.FRIDGE_Z_BOUNDS[1])
        return bool(x_ok and y_ok and z_ok)

    def play_once(self):
        arm_tag = ArmTag("right")
        # Close slightly less than "fully closed" to reduce penetration impulse.
        self.move(
            self.grasp_actor(
                self.bottle,
                arm_tag=arm_tag,
                pre_grasp_dis=0.07,
                grasp_dis=0.01,
                gripper_pos=0.01,
            )
        )
        self.attach_object(self.bottle, f"{os.environ['ASSETS_ROOT']}/objects/{self.bottle_modelname}/collision/base{self.bottle_model_id}.glb", str(arm_tag))

        self.move(self.move_by_displacement(arm_tag=arm_tag, x=0.1, z=0.05))
        self.move(self.back_to_origin(arm_tag=arm_tag))

        self.move(self.move_by_displacement(arm_tag=arm_tag, x=0.1, y=0.30, z=-0.03))
        self.move(self.move_by_displacement(arm_tag=arm_tag, y=0.1))
        self.move(self.open_gripper(arm_tag=arm_tag, pos=1.0))


        self.info["info"] = {
            "{A}": f"{self.bottle_modelname}/base{self.bottle_model_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        return self._is_bottle_inside_fridge() and self.robot.is_left_gripper_open() \
                and self.robot.is_right_gripper_open()
