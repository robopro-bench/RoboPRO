import os

import yaml

from bench_envs.kitchenl._kitchen_base_large import Kitchen_base_large
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob
import numpy as np
import transforms3d as t3d


class put_can_close_cabinet(Kitchen_base_large):
    """
    Task: start with the cabinet doors open and close them.
    """
    CAN_MASS = 0.1
    CAN_SPAWN_Z_OFFSET = 0.02

    # Cabinet interior bounds in cabinet base-link local frame.
    CABINET_SUCCESS_X_BOUNDS = (-0.30, 0.20)
    CABINET_SUCCESS_Y_BOUNDS = (-0.22, 0.22)
    CABINET_SUCCESS_Z_BOUNDS = (-0.30, 0.25)

    APPROACH_DELTA_1 = dict(x=-0.05, y=0.15, z=0.35)
    APPROACH_DELTA_2 = dict(y=0.15, z=0.02)
    RETREAT_DELTA = dict(y=-0.15)
    GRASP_CONTACT_POINT_ID = 0

    def _get_target_object_names(self) -> set[str]:
        return {self.can.get_name()}  # create_actor("071_can") → name = "071_can"

    def setup_demo(self, is_test: bool = False, **kwargs):
        kwargs["include_collision"] = True
        # Match collision-cache usage from other benchmark tasks
        self.can_modelname = "071_can"
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r", encoding="utf-8") as f:
            task_objs = yaml.safe_load(f)
        self.can_model_ids = self._target_ids("kitchenl", self.can_modelname)
        self.can_spawn_rot_deg = [90.0, 0.0, 90.0]

        rot_cfg = kwargs.pop("can_spawn_rot_deg", None)
        if rot_cfg is not None:
            self.can_spawn_rot_deg = [float(rot_cfg[0]), float(rot_cfg[1]), float(rot_cfg[2])]

        self.can_scale = 1
        cs = kwargs.pop("can_scale", None)
        if cs is not None:
            self.can_scale = float(cs)
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

        # Capture the closed configuration before opening
        if hasattr(self, "cabinet") and self.cabinet is not None:
            self._init_cabinet_states()
            # Start close-cabinet task from right-door-open state only.
            self.set_cabinet_open()
        else:
            self.cabinet_closed_qpos = None
    def _can_quat_from_cfg(self) -> list[float]:
        roll_deg, pitch_deg, yaw_deg = self.can_spawn_rot_deg
        ax = math.radians(roll_deg)
        ay = math.radians(pitch_deg)
        az = math.radians(yaw_deg)
        qx, qy, qz, qw = t3d.euler.euler2quat(ax, ay, az)
        return [qw, qx, qy, qz]

    def _table_center_spawn_pose(self, table_center: np.ndarray) -> sapien.Pose:
        # Current behavior: sampled tabletop spawn near table center.
        x = float(np.random.uniform(table_center[0], table_center[0] + 0.4))
        y = float(np.random.uniform(table_center[1] - 0.075, table_center[1]))
        z = float(table_center[2] + self.CAN_SPAWN_Z_OFFSET)
        return sapien.Pose([x, y, z], self._can_quat_from_cfg())

    def load_actors(self):

        table_center = np.array(self.table.get_pose().p, dtype=float)
        self.can_model_id = int(np.random.choice(self.can_model_ids))
        spawn_pose = self._table_center_spawn_pose(table_center)

        self.can = create_actor(
            scene=self.scene,
            pose=spawn_pose,
            modelname=self.can_modelname,
            model_id=self.can_model_id,
            is_static=False,
            convex=True,
            scale=None,
        )
        self.can.set_mass(self.CAN_MASS)

        self.add_prohibit_area(self.can, padding=0.04, area="table")
   
    def _is_can_inside_cabinet(self) -> bool:
        if self.can is None or self.cabinet is None:
            return False
        can_world = np.array(self.can.get_pose().p, dtype=float)
        base_pose = self.cabinet.get_link_pose("base_link")
        inv_tf = np.linalg.inv(base_pose.to_transformation_matrix())
        can_local_h = inv_tf @ np.array([can_world[0], can_world[1], can_world[2], 1.0], dtype=float)
        x_l, y_l, z_l = can_local_h[:3]
        x_ok = (self.CABINET_SUCCESS_X_BOUNDS[0] <= x_l <= self.CABINET_SUCCESS_X_BOUNDS[1])
        y_ok = (self.CABINET_SUCCESS_Y_BOUNDS[0] <= y_l <= self.CABINET_SUCCESS_Y_BOUNDS[1])
        z_ok = (self.CABINET_SUCCESS_Z_BOUNDS[0] <= z_l <= self.CABINET_SUCCESS_Z_BOUNDS[1])
        return bool(x_ok and y_ok and z_ok)

    def put_can(self):
        arm_tag = ArmTag("right")
        self.move(
            self.grasp_actor(
                self.can,
                arm_tag=arm_tag,
                pre_grasp_dis=0.06,
                grasp_dis=0.0,
                gripper_pos=0.0,
                contact_point_id=self.GRASP_CONTACT_POINT_ID,
            )
        )
        self.attach_object(self.can, f"{os.environ['ASSETS_ROOT']}/objects/{self.can_modelname}/collision/base{self.can_model_id}.glb", str(arm_tag))

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.15))
        self.move(self.back_to_origin(arm_tag=arm_tag))

        self.move(self.move_by_displacement(arm_tag=arm_tag, **self.APPROACH_DELTA_1))
        self.move(self.move_by_displacement(arm_tag=arm_tag, **self.APPROACH_DELTA_2))
        self.move(self.open_gripper(arm_tag=arm_tag, pos=1.0))
        self.move(self.back_to_origin(arm_tag=arm_tag))

        self.info["info"] = {
            "{A}": f"{self.can_modelname}/base{self.can_model_id}",
            "{a}": str(arm_tag),
        }
    def close_cabinet(self):
        # Provide a simple info mapping for downstream use
        arm_tag = ArmTag("right")
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=0.2, y=0.05))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.35))
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=-0.3, y=0.07))
        self.move(self.move_by_displacement(arm_tag=arm_tag, y=0.05))
    
        self.info["info"] = {
            "{A}": "122_cabinet_nkrgez",
        }
        return self.info
    def play_once(self):
        self.put_can()
        self.close_cabinet()

    def check_success(self):
         return self.is_cabinet_closed(threshold=0.02) and self._is_can_inside_cabinet() and \
                self.robot.is_left_gripper_open() \
                and self.robot.is_right_gripper_open()

