import os

import yaml

from bench_envs.kitchenl._kitchen_base_large import Kitchen_base_large
from bench_envs.utils.scene_gen_utils import get_actor_boundingbox, get_random_place_pose
from envs.utils import *
import math
import numpy as np
import sapien
import transforms3d as t3d


class pick_sauce_can_from_cabinet(Kitchen_base_large):
    SAUCE_CAN_MASS = 0.1
    IN_HAND_TCP_DIST_THRESHOLD = 0.18

    # Sauce-can spawn anchor in cabinet base-link local coordinates.
    CABINET_SAUCE_CAN_LOCAL = np.array([0.09, -0.21, -0.14], dtype=float)

    # Cabinet interior bounds in cabinet base-link local frame.
    CABINET_X_BOUNDS = (-0.30, 0.20)
    CABINET_Y_BOUNDS = (-0.22, 0.22)
    CABINET_Z_BOUNDS = (-0.30, 0.25)

    # Retrieval trajectory tuning.
    APPROACH_DELTA = dict(x=-0.05, y=0.15, z=0.35)
    RETREAT_DELTA = dict(y=-0.18, z=0.06)
    GRASP_PRE_DIS = 0.07
    GRASP_DIS = 0.01
    GRASP_CLOSE_POS = 0.0
    GRASP_CONTACT_POINT_ID = 0

    @staticmethod
    def _behind_side_contact_points(y_center: float) -> list:
        # Keep same grasp side style as pick_can_from_cabinet.
        return [
            [[6.123233995736766e-17, -1.0, 0.0, 0.0], [1.0, 6.123233995736766e-17, 0.0, y_center], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        ]

    def _ensure_sauce_can_grasp_metadata(self) -> None:
        if self.sauce_can is None or not isinstance(self.sauce_can.config, dict):
            return
        cfg = self.sauce_can.config
        y_center = float((cfg.get("center") or [0.0, 0.0, 0.0])[1])
        cfg["contact_points_pose"] = self._behind_side_contact_points(y_center)
        cfg["contact_points_group"] = [[0]]
        cfg["contact_points_mask"] = [True]

    def _ensure_cabinet_open(self) -> None:
        if getattr(self, "cabinet_closed_qpos", None) is None:
            self._init_cabinet_states()
        self.set_cabinet_open()

    def _get_target_object_names(self) -> set[str]:
        return {self.sauce_can.get_name()}

    def setup_demo(self, is_test: bool = False, **kwargs):
        self.sauce_can_modelname = "105_sauce-can"
        kwargs["include_collision"] = True 
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r", encoding="utf-8") as f:
            task_objs = yaml.safe_load(f)


        self.sauce_can_model_ids =  self._target_ids("kitchenl", self.sauce_can_modelname)
        self.sauce_can_spawn_rot_deg = [90.0, 0.0, 90.0]

        rot_cfg = kwargs.pop("sauce_can_spawn_rot_deg", None)
        if rot_cfg is not None:
            self.sauce_can_spawn_rot_deg = [float(rot_cfg[0]), float(rot_cfg[1]), float(rot_cfg[2])]

        self.sauce_can_scale = 1
        scs = kwargs.pop("sauce_can_scale", None)
        if scs is not None:
            self.sauce_can_scale = float(scs)

        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)
        self._ensure_cabinet_open()

    def _sauce_can_local_in_cabinet(self) -> np.ndarray | None:
        if self.sauce_can is None or self.cabinet is None:
            return None
        sauce_world = np.array(self.sauce_can.get_pose().p, dtype=float)
        base_pose = self.cabinet.get_link_pose("base_link")
        inv_tf = np.linalg.inv(base_pose.to_transformation_matrix())
        sauce_local_h = inv_tf @ np.array([sauce_world[0], sauce_world[1], sauce_world[2], 1.0], dtype=float)
        return np.array(sauce_local_h[:3], dtype=float)

    def load_actors(self):
        self._ensure_cabinet_open()

        self.sauce_can_model_id = int(np.random.choice(self.sauce_can_model_ids))
        spawn_pose = sapien.Pose([0.206523, 0.0804209, 1.23062], [0.5, 0.5, 0.5, 0.5])
        intrinsic_scale = self._get_asset_model_scale_create_actor(self.sauce_can_modelname, self.sauce_can_model_id)
        final_scale = float(intrinsic_scale) * float(self.sauce_can_scale)

        self.sauce_can = create_actor(
            scene=self.scene,
            pose=spawn_pose,
            modelname=self.sauce_can_modelname,
            model_id=self.sauce_can_model_id,
            is_static=False,
            convex=True,
            scale=final_scale,
        )
        if self.sauce_can is not None:
            self.sauce_can.set_mass(self.SAUCE_CAN_MASS)
            self.sauce_can.set_name("task_sauce_can")
            if isinstance(self.sauce_can.config, dict):
                self.sauce_can.config["scale"] = [final_scale] * 3
            self._ensure_sauce_can_grasp_metadata()
            self.add_prohibit_area(self.sauce_can, padding=0.04, area="table")
            if self.scene_id == 1:
                ylim = [-0.15, 0]
            else:
                ylim = [-0.12, 0.1]
        self.des_pose = get_random_place_pose(xlim = [-0.1, 0.25], ylim=ylim,
                                              col_thr=0.15,zlim=[0.78],
                                        object_bounds={})
        
        self.add_prohibit_area(self.des_pose, padding=0.03, area="table")

    def _is_sauce_can_inside_cabinet(self) -> bool:
        sauce_local = self._sauce_can_local_in_cabinet()
        if sauce_local is None:
            return False
        x_l, y_l, z_l = sauce_local
        x_ok = (self.CABINET_X_BOUNDS[0] <= x_l <= self.CABINET_X_BOUNDS[1])
        y_ok = (self.CABINET_Y_BOUNDS[0] <= y_l <= self.CABINET_Y_BOUNDS[1])
        z_ok = (self.CABINET_Z_BOUNDS[0] <= z_l <= self.CABINET_Z_BOUNDS[1])
        return bool(x_ok and y_ok and z_ok)

    def play_once(self):
        arm_tag = ArmTag("right")
        self.move(self.move_by_displacement(arm_tag=arm_tag, **self.APPROACH_DELTA))
        self.move(
            self.grasp_actor(
                self.sauce_can,
                arm_tag=arm_tag,
                pre_grasp_dis=self.GRASP_PRE_DIS,
                grasp_dis=self.GRASP_DIS,
                gripper_pos=self.GRASP_CLOSE_POS,
                contact_point_id=self.GRASP_CONTACT_POINT_ID,
            )
        )
        self.attach_object(self.sauce_can, f"{os.environ['ASSETS_ROOT']}/objects/{self.sauce_can_modelname}/collision/base{self.sauce_can_model_id}.glb", str(arm_tag))

        self.move(self.move_by_displacement(arm_tag=arm_tag, **self.RETREAT_DELTA))
        # self.move(self.back_to_origin(arm_tag=arm_tag))
        self.add_collision(objects=("cabinet"))
        self.update_world()
        self.move(
            self.place_actor(
                self.sauce_can,
                arm_tag=arm_tag,
                target_pose= self.des_pose,
                constrain="auto",
                pre_dis=0.07,
                dis=0.005,
            ))
        self.info["info"] = {
            "{A}": f"{self.sauce_can_modelname}/base{self.sauce_can_model_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        eps = 0.01
        b_pose = self.sauce_can.get_pose().p
        table_bb = get_actor_boundingbox(self.table)
        sauce_can_on_table = np.all((table_bb[0][:2] <= b_pose[:2])  &  (b_pose[:2] <= table_bb[1][:2]))
        sauce_can_on_table &= (b_pose[-1] - table_bb[1][-1]) < eps  
    
        return not self._is_sauce_can_inside_cabinet() and sauce_can_on_table \
               and self.robot.is_right_gripper_open() \
               and self.robot.is_left_gripper_open()
