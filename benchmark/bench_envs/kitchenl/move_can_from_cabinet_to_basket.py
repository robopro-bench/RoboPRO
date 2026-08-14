import os

import yaml

from bench_envs.kitchenl._kitchen_base_large import Kitchen_base_large
from bench_envs.utils.scene_gen_utils import get_actor_boundingbox, get_random_place_pose,print_c
from envs.utils import *
import math
import numpy as np
import sapien
import transforms3d as t3d


class move_can_from_cabinet_to_basket(Kitchen_base_large):
    CAN_MASS = 0.6
    IN_HAND_TCP_DIST_THRESHOLD = 0.18

    # Can spawn anchor in cabinet base-link local coordinates.
    CABINET_CAN_LOCAL = np.array([0.09, -0.21, -0.14], dtype=float)

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
    def _behind_side_can_contact_points(y_center: float) -> list:
        # Match put_can_in_cabinet style, but use the "behind" side contact.
        return [
            [[6.123233995736766e-17, -1.0, 0.0, 0.0], [1.0, 6.123233995736766e-17, 0.0, y_center], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        ]

    def _ensure_can_grasp_metadata(self) -> None:
        if self.can is None or not isinstance(self.can.config, dict):
            return
        cfg = self.can.config
        y_center = float((cfg.get("center") or [0.0, 0.0, 0.0])[1])
        cfg["contact_points_pose"] = self._behind_side_can_contact_points(y_center)
        cfg["contact_points_group"] = [[0]]
        cfg["contact_points_mask"] = [True]

    def _ensure_cabinet_open(self) -> None:
        if getattr(self, "cabinet_closed_qpos", None) is None:
            self._init_cabinet_states()
        self.set_cabinet_open()

    def _get_target_object_names(self) -> set[str]:
        return {self.can.get_name()}  # can: set_name("task_can")

    def setup_demo(self, is_test: bool = False, **kwargs):
        self.can_modelname = "071_can"
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r", encoding="utf-8") as f:
            task_objs = yaml.safe_load(f)
        self.can_model_ids = self._target_ids("kitchenl", self.can_modelname)
        # self.can_model_ids = [5]
        self.can_spawn_rot_deg = [90.0, 0.0, 90.0]

        rot_cfg = kwargs.pop("can_spawn_rot_deg", None)
        if rot_cfg is not None:
            self.can_spawn_rot_deg = [float(rot_cfg[0]), float(rot_cfg[1]), float(rot_cfg[2])]

        self.can_scale = 0.7
        cs = kwargs.pop("can_scale", None)
        if cs is not None:
            self.can_scale = float(cs)

        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

        self._ensure_cabinet_open()

    def _can_quat_from_cfg(self) -> list[float]:
        roll_deg, pitch_deg, yaw_deg = self.can_spawn_rot_deg
        ax = math.radians(roll_deg)
        ay = math.radians(pitch_deg)
        az = math.radians(yaw_deg)
        qx, qy, qz, qw = t3d.euler.euler2quat(ax, ay, az)
        return [qw, qx, qy, qz]

    def _cabinet_inside_spawn_pose(self) -> sapien.Pose:
        base_pose = self.cabinet.get_link_pose("base_link")
        base_tf = base_pose.to_transformation_matrix()
        base_R = np.array(base_tf[:3, :3], dtype=float)
        base_p = np.array(base_tf[:3, 3], dtype=float)
        world_inside = base_p + base_R @ self.CABINET_CAN_LOCAL

        # Current behavior: randomized spawn around the cabinet-local anchor.
        world_inside[0] += float(np.random.uniform(-0.05, 0.05))
        world_inside[1] += float(np.random.uniform(0.00, 0.10))

        return sapien.Pose(world_inside.tolist(), self._can_quat_from_cfg())

    def _can_local_in_cabinet(self) -> np.ndarray | None:
        if self.can is None or self.cabinet is None:
            return None
        can_world = np.array(self.can.get_pose().p, dtype=float)
        base_pose = self.cabinet.get_link_pose("base_link")
        inv_tf = np.linalg.inv(base_pose.to_transformation_matrix())
        can_local_h = inv_tf @ np.array([can_world[0], can_world[1], can_world[2], 1.0], dtype=float)
        return np.array(can_local_h[:3], dtype=float)

    def load_actors(self):
        self._ensure_cabinet_open()

        self.can_model_id = int(np.random.choice(self.can_model_ids))
        spawn_pose = self._cabinet_inside_spawn_pose()

        intrinsic_scale = self._get_asset_model_scale_create_actor(self.can_modelname, self.can_model_id)
        final_scale = float(intrinsic_scale) * float(self.can_scale)

        self.can = create_actor(
            scene=self.scene,
            pose=spawn_pose,
            modelname=self.can_modelname,
            model_id=self.can_model_id,
            is_static=False,
            convex=True,
            scale=0.06,
        )
        if self.can is not None:
            self.can.set_mass(self.CAN_MASS)
            self.can.set_name("task_can")
            if isinstance(self.can.config, dict):
                self.can.config["scale"] = [final_scale] * 3
            self._ensure_can_grasp_metadata()
            self.add_prohibit_area(self.can, padding=0.04, area="table")
       
        self.des_pose = get_random_place_pose(xlim = [-0.1, 0], ylim=[0] if self.scene_id == 1 else [-0.05,0.05], 
                                        col_thr=0.15,zlim=[0.75], qpos=(0,0,-90),
                                        object_bounds={})
        if self.scene_id != 1:
            self.add_prohibit_area(self.des_pose, padding=0.03, area="table")

        basket_bb = get_actor_boundingbox(self.basket_right.actor)
        self.des_basket_pose = sapien.Pose(
            [np.mean([basket_bb[0][0], basket_bb[1][0]]), 
             np.mean([basket_bb[0][1], basket_bb[1][1]]), 
             basket_bb[1][2] + 0.05], [1,0,0,0]
        )
    def _is_can_inside_cabinet(self) -> bool:
        can_local = self._can_local_in_cabinet()
        if can_local is None:
            return False
        x_l, y_l, z_l = can_local
        x_ok = (self.CABINET_X_BOUNDS[0] <= x_l <= self.CABINET_X_BOUNDS[1])
        y_ok = (self.CABINET_Y_BOUNDS[0] <= y_l <= self.CABINET_Y_BOUNDS[1])
        z_ok = (self.CABINET_Z_BOUNDS[0] <= z_l <= self.CABINET_Z_BOUNDS[1])
        return bool(x_ok and y_ok and z_ok)


    def take_can(self):
        arm_tag = ArmTag("right")
        self.move(self.move_by_displacement(arm_tag=arm_tag, **self.APPROACH_DELTA))
        self.move(
            self.grasp_actor(
                self.can,
                arm_tag=arm_tag,
                pre_grasp_dis=self.GRASP_PRE_DIS,
                grasp_dis=self.GRASP_DIS,
                gripper_pos=self.GRASP_CLOSE_POS,
                contact_point_id=self.GRASP_CONTACT_POINT_ID,
            )
        )
        self.attach_object(self.can, f"{os.environ['ASSETS_ROOT']}/objects/{self.can_modelname}/collision/base{self.can_model_id}.glb", str(arm_tag))

        # Lift is intentionally skipped because retreat already clears cabinet edge robustly.
        self.move(self.move_by_displacement(arm_tag=arm_tag, **self.RETREAT_DELTA))
        self.move(self.back_to_origin(arm_tag=arm_tag))


        self.move(
            self.place_actor(
                self.can,
                arm_tag=arm_tag,
                target_pose= self.des_pose,
                constrain="free",
                pre_dis=0.03,
                dis=0.005,
            ))
        self.move(self.open_gripper(arm_tag=arm_tag))
        self.move(self.move_by_displacement(arm_tag=arm_tag, y=-0.09))
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=0.05))

        self.move(self.back_to_origin(arm_tag=arm_tag))

        self.info["info"] = {
            "{A}": f"{self.can_modelname}/base{self.can_model_id}",
            "{a}": str(arm_tag),
        }
        return self.info
  
    def take_can_and_put(self):
        arm_tag = ArmTag("right")
        self.move(self.move_by_displacement(arm_tag=arm_tag, **self.APPROACH_DELTA))
        self.move(
            self.grasp_actor(
                self.can,
                arm_tag=arm_tag,
                pre_grasp_dis=self.GRASP_PRE_DIS,
                grasp_dis=self.GRASP_DIS,
                gripper_pos=self.GRASP_CLOSE_POS,
                contact_point_id=self.GRASP_CONTACT_POINT_ID,
            )
        )
        self.attach_object(self.can, f"{os.environ['ASSETS_ROOT']}/objects/{self.can_modelname}/collision/base{self.can_model_id}.glb", str(arm_tag))

        # Lift is intentionally skipped because retreat already clears cabinet edge robustly.
        self.move(self.move_by_displacement(arm_tag=arm_tag, **self.RETREAT_DELTA))
        self.move(self.back_to_origin(arm_tag=arm_tag))


        self.move(
            self.place_actor(
                self.can,
                arm_tag=arm_tag,
                target_pose= self.des_basket_pose,
                constrain="free",
                pre_dis=0.03,
                dis=0.005,
            ))
        self.move(self.open_gripper(arm_tag=arm_tag))
        self.move(self.back_to_origin(arm_tag=arm_tag))

        self.info["info"] = {
            "{A}": f"{self.can_modelname}/base{self.can_model_id}",
            "{a}": str(arm_tag),
        }
        return self.info
  
    def close_cabinet(self):
        # Provide a simple info mapping for downstream use
        arm_tag = ArmTag("right")
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=0.2, y=0.05))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.35))
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=-0.3, y=0.07))
        self.move(self.move_by_displacement(arm_tag=arm_tag, y=0.05))
        self.move(self.back_to_origin(arm_tag=arm_tag))
        self.info["info"] = {
            "{A}": "122_cabinet_nkrgez",
        }
        return self.info
    
    def put_can_in_basket_left(self):
        arm_tag = ArmTag("left")
        self.move(
            self.grasp_actor(
                self.can,
                arm_tag=arm_tag,
                pre_grasp_dis=self.GRASP_PRE_DIS
            )
        )

        self.attach_object(self.can, f"{os.environ['ASSETS_ROOT']}/objects/{self.can_modelname}/collision/base{self.can_model_id}.glb", str(arm_tag))
        if self.scene_id == 2:
            self.move(self.move_by_displacement(arm_tag=arm_tag, x = -0.2,y=-0.3, z=0.2))
        if self.scene_id == 1:
            self.move(self.move_by_displacement(arm_tag=arm_tag, x = -0.1,y=-0.2, z=0.1))

        self.move(
            self.place_actor(
                self.can,
                arm_tag=arm_tag,
                target_pose= self.des_basket_pose,
                constrain="auto",
                pre_dis=0.07,
                dis=0.005,
            ))
        self.info["info"] = {
            "{A}": f"{self.can_modelname}/base{self.can_model_id}",
            "{a}": str(arm_tag),
        }
    def _is_can_inside_basket(self) -> bool:
        """True only when the can lies inside the basket's 3D bounding volume.

        Previously this compared XY only, so a can merely hovering above the basket
        (e.g. still held by the gripper on approach) satisfied it. Including Z means
        the can must actually have descended into the basket.
        """
        box_bb = get_actor_boundingbox(self.basket_right.actor)
        can_p = np.asarray(self.can.get_pose().p, dtype=float)
        lo = np.asarray(box_bb[0], dtype=float)
        hi = np.asarray(box_bb[1], dtype=float)
        return bool(np.all((lo <= can_p) & (can_p <= hi)))

    def play_once(self):
        if self.scene_id == 1:
            self.take_can_and_put()
            self.close_cabinet()
        else:
            self.take_can()
            self.close_cabinet()
            self.put_can_in_basket_left()

    def check_success(self):
        # NOTE: `self._is_can_inside_basket` (no call) used to be passed here as a bound
        # method object, which is always truthy -- the basket check never ran. The task
        # therefore scored success for any run that merely got the can out of the cabinet
        # and closed the door, even with the can left on the table.
        return not self._is_can_inside_cabinet() and self._is_can_inside_basket() \
               and self.is_cabinet_closed(threshold=0.02) \
               and self.robot.is_right_gripper_open() \
               and self.robot.is_left_gripper_open()


