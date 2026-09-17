import os

from bench_envs.kitchenl._kitchen_base_large import Kitchen_base_large
from bench_envs.utils.scene_gen_utils import get_actor_boundingbox, get_random_place_pose,print_c
from envs.utils import *
import math
import numpy as np
import sapien
import transforms3d as t3d


class pick_can_from_basket(Kitchen_base_large):
    CAN_MASS = 0.1
    CAN_MODELNAME = "071_can"
    CAN_MODEL_IDS = [0]

    CAN_SPAWN_Z_OFFSET = 0.02
    BASKET_CAN_LOCAL = np.array([0.0, 0.035, 0.03], dtype=float)

    PLACE_WORLD_X_OFFSET = 0.08
    PLACE_WORLD_Y_OFFSET = -0.08
    PLACE_SUCCESS_X_TOL = 0.2
    PLACE_SUCCESS_Y_TOL = 0.2
    TABLE_SURFACE_Z_BOUNDS = (-0.08, 0.35)
    IN_HAND_TCP_DIST_THRESHOLD = 0.18

    GRASP_PRE_DIS = 0.06
    GRASP_DIS = 0.0
    GRASP_CLOSE_POS = 0.0
    GRASP_CONTACT_POINT_ID = 0
    PLACE_HEIGHT_ABOVE_TABLE = 0.14
    DESCEND_BEFORE_RELEASE = 0.1
    RETREAT_AFTER_RELEASE = dict(z=0.12, y=-0.12)

    @staticmethod
    def _behind_side_can_contact_points(y_center: float) -> list:
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

    def _get_target_object_names(self) -> set[str]:
        return {self.can.get_name()}

    def setup_demo(self, is_test: bool = False, **kwargs):
        kwargs["scene_id"] = 0 # Only use scene 0 for this task, to ensure the cabinet is in the same location and the same door is open across all demos.
        self.can_modelname = self.CAN_MODELNAME
        self.can_model_ids = list(self.CAN_MODEL_IDS)
        self.can_spawn_rot_deg = [90.0, -90.0, 90.0]

        rot_cfg = kwargs.pop("can_spawn_rot_deg", None)
        if rot_cfg is not None:
            self.can_spawn_rot_deg = [float(rot_cfg[0]), float(rot_cfg[1]), float(rot_cfg[2])]

        self.can_scale = 0.7
        cs = kwargs.pop("can_scale", None)
        if cs is not None:
            self.can_scale = float(cs)

        mids = kwargs.pop("can_model_ids", None)
        if mids is not None:
            self.can_model_ids = [int(x) for x in mids]

        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _can_quat_from_cfg(self) -> list[float]:
        roll_deg, pitch_deg, yaw_deg = self.can_spawn_rot_deg
        ax = math.radians(roll_deg)
        ay = math.radians(pitch_deg)
        az = math.radians(yaw_deg)
        qx, qy, qz, qw = t3d.euler.euler2quat(ax, ay, az)
        return [qw, qx, qy, qz]

    def _basket_spawn_pose(self) -> sapien.Pose:
        basket_pose = self.basket_right.get_pose()
        basket_tf = basket_pose.to_transformation_matrix()
        basket_R = np.array(basket_tf[:3, :3], dtype=float)
        basket_p = np.array(basket_tf[:3, 3], dtype=float)
        world_pos = basket_p + basket_R @ np.array(self.BASKET_CAN_LOCAL, dtype=float)
        return sapien.Pose(world_pos.tolist(), self._can_quat_from_cfg())

    def _is_can_inside_basket(self) -> bool:
        box_bb = get_actor_boundingbox(self.basket_right.actor)
        return np.all((box_bb[0][:2] <= self.can.get_pose().p[:2])  & 
                       (self.can.get_pose().p[:2] <= box_bb[1][:2]))

    def load_actors(self):
        self.can_model_id = int(np.random.choice(self.can_model_ids))
        spawn_pose = self._basket_spawn_pose()

        intrinsic_scale = self._get_asset_model_scale_create_actor(self.can_modelname, self.can_model_id)
        final_scale = float(intrinsic_scale) * float(self.can_scale)

        self.can = create_actor(
            scene=self.scene,
            pose=spawn_pose,
            modelname=self.can_modelname,
            model_id=self.can_model_id,
            is_static=False,
            convex=True,
            scale=final_scale,
        )
        if self.can is not None:
            self.can.set_mass(self.CAN_MASS)
            self.can.set_name("task_can")
            if isinstance(self.can.config, dict):
                self.can.config["scale"] = [final_scale] * 3
            self._ensure_can_grasp_metadata()
            self.add_prohibit_area(self.can, padding=0.04, area="table")
        self.des_pose = get_random_place_pose(xlim = [-0.25, -0.15], ylim=[-0.05,0],
                                        col_thr=0.15,zlim=[0.80], qpos=(0,0,0),
                                        object_bounds={})
                                        
        self.add_prohibit_area(self.des_pose, padding=0.0, area="table")
        print_c(f"Place destination {self.des_pose}", "RED")

    def play_once(self):
        arm_tag = ArmTag("left")
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

        self.move(self.back_to_origin(arm_tag=arm_tag))
        self.add_collision()
        self.update_world()
        self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=self.des_pose))
        self.move(self.open_gripper(arm_tag=arm_tag))


        self.info["info"] = {
            "{A}": f"{self.can_modelname}/base{self.can_model_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        eps = 0.01
        b_pose = self.can.get_pose().p
        table_bb = get_actor_boundingbox(self.table)
        can_on_table = np.all((table_bb[0][:2] <= b_pose[:2])  &  (b_pose[:2] <= table_bb[1][:2]))
        can_on_table &= (b_pose[-1] - table_bb[1][-1]) < eps  
        
        return not self._is_can_inside_basket() and can_on_table \
               and self.robot.is_right_gripper_open() \
               and self.robot.is_left_gripper_open()


