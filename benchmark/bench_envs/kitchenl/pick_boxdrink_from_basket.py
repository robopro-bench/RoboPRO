import os

from bench_envs.kitchenl._kitchen_base_large import Kitchen_base_large
from bench_envs.utils.scene_gen_utils import get_random_place_pose, get_actor_boundingbox, print_c
from envs.utils import *
import math
import numpy as np
import sapien
import transforms3d as t3d


class pick_boxdrink_from_basket(Kitchen_base_large):
    BOXDRINK_MASS = 0.1
    BOXDRINK_MODELNAME = "068_boxdrink"
    BOXDRINK_MODEL_IDS = [3]

    # Debug-only: table spawn (swap spawn line in load_actors).
    BOXDRINK_SPAWN_Z_OFFSET = 0.02
    TABLE_WORLD_XY_JITTER = 0.05

    # Basket mesh origin is on the rim (local Y=0); interior center is ~0.065.
    # +Y puts the drink in the bowl (same convention as pick_can_from_basket).
    BASKET_BOXDRINK_LOCAL = np.array([0.0, 0.05, 0.03], dtype=float)

    PLACE_WORLD_X_OFFSET = 0.08
    PLACE_WORLD_X_JITTER = (-0.04, 0.04)
    PLACE_WORLD_Y_OFFSET = -0.08
    PLACE_WORLD_Y_JITTER = (-0.04, 0.04)
    PLACE_SUCCESS_XY_TOL = 0.12
    TABLE_PLACE_Z_BOUNDS = (-0.02, 0.22)

    GRASP_PRE_DIS = 0.05
    GRASP_DIS = 0.0
    GRASP_CLOSE_POS = 0.0
    PLACE_HEIGHT_ABOVE_TABLE = 0.20
    DESCEND_BEFORE_RELEASE = 0.06
    RETREAT_AFTER_RELEASE = dict(z=0.12, y=-0.12)

    @staticmethod
    def _default_boxdrink_contact_points(y_center: float) -> list:
        return [
            [[6.123233995736766e-17, -1.0, 0.0, 0.0], [1.0, 6.123233995736766e-17, 0.0, y_center], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            [[6.123233995736766e-17, -6.123233995736766e-17, 1.0, 0.0], [1.0, 3.749399456654644e-33, -6.123233995736766e-17, y_center], [0.0, 1.0, 6.123233995736766e-17, 0.0], [0.0, 0.0, 0.0, 1.0]],
            [[6.123233995736766e-17, 1.0, 1.2246467991473532e-16, 0.0], [1.0, -6.123233995736766e-17, -7.498798913309288e-33, y_center], [0.0, 1.2246467991473532e-16, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            [[6.123233995736766e-17, -6.123233995736766e-17, -1.0, 0.0], [1.0, 3.749399456654644e-33, 6.123233995736766e-17, y_center], [0.0, -1.0, 6.123233995736766e-17, 0.0], [0.0, 0.0, 0.0, 1.0]],
        ]

    @staticmethod
    def _world_point_in_entity_local(entity, world_xyz: np.ndarray) -> np.ndarray:
        inv_tf = np.linalg.inv(entity.get_pose().to_transformation_matrix())
        h = inv_tf @ np.array([world_xyz[0], world_xyz[1], world_xyz[2], 1.0], dtype=float)
        return np.array(h[:3], dtype=float)

    def _ensure_boxdrink_grasp_metadata(self) -> None:
        if self.boxdrink is None or not isinstance(self.boxdrink.config, dict):
            return
        cfg = self.boxdrink.config
        if "contact_points_pose" in cfg and isinstance(cfg["contact_points_pose"], list) and len(cfg["contact_points_pose"]) > 0:
            return
        y_center = float((cfg.get("center") or [0.0, 0.0, 0.0])[1])
        cfg["contact_points_pose"] = self._default_boxdrink_contact_points(y_center)
        cfg.setdefault("contact_points_group", [list(range(len(cfg["contact_points_pose"])))])
        cfg.setdefault("contact_points_mask", [True])

    def _get_target_object_names(self) -> set[str]:
        return {self.boxdrink.get_name()}

    def setup_demo(self, is_test: bool = False, **kwargs):
        self.boxdrink_modelname = self.BOXDRINK_MODELNAME
        self.boxdrink_model_ids = list(self.BOXDRINK_MODEL_IDS)
        # Stand the carton (mesh Y-up -> world Z). [180,0,90] laid it on its side.
        self.boxdrink_spawn_rot_deg = [90.0, 0.0, 90.0]

        rot_cfg = kwargs.pop("boxdrink_spawn_rot_deg", None)
        if rot_cfg is not None:
            self.boxdrink_spawn_rot_deg = [float(rot_cfg[0]), float(rot_cfg[1]), float(rot_cfg[2])]

        self.boxdrink_scale = 1.1
        bs = kwargs.pop("boxdrink_scale", None)
        if bs is not None:
            self.boxdrink_scale = float(bs)

        mids = kwargs.pop("boxdrink_model_ids", None)
        if mids is not None:
            self.boxdrink_model_ids = [int(x) for x in mids]

        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _boxdrink_quat_from_cfg(self) -> list[float]:
        roll_deg, pitch_deg, yaw_deg = self.boxdrink_spawn_rot_deg
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
        world_pos = basket_p + basket_R @ np.array(self.BASKET_BOXDRINK_LOCAL, dtype=float)
        return sapien.Pose(world_pos.tolist(), self._boxdrink_quat_from_cfg())

    def _is_boxdrink_inside_basket(self) -> bool:
        box_bb = get_actor_boundingbox(self.basket_right.actor)
        return np.all((box_bb[0][:2] <= self.boxdrink.get_pose().p[:2])  & 
                       (self.boxdrink.get_pose().p[:2] <= box_bb[1][:2]))
       
        
    def _sample_place_world_offsets(self) -> None:
        self._place_world_x_off = float(self.PLACE_WORLD_X_OFFSET) + float(
            np.random.uniform(self.PLACE_WORLD_X_JITTER[0], self.PLACE_WORLD_X_JITTER[1])
        )
        self._place_world_y_off = float(self.PLACE_WORLD_Y_OFFSET) + float(
            np.random.uniform(self.PLACE_WORLD_Y_JITTER[0], self.PLACE_WORLD_Y_JITTER[1])
        )

    def _place_target_world_xy(self) -> np.ndarray:
        p = np.array(self.table.get_pose().p, dtype=float)
        return np.array([p[0] + self._place_world_x_off, p[1] + self._place_world_y_off], dtype=float)

    def _ee_pose_above_place_target(self, arm_tag: ArmTag) -> np.ndarray:
        ee_pose = np.array(self.get_arm_pose(arm_tag), dtype=float)
        table_p = np.array(self.table.get_pose().p, dtype=float)
        xy_w = self._place_target_world_xy()
        target = ee_pose.copy()
        target[0] = float(xy_w[0])
        target[1] = float(xy_w[1])
        target[2] = float(table_p[2] + self.PLACE_HEIGHT_ABOVE_TABLE)
        return target

    def load_actors(self):
        self._sample_place_world_offsets()

        self.boxdrink_model_id = int(np.random.choice(self.boxdrink_model_ids))
        intrinsic_scale = self._get_asset_model_scale_create_actor(self.boxdrink_modelname, self.boxdrink_model_id)
        final_scale = float(intrinsic_scale) * float(self.boxdrink_scale)

        spawn_pose = self._basket_spawn_pose()

        self.boxdrink = create_actor(
            scene=self.scene,
            pose=spawn_pose,
            modelname=self.boxdrink_modelname,
            model_id=self.boxdrink_model_id,
            is_static=False,
            convex=True,
            scale=final_scale,
        )
        
        
        if self.boxdrink is not None:
            self.boxdrink.set_mass(self.BOXDRINK_MASS)
            self.boxdrink.set_name("task_boxdrink")
            if isinstance(self.boxdrink.config, dict):
                self.boxdrink.config["scale"] = [final_scale] * 3
            self._ensure_boxdrink_grasp_metadata()
            self.add_prohibit_area(self.boxdrink, padding=0.04, area="table")

            self.des_pose = get_random_place_pose(xlim = [-0.25, -0.15], ylim=[-0.05, 0],
                                        col_thr=0.15,zlim=[0.80], qpos=(0,0,0),
                                        object_bounds={})
            self.add_prohibit_area(self.des_pose, padding=0.0, area="table")

    def play_once(self):
        arm_tag = ArmTag("left")
        self.move(
            self.grasp_actor(
                self.boxdrink,
                arm_tag=arm_tag,
                pre_grasp_dis=self.GRASP_PRE_DIS,
                grasp_dis=self.GRASP_DIS,
                gripper_pos=self.GRASP_CLOSE_POS,
            )
        )
        self.attach_object(self.boxdrink, f"{os.environ['ASSETS_ROOT']}/objects/{self.boxdrink_modelname}/collision/base{self.boxdrink_model_id}.glb", str(arm_tag))

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.15))
        self.move(self.back_to_origin(arm_tag=arm_tag))
        self.add_collision()
        self.update_world()
        self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=self.des_pose))
        self.move(self.open_gripper(arm_tag=arm_tag))
     
        self.info["info"] = {
            "{A}": f"{self.boxdrink_modelname}/base{self.boxdrink_model_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        eps = 0.01
        table_bb = get_actor_boundingbox(self.table)
        on_table = np.all((table_bb[0][:2] <= self.boxdrink.get_pose().p[:2])  &  (self.boxdrink.get_pose().p[:2] <= table_bb[1][:2]))
        on_table &= (self.boxdrink.get_pose().p[-1] - table_bb[1][-1]) < eps  
        return on_table and (not self._is_boxdrink_inside_basket())
