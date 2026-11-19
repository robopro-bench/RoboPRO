import os

import yaml

from bench_envs.kitchenl._kitchen_base_large import Kitchen_base_large
from envs.utils import *
import math
import numpy as np
import sapien
import transforms3d as t3d


class put_milk_box_in_fridge(Kitchen_base_large):
    MILK_BOX_MASS = 0.1
    MILK_BOX_SPAWN_Z_OFFSET = 0.02

    # Success region bounds in fridge base-link local frame
    FRIDGE_SUCCESS_X_BOUNDS = (-0.33, 0.16)
    FRIDGE_SUCCESS_Y_BOUNDS = (-0.22, 0.22)
    FRIDGE_SUCCESS_Z_BOUNDS = (-0.34, 0.24)

    # Placement target in fridge base-link local frame
    FRIDGE_PLACE_LOCAL = np.array([-0.10, 0.00, 0.05], dtype=float)
    APPROACH_DELTA_1 = dict(x=0.1, y=0.30, z=-0.05)
    APPROACH_DELTA_2 = dict(y=0.12, z=0.02)
    RETREAT_DELTA = dict(y=-0.2)

    @staticmethod
    def _default_milk_box_contact_points(y_center: float) -> list:
        # Fallback contact-point set (4 side grasps) for assets lacking contact_points_pose metadata.
        return [
            [[6.123233995736766e-17, -1.0, 0.0, 0.0], [1.0, 6.123233995736766e-17, 0.0, y_center], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            [[6.123233995736766e-17, -6.123233995736766e-17, 1.0, 0.0], [1.0, 3.749399456654644e-33, -6.123233995736766e-17, y_center], [0.0, 1.0, 6.123233995736766e-17, 0.0], [0.0, 0.0, 0.0, 1.0]],
            [[6.123233995736766e-17, 1.0, 1.2246467991473532e-16, 0.0], [1.0, -6.123233995736766e-17, -7.498798913309288e-33, y_center], [0.0, 1.2246467991473532e-16, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            [[6.123233995736766e-17, -6.123233995736766e-17, -1.0, 0.0], [1.0, 3.749399456654644e-33, 6.123233995736766e-17, y_center], [0.0, -1.0, 6.123233995736766e-17, 0.0], [0.0, 0.0, 0.0, 1.0]],
        ]

    def _ensure_milk_box_grasp_metadata(self) -> None:
        if self.milk_box is None or not isinstance(self.milk_box.config, dict):
            return
        cfg = self.milk_box.config
        if "contact_points_pose" in cfg and isinstance(cfg["contact_points_pose"], list) and len(cfg["contact_points_pose"]) > 0:
            return
        y_center = float((cfg.get("center") or [0.0, 0.0, 0.0])[1])
        cfg["contact_points_pose"] = self._default_milk_box_contact_points(y_center)
        cfg.setdefault("contact_points_group", [list(range(len(cfg["contact_points_pose"])))])
        cfg.setdefault("contact_points_mask", [True])

    def _close_microwave_if_present(self) -> None:
        if getattr(self, "microwave_left", None) is None:
            return
        try:
            mw_qpos = np.array(self.microwave_left.get_qpos(), dtype=float)
            if mw_qpos.size > 0:
                mw_qpos[:] = 0.0
                self.microwave_left.set_qpos(mw_qpos)
        except Exception:
            pass

    def _get_target_object_names(self) -> set[str]:
        return {self.milk_box.get_name()}

    def setup_demo(self, is_test: bool = False, **kwargs):
        self.milk_box_modelname = "038_milk-box"
        kwargs["include_collision"] = True
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r", encoding="utf-8") as f:
            task_objs = yaml.safe_load(f)
        self.milk_box_model_ids = self._target_ids("kitchenl", self.milk_box_modelname)

        # Same Euler convention used in bottle tasks: [roll, pitch, yaw] in degrees.
        self.milk_box_spawn_rot_deg = [0.0, 0.0, 90.0]
        rot_cfg = kwargs.pop("milk_box_spawn_rot_deg", None)
        if rot_cfg is not None:
            self.milk_box_spawn_rot_deg = [float(rot_cfg[0]), float(rot_cfg[1]), float(rot_cfg[2])]

        # Optional scale multiplier on top of intrinsic model_data scale.
        self.milk_box_scale = 0.7
        ms = kwargs.pop("milk_box_scale", None)
        if ms is not None:
            self.milk_box_scale = float(ms)

        # Not used by the tabletop spawn below.
        self.milk_box_spawn_local_x_range = (-0.22, 0.12)
        self.milk_box_spawn_local_y_range = (-0.32, -0.06)

        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

        self._close_microwave_if_present()
        self.set_fridge_open()

    def _milk_box_quat_from_cfg(self) -> list[float]:
        roll_deg, pitch_deg, yaw_deg = self.milk_box_spawn_rot_deg
        ax = math.radians(roll_deg)
        ay = math.radians(pitch_deg)
        az = math.radians(yaw_deg)
        qx, qy, qz, qw = t3d.euler.euler2quat(ax, ay, az)
        return [qw, qx, qy, qz]

    def _table_center_spawn_pose(self, table_center: np.ndarray) -> sapien.Pose:
        # Randomized tabletop spawn around the table center.
        x = float(np.random.uniform(table_center[0] - 0.1, table_center[0] + 0.2))
        if self.scene_id == 1:
            ylim = [-0.15, 0.05]
        else:
            ylim = [-0.12, 0.1]
        y = float(np.random.uniform(ylim[0],ylim[1]))
        z = float(table_center[2] + self.MILK_BOX_SPAWN_Z_OFFSET)

        return sapien.Pose([x, y, z], self._milk_box_quat_from_cfg())

    def load_actors(self):
        if getattr(self, "fridge_closed_qpos", None) is None:
            self._init_fridge_states()
        self.set_fridge_open()

        table_center = np.array(self.table.get_pose().p, dtype=float)
        self.milk_box_model_id = int(np.random.choice(self.milk_box_model_ids))
        spawn_pose = self._table_center_spawn_pose(table_center)

        intrinsic_scale = self._get_asset_model_scale_create_actor(self.milk_box_modelname, self.milk_box_model_id)
        final_scale = float(intrinsic_scale) * float(self.milk_box_scale)

        self.milk_box = create_actor(
            scene=self.scene,
            pose=spawn_pose,
            modelname=self.milk_box_modelname,
            model_id=self.milk_box_model_id,
            is_static=False,
            convex=True,
            scale=final_scale,
        )
        if self.milk_box is not None:
            self.milk_box.set_mass(self.MILK_BOX_MASS)
            self.milk_box.set_name("task_milk_box")
            if isinstance(self.milk_box.config, dict):
                self.milk_box.config["scale"] = [final_scale] * 3
            self._ensure_milk_box_grasp_metadata()
            self.add_prohibit_area(self.milk_box, padding=0.04, area="table")

        if self.fridge_left is not None:
            self.add_prohibit_area(self.fridge_left, padding=0.1, area="table")

    def _fridge_inside_target_pose(self) -> list[float]:
        base_pose = self.fridge_left.get_link_pose("base_link")
        base_tf = base_pose.to_transformation_matrix()
        base_R = np.array(base_tf[:3, :3], dtype=float)
        base_p = np.array(base_tf[:3, 3], dtype=float)
        world_inside = base_p + base_R @ self.FRIDGE_PLACE_LOCAL
        milk_q = self.milk_box.get_pose().q.tolist()
        return world_inside.tolist() + milk_q

    def _is_milk_box_inside_fridge(self) -> bool:
        if self.milk_box is None or self.fridge_left is None:
            return False
        milk_world = np.array(self.milk_box.get_pose().p, dtype=float)
        base_pose = self.fridge_left.get_link_pose("base_link")
        inv_tf = np.linalg.inv(base_pose.to_transformation_matrix())
        milk_local_h = inv_tf @ np.array([milk_world[0], milk_world[1], milk_world[2], 1.0], dtype=float)
        x_l, y_l, z_l = milk_local_h[:3]
        x_ok = (self.FRIDGE_SUCCESS_X_BOUNDS[0] <= x_l <= self.FRIDGE_SUCCESS_X_BOUNDS[1])
        y_ok = (self.FRIDGE_SUCCESS_Y_BOUNDS[0] <= y_l <= self.FRIDGE_SUCCESS_Y_BOUNDS[1])
        z_ok = (self.FRIDGE_SUCCESS_Z_BOUNDS[0] <= z_l <= self.FRIDGE_SUCCESS_Z_BOUNDS[1])
        return bool(x_ok and y_ok and z_ok)

    def play_once(self):
        arm_tag = ArmTag("right")
        self.move(
            self.grasp_actor(
                self.milk_box,
                arm_tag=arm_tag,
                pre_grasp_dis=0.05,
                grasp_dis=0.0,
                gripper_pos=0.0,
            )
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.15))
        self.move(self.back_to_origin(arm_tag=arm_tag))

        # Same approach-to-fridge style used in bottle task.
        self.move(self.move_by_displacement(arm_tag=arm_tag, **self.APPROACH_DELTA_1))
        self.move(self.move_by_displacement(arm_tag=arm_tag, **self.APPROACH_DELTA_2))
        self.move(self.open_gripper(arm_tag=arm_tag, pos=1.0))

        self.info["info"] = {
            "{A}": f"{self.milk_box_modelname}/base{self.milk_box_model_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        return self._is_milk_box_inside_fridge() and self.robot.is_left_gripper_open() \
                and self.robot.is_right_gripper_open()
