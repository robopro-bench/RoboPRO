import os

import yaml

from bench_envs.kitchenl._kitchen_base_large import Kitchen_base_large
from envs.utils import *
import math
import numpy as np
import sapien
import transforms3d as t3d


class move_milk_close_fridge(Kitchen_base_large):
    BOTTLE_MASS = 0.1
    FRIDGE_WORLD_XY_JITTER = 0.025

    # Bottle spawn anchor in fridge base-link local coordinates.
    # Note: local axes map to world via world = base_p + base_R @ local.
    FRIDGE_BOTTLE_LOCAL = np.array([0.125, -0.17, 0.00], dtype=float)

    # Fridge interior bounds in fridge base-link local frame
    FRIDGE_X_BOUNDS = (-0.33, 0.16)
    FRIDGE_Y_BOUNDS = (-0.22, 0.22)
    FRIDGE_Z_BOUNDS = (-0.34, 0.24)
    IN_HAND_TCP_DIST_THRESHOLD = 0.18

    # Retrieval trajectory tuning
    APPROACH_DELTA = dict(x=0.1, y=0.10, z=-0.05)
    LIFT_DELTA = dict(z=0.05)
    RETREAT_DELTA = dict(y=-0.1)
    GRASP_PRE_DIS = 0.07
    GRASP_DIS = 0.01
    GRASP_CONTACT_POINT_ID = 1
    GRASP_CLOSE_POS = 0.0

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
        return {self.milk_box.get_name()}  # rand_create_cluttered_actor("038_milk-box") → name = "038_milk-box"

    def setup_demo(self, is_test: bool = False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)
        self.fridge_start_open_angle_deg = 90.0
        self.set_fridge_open_angle_deg(self.fridge_start_open_angle_deg, open_span_deg=90.0)

    def _milk_box_local_in_fridge(self) -> np.ndarray | None:
        if self.milk_box is None or self.fridge_left is None:
            return None
        bottle_world = np.array(self.milk_box.get_pose().p, dtype=float)
        base_pose = self.fridge_left.get_link_pose("base_link")
        inv_tf = np.linalg.inv(base_pose.to_transformation_matrix())
        bottle_local_h = inv_tf @ np.array([bottle_world[0], bottle_world[1], bottle_world[2], 1.0], dtype=float)
        return np.array(bottle_local_h[:3], dtype=float)

    def load_actors(self):
        self.obj_name = "038_milk-box"
        self.add_prohibit_area(self.fridge_left.get_pose(), padding=[0.15,0.5], area="table")
        self.cuboid_collision_list.append({"name": "table", "dims": [1.2, 0.7, 0.002], "pose": [0,0,0.74,1,0,0,0]})
        self._init_fridge_states()

        self.milk_box_model_id = 0
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r", encoding="utf-8") as f:
            task_objs = yaml.safe_load(f)
        
        scale = task_objs["scales"][self.obj_name][f"{self.milk_box_model_id}"]
        if self.fridge_left.get_pose().p[0] > 0:
            xlim = [-0.15,0.55]
        else:
            xlim = [-0.55,0.15]
        success, self.milk_box = rand_create_cluttered_actor(
            scene=self.scene,
            xlim=xlim,
            ylim=[-0.25,0.15],
            zlim=[0.74],
            modelname=self.obj_name,
            scale=scale,
            modelid=self.milk_box_model_id,
            modeltype="glb",
            rotate_rand=True,
            rotate_lim=[0, np.pi/10, 0],
            qpos=[0.7071, 0.7071, 0, 0],
            obj_radius=0.03,
            z_offset=0,
            z_max=0.05,
            prohibited_area=self.prohibited_area["table"],
            constrained=False,
            is_static=False,
            size_dict=dict(),
        )
        if not success:
            raise RuntimeError("Failed to milkbox")
        self._ensure_milk_box_grasp_metadata()
        self.add_prohibit_area(self.milk_box, padding=0.04, area="table")
        self.add_operating_area(self.milk_box.get_pose().p)
        self.target_obj_pose = self.milk_box.get_pose()
        self.collision_list.append({
            "actor": self.milk_box,
            "collision_path": f"{os.environ['ASSETS_ROOT']}/objects/{self.obj_name}/collision/base{self.milk_box_model_id}.glb",
            "pose": self.target_obj_pose
        })
        self.target_obj_pose = np.concatenate([self.target_obj_pose.p, self.target_obj_pose.q]).tolist()

        # des_obj ------------------------------------------------------------
        half_size = [0.04, 0.04, 0.0005]
        p = [0.382, 0.189, 0.8]

        des_obj_pose = sapien.Pose(p=p, q=[1, 0, 0, 0])
        self.des_obj = create_box(
            scene=self,
            pose=des_obj_pose,
            half_size=half_size,
            color=(0, 0, 1),
            name="box",
            is_static=True,
        )
        self.add_prohibit_area(self.des_obj, padding=0.03, area="fridge")

        self.des_obj_pose = self.des_obj.get_pose().p.tolist() + [0.7071, 0.7071, 0, 0]
        self.des_obj_pose[2] += 0.02

    def _is_milk_box_inside_fridge(self) -> bool:
        milk_box_local = self._milk_box_local_in_fridge()
        if milk_box_local is None:
            return False
        x_l, y_l, z_l = milk_box_local
        x_ok = (self.FRIDGE_X_BOUNDS[0] <= x_l <= self.FRIDGE_X_BOUNDS[1])
        y_ok = (self.FRIDGE_Y_BOUNDS[0] <= y_l <= self.FRIDGE_Y_BOUNDS[1])
        z_ok = (self.FRIDGE_Z_BOUNDS[0] <= z_l <= self.FRIDGE_Z_BOUNDS[1])
        return bool(x_ok and y_ok and z_ok)

    def play_once(self):
        arm_tag = ArmTag("right")
        # initial_tcp_quat = np.array(self.get_arm_pose(arm_tag), dtype=float)[3:].tolist()

        # 1. Move straight to the handle and close the gripper
        # self.move(self.move_by_displacement(arm_tag=arm_tag, x=-0.05, y=0.255, z=0.02))
        # self.move(self.close_gripper(arm_tag=arm_tag, pos=0.0))

        # # 2. Execute the circular pull
        # # Tune the door_radius here to match the physical fridge
        # self.pull_door_circularly(
        #     arm_tag=arm_tag, 
        #     door_radius=0.23,
        #     total_open_angle=45.0, 
        #     num_steps=30
        # )

        # self.move(self.close_gripper(arm_tag=arm_tag, pos=1.0))
        # self.move(self.move_by_displacement(arm_tag=arm_tag, x=0.03, y=-0.1))
        # self.move(self.move_by_displacement(arm_tag=arm_tag, x=-0.2))
        # self.move(self.move_by_displacement(arm_tag=arm_tag, y=0.2))
        # self.move(self.move_by_displacement(arm_tag=arm_tag, quat=initial_tcp_quat))
        # self.move(self.move_by_displacement(arm_tag=arm_tag, x=0.21, y=-0.2))

        # 3. grab bottle
        # self.move(self.move_by_displacement(arm_tag=arm_tag, **self.APPROACH_DELTA))
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=-0.1))
        _, actions = self.grasp_actor(
                self.milk_box,
                arm_tag=arm_tag,
                pre_grasp_dis=0.06,
                grasp_dis=0.02,
                gripper_pos=0.0,
                contact_point_id=1,
            )
        actions[0].target_pose[2] += 0.03
        actions[1].target_pose[2] += 0.03

        self.move((arm_tag, [actions[0]]))
        self.enable_obstacle(False, mesh_names=[f"038_milk-box_{self.target_obj_pose}_{self.seed}"])
        self.move((arm_tag, actions[1:]))

        # 4. place bottle
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.03))
        
        self.attach_object(self.milk_box, f"{os.environ['ASSETS_ROOT']}/objects/038_milk-box/collision/base{self.milk_box_model_id}.glb", str(arm_tag))
        self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=sapien.Pose(p=[0.3,-0.15,0.95], q=[0.7071,0,0,0.7071])))
        # self.move(self.back_to_origin(arm_tag=arm_tag))

        self.move(
            self.place_actor(
                self.milk_box,
                arm_tag=arm_tag,
                target_pose=self.des_obj_pose,
                constrain="free",
                pre_dis=0,
                dis=0,
                local_up_axis=[0,0,1]
            )
        )

        # 5. close fridge
        self.move(self.back_to_origin(arm_tag=arm_tag))
        self.detach_object(arms_tag=arm_tag)
        self.move(self.move_by_displacement(arm_tag=arm_tag, y=-0.20))
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=+0.30))
        self.move(self.move_by_displacement(arm_tag=arm_tag, y=+0.30))
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=-0.30, y=+0.10))
        self.move(self.move_by_displacement(arm_tag=arm_tag, y=+0.10))

        self.info["info"] = {
            "{A}": f"{self.obj_name}/base{self.milk_box_model_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        return self._is_milk_box_inside_fridge() \
               and self.is_fridge_closed()
