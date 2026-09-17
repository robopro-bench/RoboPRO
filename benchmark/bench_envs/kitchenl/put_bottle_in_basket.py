import yaml

from bench_envs.kitchenl._kitchen_base_large import Kitchen_base_large
from bench_envs.utils.scene_gen_utils import get_actor_boundingbox
from envs.utils import *
import os
import math
import numpy as np
import sapien
import transforms3d as t3d


class put_bottle_in_basket(Kitchen_base_large):
    BOTTLE_MASS = 0.1
    BOTTLE_SPAWN_Z_OFFSET = 0.01
    BOTTLE_WORLD_XY_JITTER = 0.025

    # Basket interior bounds in basket local frame.
    BASKET_X_BOUNDS = (-0.20, 0.20)
    BASKET_Y_BOUNDS = (-0.20, 0.20)
    BASKET_Z_BOUNDS = (-0.10, 0.25)

    # Left-arm motion tuning.
    APPROACH_DELTA_1 = dict(x=-0.05, y=0.29, z=0.10)
    APPROACH_DELTA_2 = dict(z=-0.13)
    RETREAT_DELTA = dict(y=-0.20)
    GRASP_PRE_DIS = 0.07
    GRASP_DIS = 0.0
    GRASP_CLOSE_POS = 0.0
    GRASP_CONTACT_POINT_ID = 1

    def _get_target_object_names(self) -> set[str]:
        return {self.bottle.get_name()}

    def setup_demo(self, is_test: bool = False, **kwargs):

        # Match bottle asset setup used in pick_bottle_from_fridge.
        self.bottle_modelname = "001_bottle"
        kwargs["include_collision"] = True
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r", encoding="utf-8") as f:
            task_objs = yaml.safe_load(f)

        self.bottle_model_ids =  self._target_ids("kitchenl", self.bottle_modelname)
        self.bottle_spawn_rot_deg = [-45.0, 0.0, 90.0]

        rot_cfg = kwargs.pop("bottle_spawn_rot_deg", None)
        if rot_cfg is not None:
            self.bottle_spawn_rot_deg = [float(rot_cfg[0]), float(rot_cfg[1]), float(rot_cfg[2])]

        self.bottle_scale = 0.7
        bs = kwargs.pop("bottle_scale", None)
        if bs is not None:
            self.bottle_scale = float(bs)

        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _bottle_quat_from_cfg(self) -> list[float]:
        roll_deg, pitch_deg, yaw_deg = self.bottle_spawn_rot_deg
        ax = math.radians(roll_deg)
        ay = math.radians(pitch_deg)
        az = math.radians(yaw_deg)
        qx, qy, qz, qw = t3d.euler.euler2quat(ax, ay, az)
        return [qw, qx, qy, qz]

    def _table_center_spawn_pose(self, table_center: np.ndarray) -> sapien.Pose:
        # Spawn near table center with small world-frame jitter.
        x = float(np.random.uniform(table_center[0] - 0.2, table_center[0] + 0.05))
        if self.scene_id == 1:
            y = float(np.random.uniform(table_center[1] - 0.15, table_center[1] + 0.05))
        else:  
            y = float(np.random.uniform(table_center[1] - 0.15, table_center[1] + 0))
        z = float(table_center[2] + self.BOTTLE_SPAWN_Z_OFFSET)
        return sapien.Pose([x, y, z], self._bottle_quat_from_cfg())

    def load_actors(self):
        table_center = np.array(self.table.get_pose().p, dtype=float)
        self.bottle_model_id = int(np.random.choice(self.bottle_model_ids))
        spawn_pose = self._table_center_spawn_pose(table_center)

        intrinsic_scale = self._get_asset_model_scale_create_actor(self.bottle_modelname, self.bottle_model_id)
        final_scale = float(intrinsic_scale) * float(self.bottle_scale)

        self.bottle = create_actor(
            scene=self.scene,
            pose=spawn_pose,
            modelname=self.bottle_modelname,
            model_id=self.bottle_model_id,
            is_static=False,
            convex=True,
            scale=final_scale,
        )
        if self.bottle is not None:
            self.bottle.set_mass(self.BOTTLE_MASS)
            self.bottle.set_name("task_bottle")
            if isinstance(self.bottle.config, dict):
                self.bottle.config["scale"] = [final_scale] * 3
            self.add_prohibit_area(self.bottle, padding=0.04, area="table")

        basket_bb = get_actor_boundingbox(self.basket_right.actor)
        self.des_pose = sapien.Pose(
            [np.mean([basket_bb[0][0], basket_bb[1][0]]), 
             np.mean([basket_bb[0][1], basket_bb[1][1]]), 
             basket_bb[1][2] + 0.05],
            [1, 0, 0, 0]
        )
    def _is_can_inside_basket(self) -> bool:
        box_bb = get_actor_boundingbox(self.basket_right.actor)
        return np.all((box_bb[0][:2] <= self.bottle.get_pose().p[:2])  & 
                       (self.bottle.get_pose().p[:2] <= box_bb[1][:2]))

    def play_once(self):
        arm_tag = ArmTag("left")
        self.move(
            self.grasp_actor(
                self.bottle,
                arm_tag=arm_tag,
                pre_grasp_dis=self.GRASP_PRE_DIS,
                grasp_dis=self.GRASP_DIS,
                gripper_pos=self.GRASP_CLOSE_POS,
                contact_point_id=self.GRASP_CONTACT_POINT_ID,
            )
        )

        self.attach_object(self.bottle, f"{os.environ['ASSETS_ROOT']}/objects/{self.bottle_modelname}/collision/base{self.bottle_model_id}.glb", str(arm_tag))
        
        if self.scene_id == 2:
            self.move(self.move_by_displacement(arm_tag=arm_tag, x = -0.2, y=-0.2, z=0.2))
        if self.scene_id == 1:
            self.move(self.move_by_displacement(arm_tag=arm_tag, x = -0.1, y=-0.1, z=0.1))

        self.move(
            self.place_actor(
                self.bottle,
                arm_tag=arm_tag,
                target_pose= self.des_pose,
                constrain="auto",
                pre_dis=0.07,
                dis=0.005,
            ))
        self.info["info"] = {
            "{A}": f"{self.bottle_modelname}/base{self.bottle_model_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        return self._is_can_inside_basket() and self.robot.is_left_gripper_open() \
                and self.robot.is_right_gripper_open()
