import os

import yaml

from bench_envs.kitchenl._kitchen_base_large import Kitchen_base_large
from bench_envs.utils.scene_gen_utils import get_actor_boundingbox, place_actor, point_to_box_distance, print_c
from envs.utils import *
import math
import numpy as np
import sapien
import transforms3d as t3d


class put_can_next_to_basket(Kitchen_base_large):
    MILK_BOX_MASS = 0.1
    MILK_BOX_SPAWN_Z_OFFSET = 0
    TABLE_WORLD_XY_JITTER = 0.05

    # Basket interior bounds in basket local frame.
    BASKET_X_BOUNDS = (-0.20, 0.20)
    BASKET_Y_BOUNDS = (-0.20, 0.20)
    BASKET_Z_BOUNDS = (-0.10, 0.25)

    # Success region multiplier around basket bounds.
    BASKET_EXPANSION_RATIO = 1

    # Left-arm motion tuning.
    APPROACH_DELTA_1 = dict(x=-0.22, y=0.22, z=-0.14)
    RETREAT_DELTA = dict(y=-0.07, z=0.02)
    GRASP_PRE_DIS = 0.07
    GRASP_DIS = 0.01
    GRASP_CLOSE_POS = 0.0,

    # Must stay larger than SUCCESS_DIST_THR so spawn cannot already succeed.
    SUCCESS_DIST_THR = 0.20
    SPAWN_GAP = 0.25
    # Fall detector, not a placement-precision gate: a can knocked onto its side
    # still lands inside the 20cm halo and used to score success. Measured on
    # rollouts a settled can sits at 0.0 deg and a fallen one at ~90 deg, so the
    # exact value is uncontentious; 45 deg keeps a can leaning on the basket
    # passing. Yaw is unconstrained by construction (see check_success).
    EPS_TILT_DEG = 45.0

    def _get_target_object_names(self) -> set[str]:
        return {self.can.get_name()}  # place_actor("071_can") → name = "071_can"

    def setup_demo(self, is_test: bool = False, **kwargs):
        self.can_modelname = "071_can"
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r", encoding="utf-8") as f:
            task_objs = yaml.safe_load(f)
        self.can_model_ids = self._target_ids("kitchenl", self.can_modelname)
        np.random.seed(kwargs.get("seed", 0))
        # Scene 2 sits the basket on the microwave (z=0.927), so "next to" on
        # the table is not well-defined. Vary table-level layouts like the
        # other next-to tasks (original intent was choice([0, 1])).
        if kwargs.get("scene_id") is None:
            kwargs["scene_id"] = int(np.random.choice([0, 1]))
        kwargs["include_collision"] = True
        self.can_box_spawn_rot_deg = [0.45, 0.0, 90.0]
        kwargs["jitter_basket"] = False
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)


    def load_actors(self):
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r", encoding="utf-8") as f:
            task_objs = yaml.safe_load(f)
        box_bb = get_actor_boundingbox(self.basket_right.actor)
        gap = self.SPAWN_GAP
        basket_cx = 0.5 * (float(box_bb[0][0]) + float(box_bb[1][0]))
        # Spawn on the opposite side of the basket so the episode is not
        # already in the success halo (same idea as study next-to place_gap).
        if basket_cx <= 0:
            xlim = [float(box_bb[1][0]) + gap, float(box_bb[1][0]) + gap + 0.12]
        else:
            xlim = [float(box_bb[0][0]) - gap - 0.12, float(box_bb[0][0]) - gap]
        xlim = [float(np.clip(xlim[0], -0.45, 0.15)), float(np.clip(xlim[1], -0.45, 0.15))]
        if xlim[1] < xlim[0]:
            xlim[0], xlim[1] = xlim[1], xlim[0]

        self.can, self.can_model_id, self.target_pose = \
        place_actor(self.can_modelname, self, col_thr=gap, xlim=xlim, ylim=[-0.08, -0.02],
                    qpos=(90,0,0), object_bounds=[box_bb], task_objs=task_objs,
                     mass = 0.2, rotation=False, scene_name='kitchenl')

        # Reference orientation for the fall check. Taken from the spawn pose
        # rather than assuming a fixed local axis: the can is placed with
        # qpos=(90,0,0), so its local +z is not world-up.
        self._can_init_quat = np.array(self.can.get_pose().q, dtype=np.float64)

        self.add_prohibit_area(self.can, padding=0.04, area="table")
        x_place = float(box_bb[0][0]) + np.random.uniform(low=-0.02, high=0.02)
        y_place = np.random.uniform(low=float(box_bb[0][1]) - 0.1, high=float(box_bb[0][1]) - 0.05)
        table_z = float(self.table.get_pose().p[2])
        self.des_obj_pose = [x_place, y_place, table_z] + [1, 0, 0, 0]

        self.add_prohibit_area(self.des_obj_pose, padding=0.03, area="table")

        print_c(f"Placement destination pose {self.des_obj_pose}", "RED")

        dist = point_to_box_distance(self.can.get_pose().p, box_bb[0], box_bb[1])
        if dist < self.SUCCESS_DIST_THR:
            raise RuntimeError(
                f"Can spawned inside the basket success region (dist={dist:.3f} < {self.SUCCESS_DIST_THR})."
            )
        

    def play_once(self):
        arm_tag = ArmTag("left")
        self.move(
            self.grasp_actor(
                self.can,
                arm_tag=arm_tag,
                pre_grasp_dis=self.GRASP_PRE_DIS,
                grasp_dis=self.GRASP_DIS
                # contact_point_id= self.contact_id
                # gripper_pos=self.GRASP_CLOSE_POS,
            )
        )
        self.attach_object(self.can, f"{os.environ['ASSETS_ROOT']}/objects/{self.can_modelname}/collision/base{self.can_model_id}.glb", str(arm_tag))
        lift = 0.15
        if self.scene_id == 0:
            self.move(self.move_by_displacement(arm_tag, z = lift, y= -lift, x= lift))
        else:
            self.move(self.move_by_displacement(arm_tag, z = lift, y= -lift, x = -lift))
        self.move(
            self.place_actor(
                self.can,
                arm_tag=arm_tag,
                target_pose= self.des_obj_pose,
                constrain= "auto",
                pre_dis=0.02,
                dis=0.003,
            ))
        self.move(self.move_by_displacement(arm_tag, z = 0.04))
        self.info["info"] = {
            "{A}": f"{self.can_modelname}/base{self.can_model_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        pose = self.can.get_pose()
        box_bb = get_actor_boundingbox(self.basket_right.actor)
        dist_to_box = point_to_box_distance(pose.p, box_bb[0], box_bb[1])

        # UPRIGHT: a can toppled by the arm rolls and still lands inside the
        # 20cm halo, which used to score success. Track the local axis that
        # pointed world-up at spawn and require it to stay near world-up. A pure
        # yaw leaves that axis untouched, so spinning the can about z passes;
        # only tipping fails. Same convention as put_milktea_next_to_laptop.
        R0 = t3d.quaternions.quat2mat(self._can_init_quat)
        Rn = t3d.quaternions.quat2mat(np.array(pose.q, dtype=np.float64))
        up_local = R0.T @ np.array([0.0, 0.0, 1.0])
        upright = (Rn @ up_local)[2] > np.cos(np.deg2rad(self.EPS_TILT_DEG))

        return bool(dist_to_box < self.SUCCESS_DIST_THR
                    and upright
                    and self.robot.is_left_gripper_open()
                    and self.robot.is_right_gripper_open())

