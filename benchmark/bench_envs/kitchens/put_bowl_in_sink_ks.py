from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
import os
import numpy as np
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class put_bowl_in_sink_ks(KitchenS_base_task):

    # Mirrors put_spoon_in_sink_ks: grasp bowl from counter with the
    # side-grasp recipe, then place in the sink with constrain="free".
    # Sink is always at sink_x >= +0.10, so the bowl spawns on the right
    # half of the counter and the right arm does the full carry.

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        # Keep the entire bowl footprint clear of the sink. The old fixed
        # x-range [0.05, 0.30] has no valid point when the sink is at x=0.10:
        # the sink prohibition ends at x=0.26 and the bowl uses 8 cm footprint
        # padding, so its centre must be x>0.34. The sampler consequently
        # exhausted all attempts and returned a forbidden last sample; many
        # bowls fell into the sink before the demonstration began.
        obj_padding = 0.08
        sink_p = self.sink.get_pose().p
        sink_geom = self.kitchens_info["sink_geom"]
        sink_pad = 0.03  # must match _load_sink's prohibited-area padding
        clearance = sink_geom["hole_hx"] + sink_pad + obj_padding + 0.01
        counter_left = (
            self.kitchens_info["table_lims"][0] + self.table_xy_bias[0]
        )
        counter_right = (
            self.kitchens_info["table_lims"][2] + self.table_xy_bias[0]
        )

        # Stay on the positive-x/right-arm half. Prefer the side of the sink
        # with a usable interval; scene 1 uses the right interval, scenes 0/2
        # use the left one. Cap the right edge at the proven bowl-grasp workspace.
        right_xlim = [
            float(sink_p[0]) + clearance,
            min(
                0.48 + self.table_xy_bias[0],
                counter_right - obj_padding,
            ),
        ]
        left_xlim = [
            max(
                0.05 + self.table_xy_bias[0],
                counter_left + obj_padding,
            ),
            float(sink_p[0]) - clearance,
        ]

        if right_xlim[1] - right_xlim[0] >= 0.03:
            spawn_xlim = right_xlim
        elif left_xlim[1] - left_xlim[0] >= 0.03:
            spawn_xlim = left_xlim
        else:
            raise RuntimeError(
                f"No sink-clear bowl spawn interval: "
                f"sink_x={float(sink_p[0]):.3f}, "
                f"left={left_xlim}, right={right_xlim}"
            )

        rand_pos = self.rand_pose_on_counter(
            xlim=spawn_xlim,
            ylim=[-0.20, -0.14],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
            obj_padding=obj_padding,
        )

        self.bowl_id = 3
        self.target_obj = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="002_bowl",
            convex=True,
            model_id=self.bowl_id,
        )
        self.target_obj.set_mass(0.05)

        self.add_prohibit_area(self.target_obj, padding=0.02, area="table")

    def play_once(self):
        arm_tag = ArmTag("right")
        self._target_start_p = np.asarray(
            self.target_obj.get_pose().p,
            dtype=float,
        ).copy()

        self.grasp_actor_from_table(
            self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.10,
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))

        self.attach_object(
            self.target_obj,
            f"{os.environ['ASSETS_ROOT']}/objects/002_bowl/collision/base{self.bowl_id}.glb",
            str(arm_tag),
        )
        self.enable_table(enable=True)

        # place_actor(constrain="free") offsets pre_dis along bowl's LOCAL
        # approach axis — for a side-grasped bowl that tilts into the sink
        # walls and trips INVALID_PARTIAL_POSE_COST_METRIC. Use the working
        # dishrack recipe: two-stage move_to_pose with INIT_Q (front-facing
        # home quat, IK-reachable from side-grasp config) then open gripper.
        INIT_Q = [0.707, 0, 0, 0.707]
        sink_p = self.sink.get_pose().p
        sg = self.kitchens_info["sink_geom"]
        # Drop toward the outer (robot-side) half of the sink basin.
        # Robot is at y<0, so "closer to robot" = negative y offset.
        drop_x = float(sink_p[0]) + 0.02
        drop_y = float(sink_p[1]) - 1.15 * sg["hole_hy"]
        hover_pose = [drop_x, drop_y, float(sink_p[2]) + 0.25] + INIT_Q
        self.move(self.move_to_pose(arm_tag, hover_pose))
        drop_pose = [drop_x, drop_y, float(sink_p[2]) + 0.10] + INIT_Q
        self.move(self.move_to_pose(arm_tag, drop_pose))
        self.move(self.open_gripper(arm_tag, pos=1.0))

    def check_success(self):
        sink_p = self.sink.get_pose().p
        sg = self.kitchens_info["sink_geom"]
        tp = self.target_obj.get_pose().p
        start_p = getattr(self, "_target_start_p", None)
        moved = (
            start_p is not None
            and np.linalg.norm(
                np.asarray(tp, dtype=float) - start_p
            ) > 0.10
        )
        in_xy = (abs(tp[0] - sink_p[0]) < sg["hole_hx"]
                 and abs(tp[1] - sink_p[1]) < sg["hole_hy"])
        below_rim = tp[2] < sink_p[2] + 0.02
        return (moved and in_xy and below_rim
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
