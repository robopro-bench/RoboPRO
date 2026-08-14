from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
import os
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class put_spoon_in_dishrack_ks(KitchenS_base_task):

    # Spoon pick: grasp_actor_from_table side-grasp (from the old
    # move_spoon_next_to_plate_ks recipe). Drop: two-stage move_to_pose
    # + open_gripper above the dishrack basin (from place_bowl_in_dishrack_ks).

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        rack_p = self.dishrack.get_pose().p
        self.arm_tag = ArmTag("right" if rack_p[0] > 0 else "left")
        side_sign = 1 if self.arm_tag == "right" else -1

        rack_x = float(rack_p[0])
        if side_sign > 0:  # right arm
            if rack_x > 0.35:
                x_range = [0.10, 0.24]
            else:
                x_range = [0.38, 0.48]
        else:  # left arm
            x_range = [-0.22, -0.08]

        rand_pos = self.rand_pose_on_counter(
            xlim=x_range,
            ylim=[-0.22, 0.00],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, 3.14, 0],
            obj_padding=0.06,
        )

        self.spoon_id = 0
        self.target_obj = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="134_spoon",
            convex=True,
            model_id=self.spoon_id,
        )
        self.target_obj.set_mass(0.05)

        self.add_prohibit_area(self.target_obj, padding=0.02, area="table")

    def play_once(self):
        arm_tag = self.arm_tag

        self.grasp_actor_from_table(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.10)
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))
        self.attach_object(
            self.target_obj,
            f"{os.environ.get('BENCH_ROOT', os.environ['SIM_ROOT'] + '/../benchmark')}/assets/objects/134_spoon/collision/base{self.spoon_id}.glb",
            str(arm_tag),
        )

        # Pop rack from curobo world so planner isn't blocked by overlap.
        self.collision_list = [
            e for e in self.collision_list if e.get("actor") is not self.dishrack
        ]
        self.update_world()

        # Two-stage drop over rack basin, INIT_Q front-facing wrist.
        INIT_Q = [0.707, 0, 0, 0.707]
        rack_p = self.dishrack.get_pose().p
        drop_x = float(rack_p[0])
        drop_y = float(rack_p[1]) - 0.35

        hover_drop_pose = [drop_x, drop_y - 0.20, 1.20] + INIT_Q
        self.move(self.move_to_pose(arm_tag, hover_drop_pose))

        drop_pose = [drop_x, drop_y, 1.00] + INIT_Q
        self.move(self.move_to_pose(arm_tag, drop_pose))
        self.move(self.open_gripper(arm_tag, pos=1.0))

    def check_success(self):
        rack_p = self.dishrack_center  # true rack center, not the offset entity pose
        tp = self.target_obj.get_pose().p
        eps_x, eps_y = 0.12, 0.10  # tightened Y toward the true rack half-extent (~0.08)
        return (abs(tp[0] - rack_p[0]) < eps_x
                and abs(tp[1] - rack_p[1]) < eps_y
                # elevated ONTO the rack, not resting on the counter beside it (rack_z ~0.89 vs table ~0.74)
                and tp[2] > rack_p[2] - 0.08
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
