from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
import os
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class pick_apple_from_bowl_ks(KitchenS_base_task):

    # Scripted top-down grasp. 035_apple's labeled contacts all encode
    # side-grasp orientations; inside a bowl cavity those IK solutions
    # are narrow/infeasible. A scripted top-down descent bypasses that.
    # For aloha: input pose maps directly to end-link (gripper_bias=0.12,
    # delta_matrix=identity). This quat makes link +x point in world -z.
    TOP_DOWN_Q = [-0.5, 0.5, -0.5, -0.5]
    # TCP = link_pos + link_x * 0.12; for top-down wrist, link_z = tcp_z + 0.12.
    TCP_OFFSET = 0.12

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        bowl_pose = self.rand_pose_on_counter(
            xlim=[-0.32, 0.32],
            ylim=[-0.23, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, np.pi / 4, 0],
            obj_padding=0.12,
        )
        # Force the bowl off-center so one arm is the obvious picker.
        while abs(bowl_pose.p[0]) < 0.3:
            bowl_pose = self.rand_pose_on_counter(
                xlim=[-0.32, 0.32],
                ylim=[-0.23, 0.05],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 4, 0],
                obj_padding=0.12,
            )

        # 002_bowl variants 3/4/5 are the large ones (~17 cm diameter,
        # 5–7 cm tall). Shallow rim + wide opening gives the gripper
        # plenty of clearance around a ~6.6 cm apple.
        self.bowl_id = int(np.random.choice([3, 4, 5]))
        self.bowl = create_actor(
            scene=self,
            pose=bowl_pose,
            modelname="002_bowl",
            convex=True,
            model_id=self.bowl_id,
            is_static=True,
        )
        self.bowl.set_name("bowl")
        self.add_prohibit_area(self.bowl, padding=0.02, area="table")

        bp = self.bowl.get_pose().p
        ax = bp[0] + np.random.uniform(-0.03, 0.03)
        ay = bp[1] + np.random.uniform(-0.03, 0.03)
        # Apple center just above the bowl rim — gravity settles it in.
        az = bp[2] + 0.05
        apple_pose = sapien.Pose([ax, ay, az], [0.5, 0.5, 0.5, 0.5])

        self.apple_id = int(np.random.choice([0, 1]))
        self.target_obj = create_actor(
            scene=self,
            pose=apple_pose,
            modelname="035_apple",
            convex=True,
            model_id=self.apple_id,
        )
        self.target_obj.set_mass(0.05)

        self.arm_tag = ArmTag("right" if bp[0] > 0 else "left")
        self.lift_z = bp[2] + 0.30

        # Counter drop pose on the SAME arm side as the bowl, but in the
        # inner half of that side (bowl occupies |x|>=0.3, so [0.05, 0.22]
        # is clear and reachable by the same arm).
        side_sign = 1 if self.arm_tag == "right" else -1
        target_rand_pose = self.rand_pose_on_counter(
            xlim=[0.05, 0.22] if side_sign > 0 else [-0.22, -0.05],
            ylim=[-0.20, 0.00],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
            obj_padding=0.05,
        )
        self.des_obj_pose = target_rand_pose.p.tolist() + [0, 0, 0, 1]
        self.des_obj_pose[2] += 0.04

    def play_once(self):
        arm_tag = self.arm_tag

        # Counter cuboid must be out of the Curobo world so the wrist can
        # drop to apple height inside the bowl.
        self.enable_table(enable=False)

        self.move(self.open_gripper(arm_tag, pos=1.0))

        apple_p = self.target_obj.get_pose().p
        bp = self.bowl.get_pose().p

        # Hover above the bowl rim, centered over the apple.
        hover_tcp_z = float(bp[2]) + 0.12
        hover_pose = [
            float(apple_p[0]),
            float(apple_p[1]),
            hover_tcp_z + self.TCP_OFFSET,
        ] + self.TOP_DOWN_Q
        self.move(self.move_to_pose(arm_tag, hover_pose))

        # Descend to apple.
        grasp_tcp_z = float(apple_p[2]) + 0.005
        grasp_pose = [
            float(apple_p[0]),
            float(apple_p[1]),
            grasp_tcp_z + self.TCP_OFFSET,
        ] + self.TOP_DOWN_Q
        self.move(self.move_to_pose(arm_tag, grasp_pose))

        self.move(self.close_gripper(arm_tag, pos=0.0))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))

        # Place on counter. place_actor recomputes pre-place from the
        # scripted top-down attach frame and fails IK; use explicit
        # move_to_pose + open_gripper, keeping the top-down wrist.
        self.attach_object(
            self.target_obj,
            f"{os.environ['ASSETS_ROOT']}/objects/035_apple/collision/base{self.apple_id}.glb",
            str(arm_tag),
        )
        self.enable_table(enable=True)

        tgt_x, tgt_y, tgt_z = self.des_obj_pose[:3]
        hover_pose = [tgt_x, tgt_y, tgt_z + 0.15 + self.TCP_OFFSET] + self.TOP_DOWN_Q
        self.move(self.move_to_pose(arm_tag, hover_pose))
        drop_pose = [tgt_x, tgt_y, tgt_z + 0.04 + self.TCP_OFFSET] + self.TOP_DOWN_Q
        self.move(self.move_to_pose(arm_tag, drop_pose))
        self.move(self.open_gripper(arm_tag, pos=1.0))

    def check_success(self):
        tp = self.target_obj.get_pose().p
        table_top_z = self.kitchens_info["table_height"] + self.table_z_bias
        bp = self.bowl.get_pose().p
        on_counter_z = abs(tp[2] - table_top_z) < 0.08
        outside_bowl = (abs(tp[0] - bp[0]) > 0.12 or abs(tp[1] - bp[1]) > 0.12)
        return (on_counter_z
                and outside_bowl
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
