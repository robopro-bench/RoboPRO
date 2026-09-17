from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class pick_apple_from_sink_ks(KitchenS_base_task):

    # Scripted top-down grasp. The apple's labeled contacts all encode
    # side-grasp orientations, which have no IK in the sink basin. A
    # scripted top-down descent bypasses that.
    # For aloha (delta_matrix=identity, gripper_bias=0.12), the quat passed
    # to move_to_pose maps directly to the end-link orientation, and the
    # input position IS the end-link position (the 0.12 - gripper_bias
    # offset cancels). With this quat, link +x points in world -z.
    TOP_DOWN_Q = [-0.5, 0.5, -0.5, -0.5]
    # TCP sits 0.12 m ahead of the link along link +x. For a top-down
    # wrist, link +x = world -z, so TCP = link_pos + [0, 0, -0.12].
    # To place TCP at desired z, set link_z = desired_tcp_z + 0.12.
    TCP_OFFSET = 0.12

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        sink_p = self.sink.get_pose().p
        sg = self.kitchens_info["sink_geom"]

        # Apple spawn: near middle of the sink in x (tight band around
        # sink center) and restricted to the FRONT 1/3 of the basin
        # (closest to the robot — most IK-reachable for top-down descent).
        bx = sink_p[0] + np.random.uniform(-sg["inner_hx"] * 0.25, sg["inner_hx"] * 0.25)
        by = sink_p[1] + np.random.uniform(-sg["inner_hy"] * 0.9, -sg["inner_hy"] / 3)
        bz = sink_p[2] - 0.01
        rand_pos = sapien.Pose([bx, by, bz], [0.5, 0.5, 0.5, 0.5])

        self.apple_id = int(np.random.choice([0, 1]))
        self.target_obj = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="035_apple",
            convex=True,
            model_id=self.apple_id,
        )
        self.target_obj.set_mass(0.05)

        # Arm choice is driven by the sampled apple x.
        self.arm_tag = ArmTag("right" if bx > 0 else "left")
        self.lift_z = sink_p[2] + 0.25

        # Pre-compute a clear counter pose on the same arm side to place the
        # apple after retrieving it from the sink.
        side_sign = 1 if self.arm_tag == "right" else -1
        target_rand_pose = self.rand_pose_on_counter(
            xlim=[0.05 * side_sign, 0.32 * side_sign] if side_sign > 0 else [-0.32, -0.05],
            ylim=[-0.20, 0.00],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
            obj_padding=0.05,
        )
        self.des_obj_pose = target_rand_pose.p.tolist() + [0, 0, 0, 1]
        self.des_obj_pose[2] += 0.04

    def play_once(self):
        arm_tag = self.arm_tag

        # Table/counter cuboid must be out of the Curobo world so the wrist
        # can dip below the rim. Sink walls are not in the Curobo world.
        self.enable_table(enable=False)

        # --- scripted top-down grasp ---
        self.move(self.open_gripper(arm_tag, pos=1.0))

        apple_p = self.target_obj.get_pose().p
        sink_p = self.sink.get_pose().p

        # Hover: TCP ~8 cm above counter rim, centered over apple.
        hover_tcp_z = float(sink_p[2]) + 0.08
        hover_pose = [
            float(apple_p[0]),
            float(apple_p[1]),
            hover_tcp_z + self.TCP_OFFSET,
        ] + self.TOP_DOWN_Q
        self.move(self.move_to_pose(arm_tag, hover_pose))

        # Descend: TCP at apple's center (fingers straddle the apple).
        grasp_tcp_z = float(apple_p[2]) + 0.005
        grasp_pose = [
            float(apple_p[0]),
            float(apple_p[1]),
            grasp_tcp_z + self.TCP_OFFSET,
        ] + self.TOP_DOWN_Q
        self.move(self.move_to_pose(arm_tag, grasp_pose))

        # Close and lift clear of the rim.
        self.move(self.close_gripper(arm_tag, pos=0.0))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))

        # Place on counter. place_actor recomputes pre-place from the
        # scripted top-down attach frame and fails IK; use explicit
        # move_to_pose + open_gripper, keeping the top-down wrist
        # (same pattern as pick_apple_from_bowl_ks).
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
        sink_p = self.sink.get_pose().p
        sg = self.kitchens_info["sink_geom"]
        on_counter_z = abs(tp[2] - table_top_z) < 0.08
        outside_sink = (abs(tp[0] - sink_p[0]) > sg["hole_hx"]
                        or abs(tp[1] - sink_p[1]) > sg["hole_hy"])
        return (on_counter_z
                and outside_sink
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
