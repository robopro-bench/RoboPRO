from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
import os
import transforms3d as t3d
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class pick_fork_from_sink_ks(KitchenS_base_task):

    # Scripted top-down grasp — the fork's labeled side-grasp contact has
    # no IK inside the sink basin (same issue as pick_apple_from_sink).
    # link +x → world -z with this quat; fingers (link +y) → world -x by default.
    TOP_DOWN_Q = [-0.5, 0.5, -0.5, -0.5]
    TCP_OFFSET = 0.12
    # 033_fork mesh origin is at the tines end; geometric center is offset
    # by center_in_mesh * scale (from model_data0.json) along mesh-y (long axis).
    # Grasping at this world-frame center puts fingers on the fork neck/middle.
    FORK_CENTER_MESH = np.array([0.006306729695638125, 0.7365346359265317, -0.020780860566354562])
    FORK_SCALE = 0.15

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        sink_p = self.sink.get_pose().p
        sg = self.kitchens_info["sink_geom"]

        # Identity qpos lays the fork flat: mesh-y (22 cm long axis) → world +y,
        # mesh-z (thin dim) → world +z. Fork extends +0.22 m along +y from its
        # origin (tines end at `by`, handle at `by + 0.22`). Grasp point is the
        # geometric center at `by + 0.11`, so by must be pushed toward -y to
        # keep the grasp in the front (robot-side) of the basin.
        bx = sink_p[0] + np.random.uniform(-sg["inner_hx"] * 0.25, sg["inner_hx"] * 0.25)
        by = sink_p[1] + np.random.uniform(-sg["inner_hy"] * 0.95, -sg["inner_hy"] * 0.65)
        bz = sink_p[2] + 0.02
        rand_pos = sapien.Pose([bx, by, bz], [1.0, 0.0, 0.0, 0.0])

        self.fork_id = 0
        self.target_obj = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="033_fork",
            convex=True,
            model_id=self.fork_id,
        )
        self.target_obj.set_mass(0.05)

        self.arm_tag = ArmTag("right" if bx > 0 else "left")
        self.lift_z = sink_p[2] + 0.25

        # Counter drop pose on the same arm side.
        side_sign = 1 if self.arm_tag == "right" else -1
        target_rand_pose = self.rand_pose_on_counter(
            xlim=[0.05, 0.32] if side_sign > 0 else [-0.32, -0.05],
            ylim=[-0.20, 0.00],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
            obj_padding=0.05,
        )
        self.des_obj_pose = target_rand_pose.p.tolist() + [0, 0, 0, 1]
        self.des_obj_pose[2] += 0.04

    def _wrist_q_for_fork(self):
        # Read fork's settled long-axis direction (mesh-y in world xy plane)
        # and rotate TOP_DOWN_Q about world-z so gripper fingers end up
        # perpendicular to that axis. Wrap to [-pi/2, pi/2] since the fingers
        # are symmetric — pick the smallest wrist rotation.
        fork_q = self.target_obj.get_pose().q
        R = t3d.quaternions.quat2mat(fork_q)
        long_world = R @ np.array([0.0, 1.0, 0.0])
        theta = math.atan2(long_world[1], long_world[0])

        # Default TOP_DOWN_Q fingers open along world -x (yaw = pi).
        # Target finger yaw = theta + pi/2. Delta = theta - pi/2, wrap to (-pi/2, pi/2].
        delta = theta - math.pi / 2
        delta = (delta + math.pi / 2) % math.pi - math.pi / 2

        yaw_rot_q = np.array([math.cos(delta / 2), 0.0, 0.0, math.sin(delta / 2)])
        return list(t3d.quaternions.qmult(yaw_rot_q, np.array(self.TOP_DOWN_Q)))

    def play_once(self):
        arm_tag = self.arm_tag

        # Counter cuboid must be out of the Curobo world so the wrist can
        # dip below the rim into the basin.
        self.enable_table(enable=False)

        self.move(self.open_gripper(arm_tag, pos=1.0))

        wrist_q = self._wrist_q_for_fork()
        fork_pose = self.target_obj.get_pose()
        R = t3d.quaternions.quat2mat(fork_pose.q)
        # Shift from mesh origin (tines end) to geometric center (fork neck/middle).
        grasp_world = np.array(fork_pose.p) + R @ (self.FORK_CENTER_MESH * self.FORK_SCALE)
        sink_p = self.sink.get_pose().p

        hover_tcp_z = float(sink_p[2]) + 0.08
        hover_pose = [
            float(grasp_world[0]),
            float(grasp_world[1]),
            hover_tcp_z + self.TCP_OFFSET,
        ] + wrist_q
        self.move(self.move_to_pose(arm_tag, hover_pose))

        grasp_tcp_z = float(grasp_world[2]) + 0.005
        grasp_pose = [
            float(grasp_world[0]),
            float(grasp_world[1]),
            grasp_tcp_z + self.TCP_OFFSET,
        ] + wrist_q
        self.move(self.move_to_pose(arm_tag, grasp_pose))

        self.move(self.close_gripper(arm_tag, pos=0.0))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))

        # Drop on counter. Keep top-down wrist; explicit move_to_pose + open_gripper
        # (place_actor recomputes pre-place from scripted attach frame and fails IK).
        self.attach_object(
            self.target_obj,
            f"{os.environ['ASSETS_ROOT']}/objects/033_fork/collision/base{self.fork_id}.glb",
            str(arm_tag),
        )
        self.enable_table(enable=True)

        tgt_x, tgt_y, tgt_z = self.des_obj_pose[:3]
        hover_pose = [tgt_x, tgt_y, tgt_z + 0.15 + self.TCP_OFFSET] + wrist_q
        self.move(self.move_to_pose(arm_tag, hover_pose))
        drop_pose = [tgt_x, tgt_y, tgt_z + 0.04 + self.TCP_OFFSET] + wrist_q
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
