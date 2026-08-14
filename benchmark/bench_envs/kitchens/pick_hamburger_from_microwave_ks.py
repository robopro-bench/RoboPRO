from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
import os
import numpy as np
import transforms3d as t3d
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class pick_hamburger_from_microwave_ks(KitchenS_base_task):

    # Skip scene 2 (microwave center): the cavity is out of the arm's IK reach
    # for the pick grasp. Only scenes 0/1 (microwave on the left) are used.
    allowed_scene_ids = (0, 1)

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def _microwave_mesh_name(self) -> str:
        """Reproduce the key used by Bench_base_task.update_world() so we
        can toggle the microwave in the Curobo collision world."""
        pose = self.microwave.get_pose()
        np_pose = np.concatenate([pose.p, pose.q]).tolist()
        return f"{self.microwave.get_name()}_{np_pose}_{self.seed}"

    def _disable_microwave_obstacle(self):
        """Remove the microwave from Curobo's collision list so the arm
        can plan inside the mouth without being rejected."""
        try:
            self.collision_list = [
                e for e in self.collision_list if e.get("actor") is not self.microwave
            ]
            self.update_world()
        except Exception:
            pass

    def load_actors(self):
        # Force microwave door fully open so the interior is accessible.
        # Same recipe as put_hamburger_in_microwave_ks (which is proven).
        limits = self.microwave.get_qlimits()
        qpos_mw = self.microwave.get_qpos()
        qpos_mw[0] = limits[0][1] * 0.95
        self.microwave.set_qpos(qpos_mw)

        mw_p = self.microwave.get_pose().p
        mw_x = float(mw_p[0])
        mw_y = float(mw_p[1])
        mw_z = float(mw_p[2])

        # Geometry (per put_hamburger_in_microwave_ks notes):
        # Microwave is yaw +π/2. Door face at world +y (toward robot).
        # Mouth plane ≈ mw_y + 0.18. Back wall ≈ mw_y - 0.11.
        # A target just inside the mouth sits at y ≈ mw_y + 0.08..0.12.
        #
        # Spawn hamburger just inside the mouth so the arm does not have
        # to reach deep into the cavity (poor IK inside the enclosure).
        self.hamburger_id = int(np.random.choice([0, 1, 2]))
        # Cavity is asymmetric: mesh +x half (world +y) is the solid
        # electronics compartment; true hollow cavity spans mw_y + [-0.17, 0]
        # in world y and sits slightly to -x in world x.
        spawn_x = mw_x + float(np.random.uniform(-0.08, -0.03))
        spawn_y = mw_y + float(np.random.uniform(-0.10, -0.04))
        # Spawn at the cavity floor (measured: hamburger rests at mw_z - 0.074).
        # The old mw_z + 0.02 spawned it ~9.5 cm in midair → free-fall + roll.
        spawn_z = mw_z - 0.07

        hpose = sapien.Pose(
            [spawn_x, spawn_y, spawn_z],
            [0.5, 0.5, 0.5, 0.5],
        )
        self.target_obj = create_actor(
            scene=self,
            pose=hpose,
            modelname="006_hamburg",
            convex=True,
            model_id=self.hamburger_id,
        )
        self.target_obj.set_mass(0.05)

        # Pick the arm on the microwave's side. mw at x=-0.32 → left arm.
        self.arm_tag = ArmTag("right" if mw_x > 0 else "left")

        # Counter drop pose on the opposite half from the microwave.
        side_sign = 1 if self.arm_tag == "right" else -1
        target_rand_pose = self.rand_pose_on_counter(
            xlim=[0.05, 0.25] if side_sign > 0 else [-0.25, -0.05],
            ylim=[-0.20, 0.00],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
            obj_padding=0.05,
        )
        self.des_obj_pose = target_rand_pose.p.tolist() + [0, 0, 0, 1]
        self.des_obj_pose[2] += 0.04

    def play_once(self):
        arm_tag = self.arm_tag

        # Pop the microwave from the Curobo collision world so the arm can
        # plan into the mouth; table disabled so the wrist can dip low.
        self._disable_microwave_obstacle()
        self.enable_table(enable=False)
        self.move(self.open_gripper(arm_tag, pos=1.0))

        # Forward-facing wrist (INIT_Q = 90° about z → TCP points world +y,
        # i.e. into the microwave from the robot side) tilted a bit down so
        # the TCP dives into the cavity. Top-down/side grasps have no IK
        # inside this enclosure.
        base_q = np.array([0.707, 0.0, 0.0, 0.707])
        # Rotate -20° about world +x: TCP direction (world +y) → toward -z.
        tilt_rad = -math.pi / 9
        tilt_q = np.array([math.cos(tilt_rad / 2), math.sin(tilt_rad / 2), 0.0, 0.0])
        grasp_q = list(t3d.quaternions.qmult(tilt_q, base_q))

        h_p = self.target_obj.get_pose().p
        hx, hy, hz = float(h_p[0]), float(h_p[1]), float(h_p[2])

        # Hover in front of the mouth (robot side), level with hamburger.
        # With tilt -20°, TCP sits 4 cm below link; lift link ~4 cm above
        # hamburger so TCP lines up with the hamburger's height.
        hover_pose = [hx, hy - 0.15, hz ] + grasp_q
        self.move(self.move_to_pose(arm_tag, hover_pose))

        # Dive into cavity and close onto hamburger. Link z picked so the
        # tilted TCP (link_z - 0.04) lands just below the hamburger center.
        grasp_pose = [hx, hy, hz - 0.02] + grasp_q
        self.move(self.move_to_pose(arm_tag, grasp_pose))
        self.move(self.close_gripper(arm_tag, pos=0.0))

        self.attach_object(
            self.target_obj,
            f"{os.environ['ASSETS_ROOT']}/objects/006_hamburg/collision/base{self.hamburger_id}.glb",
            str(arm_tag),
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))

        # Place on counter with the same wrist orientation.
        tgt_x, tgt_y, tgt_z = self.des_obj_pose[:3]
        hover_drop = [tgt_x, tgt_y, tgt_z + 0.20] + grasp_q
        self.move(self.move_to_pose(arm_tag, hover_drop))
        drop_pose = [tgt_x, tgt_y, tgt_z + 0.06] + grasp_q
        self.move(self.move_to_pose(arm_tag, drop_pose))
        self.move(self.open_gripper(arm_tag, pos=1.0))

    def check_success(self):
        tp = self.target_obj.get_pose().p
        mw_p = self.microwave.get_pose().p
        mw_y = float(mw_p[1])
        table_top_z = self.kitchens_info["table_height"] + self.table_z_bias
        on_counter_z = abs(tp[2] - table_top_z) < 0.08
        # Robot side of microwave = -y; "outside" for this task = past mouth plane.
        outside_mw = tp[1] < mw_y - 0.18
        return (on_counter_z
                and outside_mw
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
