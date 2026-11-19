from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
import transforms3d as t3d
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class close_microwave_ks(KitchenS_base_task):

    # Keep the door-swing zone in front of the microwave clear of spawned
    # objects, or the door cannot be closed.
    microwave_front_keepout = True

    def _get_target_object_names(self) -> set[str]:
        return set()  # articulation-only task, no actor targets

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        # microwave is already loaded by KitchenS_base_task as self.microwave
        # Start with the door mostly open so the task is to close it.
        limits = self.microwave.get_qlimits()
        qpos = self.microwave.get_qpos()
        qpos[0] = limits[0][1] * 0.9  # mostly open
        self.microwave.set_qpos(qpos)

    def play_once(self):
        arm_tag = ArmTag("left")

        # Forward-facing wrist: INIT_Q (90 deg about z, TCP toward world +y)
        # rotated -45 deg about world +x, so the TCP points into the microwave
        # with a downward tilt.
        base_q = np.array([0.707, 0.0, 0.0, 0.707])
        tilt_rad = -math.pi / 4
        tilt_q = np.array([math.cos(tilt_rad / 2), math.sin(tilt_rad / 2), 0.0, 0.0])
        grasp_q = list(t3d.quaternions.qmult(tilt_q, base_q))
        cp_mat = self.microwave.get_contact_point(0, "matrix")
        hx, hy, hz = float(cp_mat[0, 3]), float(cp_mat[1, 3]), float(cp_mat[2, 3])

        # Close the gripper first: the handle is pushed with the closed
        # gripper, not grasped.
        self.move(self.close_gripper(arm_tag, pos=0.0))

        # Random pre-approach jitter (before hover) diversifies the
        # trajectory out of INIT_Q before committing to the approach.
        for _ in range(3):
            dx = float(np.random.uniform(-0.03, 0.03))
            dy = float(np.random.uniform(-0.02, 0.02))
            dz = float(np.random.uniform(-0.02, 0.03))
            self.move(self.move_by_displacement(arm_tag=arm_tag, x=dx, y=dy, z=dz))
            if not self.plan_success:
                self.plan_success = True  # don't abort; these are exploratory
                break

        # Hover a bit left of and in front of the handle (robot side = -y)
        # so the straight-line descent to the handle doesn't graze the door.
        hover_pose = [hx - 0.10, hy - 0.15, hz + 0.10] + grasp_q
        self.move(self.move_to_pose(arm_tag, hover_pose))

        push_pose = [hx - 0.10, hy - 0.15, hz ] + grasp_q
        self.move(self.move_to_pose(arm_tag, push_pose))

        limits = self.microwave.get_qlimits()
        start_qpos = self.microwave.get_qpos()[0]

        # Push the handle diagonally toward the microwave body to close it,
        # until check_success() holds (door within 3 degrees of shut).
        for _ in range(3):
            self.move(self.move_by_displacement(arm_tag=arm_tag, x=0.10, y=0.00))

            new_qpos = self.microwave.get_qpos()[0]
            if not self.plan_success:
                break
            if self.check_success():
                break
            # If nothing is happening, bail out early.
            if abs(new_qpos - start_qpos) <= 0.001 and _ > 2:
                break
            start_qpos = new_qpos
        # Push the handle diagonally toward the microwave body to close it.
        for _ in range(3):
            self.move(self.move_by_displacement(arm_tag=arm_tag, x=0.00, y=0.10, z=-0.01))

            new_qpos = self.microwave.get_qpos()[0]
            if not self.plan_success:
                break
            if self.check_success():
                break
            # If nothing is happening, bail out early.
            if abs(new_qpos - start_qpos) <= 0.001 and _ > 2:
                break
            start_qpos = new_qpos

    # Door counts as closed only within 3 degrees of fully shut.
    DOOR_CLOSED_RAD = math.radians(3.0)

    def check_success(self):
        qpos = self.microwave.get_qpos()
        return qpos[0] <= self.DOOR_CLOSED_RAD
