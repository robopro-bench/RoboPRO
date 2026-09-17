from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
import os
import transforms3d as t3d
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob

# Microwave door counts as closed only within 3 degrees of fully shut.
DOOR_CLOSED_RAD = math.radians(3.0)


class chain_heat_hamburger_ks(KitchenS_base_task):
    """
    Chain: hamburger on counter → into (already-open) microwave → close door.
    Composed from put_hamburger_in_microwave_ks + close_microwave_ks.
    """

    # Keep the door-swing zone in front of the microwave clear of spawned
    # objects, or the door cannot be closed.
    microwave_front_keepout = True

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        # Microwave door starts fully open (same recipe as put_hamburger task).
        limits = self.microwave.get_qlimits()
        qpos_mw = self.microwave.get_qpos()
        qpos_mw[0] = limits[0][1] * 0.95
        self.microwave.set_qpos(qpos_mw)

        # Same window as put_hamburger_in_microwave_ks. The door-swing
        # keepout is still on (clutter must not sit there), but the
        # hamburger is gone before the door closes, so it may spawn in
        # that strip. Skipping only that extra box — the microwave body
        # AABB still applies, which is what the atomic already clears.
        mw_x = float(self.microwave.get_pose().p[0])
        if mw_x < 0:
            xlim = [-0.45, -0.20]
        else:
            xlim = [0.20, 0.45]

        rand_pos = self.rand_pose_on_counter(
            xlim=xlim,
            ylim=[-0.23, -0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, np.pi, 0],
            obj_padding=0.06,
            allow_microwave_front=True,
        )

        self.hamburger_id = int(np.random.choice([0, 1, 2]))
        self.target_obj = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="006_hamburg",
            convex=True,
            model_id=self.hamburger_id,
        )
        self.target_obj.set_mass(0.05)
        self.add_prohibit_area(self.target_obj, padding=0.02, area="table")

    # -----------------------------------------------------------------
    # Step 1: put hamburger into microwave (mirrors put_hamburger_in_microwave_ks)
    # -----------------------------------------------------------------
    def _put_hamburger_in_microwave(self):
        mw_p = self.microwave.get_pose().p
        mw_x = float(mw_p[0])
        mw_y = float(mw_p[1])
        mw_z = float(mw_p[2])

        arm_tag = ArmTag("right" if mw_x > 0 else "left")

        self.grasp_actor_from_table(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.08)
        if not self.plan_success:
            return

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))

        self.attach_object(
            self.target_obj,
            f"{os.environ['ASSETS_ROOT']}/objects/006_hamburg/collision/base{self.hamburger_id}.glb",
            str(arm_tag),
        )
        self.enable_table(enable=True)

        # Pop microwave from curobo collision world so planner can enter the cavity.
        self.collision_list = [
            e for e in self.collision_list if e.get("actor") is not self.microwave
        ]
        self.update_world()

        base_q = np.array([0.707, 0.0, 0.0, 0.707])
        tilt_rad = -math.pi / 9
        tilt_q = np.array([math.cos(tilt_rad / 2), math.sin(tilt_rad / 2), 0.0, 0.0])
        grasp_q = list(t3d.quaternions.qmult(tilt_q, base_q))

        tgt_x = mw_x + float(np.random.uniform(-0.08, -0.03))
        tgt_y = mw_y + float(np.random.uniform(-0.10, -0.04))
        tgt_z = mw_z + 0.02

        hover_pose = [tgt_x, tgt_y - 0.15, tgt_z] + grasp_q
        self.move(self.move_to_pose(arm_tag, hover_pose))
        if not self.plan_success:
            self.plan_success = True

        insert_pose = [tgt_x, tgt_y, tgt_z - 0.02] + grasp_q
        self.move(self.move_to_pose(arm_tag, insert_pose))
        if not self.plan_success:
            self.plan_success = True

        self.move(self.open_gripper(arm_tag, pos=1.0))
        self.move(self.move_by_displacement(arm_tag=arm_tag, y=-0.20))
        if not self.plan_success:
            self.plan_success = True
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))
        self.move(self.back_to_origin(arm_tag))

    # -----------------------------------------------------------------
    # Step 2: close microwave door (mirrors close_microwave_ks.play_once)
    # -----------------------------------------------------------------
    def _close_microwave(self):
        arm_tag = ArmTag("left")

        base_q = np.array([0.707, 0.0, 0.0, 0.707])
        tilt_rad = -math.pi / 4
        tilt_q = np.array([math.cos(tilt_rad / 2), math.sin(tilt_rad / 2), 0.0, 0.0])
        grasp_q = list(t3d.quaternions.qmult(tilt_q, base_q))
        cp_mat = self.microwave.get_contact_point(0, "matrix")
        hx, hy, hz = float(cp_mat[0, 3]), float(cp_mat[1, 3]), float(cp_mat[2, 3])

        self.move(self.close_gripper(arm_tag, pos=0.0))

        for _ in range(3):
            dx = float(np.random.uniform(-0.03, 0.03))
            dy = float(np.random.uniform(-0.02, 0.02))
            dz = float(np.random.uniform(-0.02, 0.03))
            self.move(self.move_by_displacement(arm_tag=arm_tag, x=dx, y=dy, z=dz))
            if not self.plan_success:
                self.plan_success = True
                break

        hover_pose = [hx - 0.10, hy - 0.15, hz + 0.10] + grasp_q
        self.move(self.move_to_pose(arm_tag, hover_pose))

        push_pose = [hx - 0.10, hy - 0.15, hz] + grasp_q
        self.move(self.move_to_pose(arm_tag, push_pose))

        limits = self.microwave.get_qlimits()
        start_qpos = self.microwave.get_qpos()[0]

        # Push until the door is within 3 degrees of shut (strict eval threshold).
        for _ in range(3):
            self.move(self.move_by_displacement(arm_tag=arm_tag, x=0.10, y=0.00))
            new_qpos = self.microwave.get_qpos()[0]
            if not self.plan_success:
                break
            if new_qpos <= DOOR_CLOSED_RAD:
                break
            if abs(new_qpos - start_qpos) <= 0.001 and _ > 2:
                break
            start_qpos = new_qpos

        for _ in range(3):
            self.move(self.move_by_displacement(arm_tag=arm_tag, x=0.00, y=0.10, z=-0.01))
            new_qpos = self.microwave.get_qpos()[0]
            if not self.plan_success:
                break
            if new_qpos <= DOOR_CLOSED_RAD:
                break
            if abs(new_qpos - start_qpos) <= 0.001 and _ > 2:
                break
            start_qpos = new_qpos

    def play_once(self):
        self._put_hamburger_in_microwave()
        # Reset plan_success so the close phase still runs even if insertion
        # hit a planner hiccup — hamburger may still have been released.
        self.plan_success = True
        self._close_microwave()

    def check_success(self):
        # NOTE: no gripper-open check — the close phase leaves the left
        # gripper closed (pushing the door), which is the final task state.
        tp = self.target_obj.get_pose().p
        mw_p = self.microwave.get_pose().p
        limits = self.microwave.get_qlimits()
        burger_in_mw = (abs(tp[0] - mw_p[0]) < 0.14
                        and abs(tp[1] - mw_p[1]) < 0.22
                        and tp[2] < mw_p[2] + 0.18
                        and tp[2] > mw_p[2] - 0.15)
        door_closed = self.microwave.get_qpos()[0] <= DOOR_CLOSED_RAD
        return burger_in_mw and door_closed
