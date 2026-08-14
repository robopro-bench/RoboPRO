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

# Microwave door counts as closed only within 3 degrees of fully shut.
DOOR_CLOSED_RAD = math.radians(3.0)


class chain_serve_hamburger_ks(KitchenS_base_task):
    """
    Chain: hamburger starts inside (already-open) microwave →
    pick it out → place in bowl on counter → close microwave door.
    Composed from pick_hamburger_from_microwave_ks + close_microwave_ks.
    """

    # Keep the door-swing zone in front of the microwave clear of spawned
    # objects, or the door cannot be closed.
    microwave_front_keepout = True

    # Skip scene 2 (microwave center): the cavity is out of the arm's IK reach
    # for the pick grasp. Only scenes 0/1 (microwave on the left) are used.
    allowed_scene_ids = (0, 1)

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name(), self.bowl.get_name()}

    def load_actors(self):
        # Microwave door fully open.
        limits = self.microwave.get_qlimits()
        qpos_mw = self.microwave.get_qpos()
        qpos_mw[0] = limits[0][1] * 0.95
        self.microwave.set_qpos(qpos_mw)

        mw_p = self.microwave.get_pose().p
        mw_x = float(mw_p[0])
        mw_y = float(mw_p[1])
        mw_z = float(mw_p[2])

        # Hamburger inside the microwave cavity (same band as pick_hamburger).
        self.hamburger_id = int(np.random.choice([0, 1, 2]))
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

        # Pick-side arm (same side as microwave).
        self.arm_tag = ArmTag("right" if mw_x > 0 else "left")

        # Bowl stays on the counter while the door closes, so it cannot
        # use allow_microwave_front. The keepout eats the original
        # ylim=[-0.20, 0.00] completely; the only gap that also clears
        # the scene-1 sink is the front strip (inboard x hits the basin).
        side_sign = 1 if self.arm_tag == "right" else -1
        bowl_pose = self.rand_pose_on_counter(
            xlim=[0.05, 0.25] if side_sign > 0 else [-0.25, -0.05],
            ylim=[-0.31, -0.26],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
            obj_padding=0.08,
        )
        self.bowl_id = 3
        self.bowl = create_actor(
            scene=self,
            pose=bowl_pose,
            modelname="002_bowl",
            convex=True,
            model_id=self.bowl_id,
            is_static=True,
        )
        self.bowl.set_mass(0.1)
        self.add_prohibit_area(self.bowl, padding=0.02, area="table")

    # -----------------------------------------------------------------
    # Step 1: pick hamburger out of microwave, place on plate
    # (mirrors pick_hamburger_from_microwave_ks, drop on plate not counter)
    # -----------------------------------------------------------------
    def _microwave_mesh_name(self) -> str:
        pose = self.microwave.get_pose()
        np_pose = np.concatenate([pose.p, pose.q]).tolist()
        return f"{self.microwave.get_name()}_{np_pose}_{self.seed}"

    def _disable_microwave_obstacle(self):
        try:
            self.collision_list = [
                e for e in self.collision_list if e.get("actor") is not self.microwave
            ]
            self.update_world()
        except Exception:
            pass

    def _pick_hamburger_into_bowl(self):
        arm_tag = self.arm_tag

        self._disable_microwave_obstacle()
        self.enable_table(enable=False)
        self.move(self.open_gripper(arm_tag, pos=1.0))

        base_q = np.array([0.707, 0.0, 0.0, 0.707])
        tilt_rad = -math.pi / 9
        tilt_q = np.array([math.cos(tilt_rad / 2), math.sin(tilt_rad / 2), 0.0, 0.0])
        grasp_q = list(t3d.quaternions.qmult(tilt_q, base_q))

        h_p = self.target_obj.get_pose().p
        hx, hy, hz = float(h_p[0]), float(h_p[1]), float(h_p[2])

        hover_pose = [hx, hy - 0.15, hz] + grasp_q
        self.move(self.move_to_pose(arm_tag, hover_pose))

        grasp_pose = [hx, hy, hz - 0.02] + grasp_q
        self.move(self.move_to_pose(arm_tag, grasp_pose))
        self.move(self.close_gripper(arm_tag, pos=0.0))

        self.attach_object(
            self.target_obj,
            f"{os.environ['ASSETS_ROOT']}/objects/006_hamburg/collision/base{self.hamburger_id}.glb",
            str(arm_tag),
        )

        self.move(self.move_by_displacement(arm_tag=arm_tag, y=-0.20))
        self.enable_table(enable=True)
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))

        # Place in bowl with the same top-down recipe as
        # move_hamburger_onto_plate_ks: place_actor + align constraint +
        # identity target quat lets the helper pick the best top-down wrist.
        bowl_p = self.bowl.get_pose().p
        bowl_target = [float(bowl_p[0]), float(bowl_p[1]),
                       float(bowl_p[2]) + 0.03, 0, 0, 0, 1]
        self.move(self.place_actor(
            self.target_obj, arm_tag=arm_tag, target_pose=bowl_target,
            constrain="align", pre_dis=0.05, dis=0.005,
        ))
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
        self._pick_hamburger_into_bowl()
        self.plan_success = True
        self._close_microwave()

    def check_success(self):
        # NOTE: no gripper-open check — the close phase leaves the left
        # gripper closed (pushing the door), which is the final task state.
        tp = self.target_obj.get_pose().p
        bowl_p = self.bowl.get_pose().p
        limits = self.microwave.get_qlimits()
        burger_in_bowl = (abs(tp[0] - bowl_p[0]) < 0.08
                          and abs(tp[1] - bowl_p[1]) < 0.08
                          and tp[2] > bowl_p[2] - 0.02)
        door_closed = self.microwave.get_qpos()[0] <= DOOR_CLOSED_RAD
        return burger_in_bowl and door_closed
