from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
import os
import transforms3d as t3d
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class put_hamburger_in_microwave_ks(KitchenS_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        # Force microwave door fully open so the interior is accessible.
        limits = self.microwave.get_qlimits()
        qpos_mw = self.microwave.get_qpos()
        qpos_mw[0] = limits[0][1] * 0.95
        self.microwave.set_qpos(qpos_mw)

        # Spawn the hamburger on the counter, in the front band, on the
        # same side of the table as the microwave so the same arm can
        # perform both the grasp and the insertion.
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
        )

        # 006_hamburg has three compact variants that all grasp reliably.
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

    def play_once(self):
        mw_p = self.microwave.get_pose().p
        mw_x = float(mw_p[0])
        mw_y = float(mw_p[1])
        mw_z = float(mw_p[2])

        # Grasp with the same-side arm (guaranteed by spawn band).
        arm_tag = ArmTag("right" if mw_x > 0 else "left")

        # -------------------------------------------------------------
        # 1) Grasp the hamburger.
        # -------------------------------------------------------------
        self.grasp_actor_from_table(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.08)
        if not self.plan_success:
            return

        # Lift clear of the counter.
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))

        self.attach_object(
            self.target_obj,
            f"{os.environ['ASSETS_ROOT']}/objects/006_hamburg/collision/base{self.hamburger_id}.glb",
            str(arm_tag),
        )
        self.enable_table(enable=True)

        # -------------------------------------------------------------
        # 2) Pop microwave from curobo collision world so the planner
        #    doesn't reject poses that graze the mouth/walls. Also keep
        #    the table enabled so the planner respects the counter top.
        # -------------------------------------------------------------
        self.collision_list = [
            e for e in self.collision_list if e.get("actor") is not self.microwave
        ]
        self.update_world()

        # -------------------------------------------------------------
        # 3) Reorient wrist to forward-facing tilted-down (INIT_Q rotated
        #    -20° about world +x), same orientation proven in
        #    pick_hamburger_from_microwave_ks.py. The microwave mouth
        #    faces world -y (toward the robot), so we approach from -y.
        # -------------------------------------------------------------
        base_q = np.array([0.707, 0.0, 0.0, 0.707])
        tilt_rad = -math.pi / 9
        tilt_q = np.array([math.cos(tilt_rad / 2), math.sin(tilt_rad / 2), 0.0, 0.0])
        grasp_q = list(t3d.quaternions.qmult(tilt_q, base_q))

        # Target pose inside the cavity — same band the pick task uses to
        # spawn the hamburger (interior is asymmetric: true cavity is in
        # world -x half and mw_y + [-0.17, 0]).
        tgt_x = mw_x + float(np.random.uniform(-0.08, -0.03))
        tgt_y = mw_y + float(np.random.uniform(-0.10, -0.04))
        tgt_z = mw_z + 0.02

        # Hover outside the mouth on the robot side, level with target.
        # With tilt -20°, TCP = link + (0, 0.11, -0.04), so link z ~ tgt_z.
        hover_pose = [tgt_x, tgt_y - 0.15, tgt_z] + grasp_q
        self.move(self.move_to_pose(arm_tag, hover_pose))
        if not self.plan_success:
            self.plan_success = True

        # -------------------------------------------------------------
        # 4) Insert: dive into the cavity and stop above the release point.
        # -------------------------------------------------------------
        insert_pose = [tgt_x, tgt_y, tgt_z - 0.02] + grasp_q
        self.move(self.move_to_pose(arm_tag, insert_pose))
        if not self.plan_success:
            self.plan_success = True

        # -------------------------------------------------------------
        # 5) Release the hamburger and retreat out of the mouth.
        # -------------------------------------------------------------
        self.move(self.open_gripper(arm_tag, pos=1.0))
        self.move(self.move_by_displacement(arm_tag=arm_tag, y=-0.20))
        if not self.plan_success:
            self.plan_success = True
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))

    def check_success(self):
        tp = self.target_obj.get_pose().p
        mw_p = self.microwave.get_pose().p
        return (abs(tp[0] - mw_p[0]) < 0.14
                and abs(tp[1] - mw_p[1]) < 0.22
                and tp[2] < mw_p[2] + 0.18
                and tp[2] > mw_p[2] - 0.15
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
