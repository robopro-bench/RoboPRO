from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class drop_apple_in_bin_ks(KitchenS_base_task):

    supports_eval_env = True

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        rand_pos = self.rand_pose_on_counter(
            xlim=[-0.32, 0.32],
            ylim=[-0.15, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, np.pi, 0],
            obj_padding=0.05,
        )
        while abs(rand_pos.p[0]) < 0.3:
            rand_pos = self.rand_pose_on_counter(
                xlim=[-0.4, 0.4],
                ylim=[-0.15, 0.05],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, np.pi, 0],
                obj_padding=0.05,
            )

        # 035_apple has two variants (base0 ~6.6cm, base1 ~5.4cm). Both are
        # roughly spherical so they grasp reliably from any side.
        self.apple_id = int(np.random.choice([0, 1]))
        self.target_obj = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="035_apple",
            convex=True,
            model_id=self.apple_id,
        )
        self.target_obj.set_mass(0.05)

        # 063_tabletrashbin at scale 0.10 gives ~19x10x13 cm open-top bin. qpos
        # [0.5,0.5,0.5,0.5] rotates mesh-y (height) → world-z so the opening
        # faces up. IDs 0 and 6 are straight-walled bins used elsewhere in the
        # benchmark; they keep the drop footprint rectangular and predictable.
        # Static bin may sit over the middle sink; keep the original center
        # band so either arm can place into it.
        target_rand_pose = self.rand_pose_on_counter(
            xlim=[-0.12, 0.12],
            ylim=[-0.23, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, np.pi / 4, 0],
            obj_padding=0.12,
            allow_sink=True,
        )
        self.bin_id = int(np.random.choice([0, 6]))
        self.des_obj = create_actor(
            scene=self,
            pose=target_rand_pose,
            modelname="063_tabletrashbin",
            convex=True,
            model_id=self.bin_id,
            scale=[0.10, 0.10, 0.10],
            is_static=True,
        )
        self.des_obj.set_name("bin")
        self.add_prohibit_area(self.des_obj, padding=0.02, area="table")
        self.add_prohibit_area(self.target_obj, padding=0.02, area="table")

        # Drop point is above the bin opening. Bin scaled height is ~0.10 m,
        # so we target ~0.08 m above the actor origin — the gripper releases
        # the apple just above the rim and it falls in.
        self.des_obj_pose = self.des_obj.get_pose().p.tolist() + [0, 0, 0, 1]
        self.des_obj_pose[2] += 0.08

    def capture_task_spec(self) -> dict:
        return {
            "apple_id": int(self.apple_id),
            "bin_id": int(self.bin_id),
            "target_name": self.target_obj.get_name(),
            "des_name": self.des_obj.get_name(),
        }

    def load_actors_from_spec(self, spec, init_state):
        self.apple_id = int(spec["apple_id"])
        apple_pose = self.pose_from_init_state(init_state, spec.get("target_name", "035_apple"))
        self.target_obj = create_actor(
            scene=self,
            pose=apple_pose,
            modelname="035_apple",
            convex=True,
            model_id=self.apple_id,
        )
        self.target_obj.set_mass(0.05)

        self.bin_id = int(spec["bin_id"])
        bin_name = spec.get("des_name", "bin")
        bin_pose = self.pose_from_init_state(init_state, bin_name)
        self.des_obj = create_actor(
            scene=self,
            pose=bin_pose,
            modelname="063_tabletrashbin",
            convex=True,
            model_id=self.bin_id,
            scale=[0.10, 0.10, 0.10],
            is_static=True,
        )
        self.des_obj.set_name("bin")
        self.add_prohibit_area(self.des_obj, padding=0.02, area="table")
        self.add_prohibit_area(self.target_obj, padding=0.02, area="table")
        self._refresh_des_obj_pose()

    def after_eval_env_restore(self):
        self._refresh_des_obj_pose()

    def _refresh_des_obj_pose(self):
        self.des_obj_pose = self.des_obj.get_pose().p.tolist() + [0, 0, 0, 1]
        self.des_obj_pose[2] += 0.08

    def play_once(self):
        arm_tag = ArmTag("right" if self.target_obj.get_pose().p[0] > 0 else "left")

        self.grasp_actor_from_table(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.07)
        if not self.plan_success:
            return

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))

        self.attach_object(
            self.target_obj,
            f"{os.environ['BENCH_ROOT']}/assets/objects/035_apple/collision/base{self.apple_id}.glb",
            str(arm_tag),
        )
        self.enable_table(enable=True)

        self.move(
            self.place_actor(
                self.target_obj,
                arm_tag=arm_tag,
                target_pose=self.des_obj_pose,
                constrain="align",
                pre_dis=0.08,
                dis=0.01,
            ))

    def check_success(self):
        end_pose_actual = self.target_obj.get_pose().p
        end_pose_desired = self.des_obj.get_pose().p
        # Apple inside bin footprint (~±9.5cm in x, ±6.5cm in y after rotation)
        # with a little slack. Also require the apple to be near or below the
        # bin top — i.e. it actually fell in instead of being balanced above.
        eps1 = 0.08
        eps2 = 0.06

        return (np.all(abs(end_pose_actual[:2] - end_pose_desired[:2]) < np.array([eps1, eps2]))
                and end_pose_actual[2] < end_pose_desired[2] + 0.10
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
