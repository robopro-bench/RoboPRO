from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class move_hamburger_onto_plate_ks(KitchenS_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name(), self.des_obj.get_name()}

    def load_actors(self):
        # Plate and hamburger on the SAME arm side (pattern mirrors
        # put_bread_on_board_ks: tight prohibit padding 0.01, separate x
        # bands so the two samplers don't compete). Plate is NOT static —
        # a real physics object that can shift if bumped.
        side_sign = int(np.random.choice([-1, 1]))
        # Hamburger on the outer half, plate closer to midline — both same side.
        hx_range = [0.22, 0.40] if side_sign > 0 else [-0.40, -0.22]
        px_range = [0.05, 0.18] if side_sign > 0 else [-0.18, -0.05]

        # Sample hamburger first (wider outer band).
        rand_pos = self.rand_pose_on_counter(
            xlim=hx_range,
            ylim=[-0.23, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, 3.14, 0],
            obj_padding=0.05,
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

        # Plate in the inner band.
        target_rand_pose = self.rand_pose_on_counter(
            xlim=px_range,
            ylim=[-0.23, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
            obj_padding=0.08,
        )
        self.plate_id = 0
        self.des_obj = create_actor(
            scene=self,
            pose=target_rand_pose,
            modelname="003_plate",
            convex=True,
            model_id=self.plate_id,
        )
        self.des_obj.set_mass(0.15)
        self.add_prohibit_area(self.des_obj, padding=0.01, area="table")

        self.des_obj_pose = self.des_obj.get_pose().p.tolist() + [0, 0, 0, 1]
        self.des_obj_pose[2] += 0.03

    def play_once(self):
        arm_tag = ArmTag("right" if self.target_obj.get_pose().p[0] > 0 else "left")

        # Hamburger is bulkier than bread — larger pre_grasp_dis for clearance.
        self.grasp_actor_from_table(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.08)

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1))

        self.attach_object(
            self.target_obj,
            f"{os.environ['ASSETS_ROOT']}/objects/006_hamburg/collision/base{self.hamburger_id}.glb",
            str(arm_tag),
        )
        self.enable_table(enable=True)

        self.move(
            self.place_actor(
                self.target_obj,
                arm_tag=arm_tag,
                target_pose=self.des_obj_pose,
                constrain="align",
                pre_dis=0.05,
                dis=0.005,
            ))

    def check_success(self):
        end_pose_actual = self.target_obj.get_pose().p
        end_pose_desired = self.des_obj.get_pose().p
        eps1 = 0.03
        eps2 = 0.03

        return (np.all(abs(end_pose_actual[:2] - end_pose_desired[:2]) < np.array([eps1, eps2]))
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
