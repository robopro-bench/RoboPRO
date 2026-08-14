# from envs._base_task import Base_Task
from bench_envs.office._office_base_task import Office_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class put_phone_next_to_cube(Office_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        self.side = np.random.choice(["left", "right"])
        if self.side == "left":
            xlim1 = [0, 0.2]
            xlim2 = [self.office_info["table_lims"][0]+0.05, self.office_info["table_lims"][0]+0.25]
            bias = -0.14
        else:
            xlim1 = [-0.2, 0]
            xlim2 = [self.office_info["table_lims"][2]-0.25, self.office_info["table_lims"][2]-0.05]
            bias = 0.14
        ylim1 = [self.office_info["table_lims"][1]+0.07, self.office_info["shelf_lims"][1]-0.15]
        ylim2 = [self.office_info["table_lims"][1]+0.07, self.office_info["shelf_lims"][1]-0.15]

        # des_obj
        self.cube_id = np.random.choice(self._target_ids("office", "073_rubikscube"))
        self.rubikscube = rand_create_actor(
            self,
            xlim=xlim1,
            ylim=ylim1,
            zlim=[self.office_info["table_height"]+0.03],
            modelname="073_rubikscube",
            rotate_rand=True,
            rotate_lim=[0, np.pi/4, 0],
            qpos=[0, 0, 0.7071, 0.7071],
            convex=True,
            model_id=self.cube_id,
            is_static=False,
        )
        self.add_prohibit_area(self.rubikscube, padding=0.01, area="table")

        p = self.rubikscube.get_pose().p.tolist()
        p[0] += bias
        p[2] = self.office_info["table_height"] - 0.001
        des_obj_pose = sapien.Pose(p=p, q=[1, 0, 0, 0])
        half_size = [0.05, 0.05, 0.0005]
        self.des_obj = create_box(
            scene=self,
            pose=des_obj_pose,
            half_size=half_size,
            color=(0, 0, 1),
            name="box",
            is_static=True,
        )
        self.add_prohibit_area(self.des_obj, padding=0.03, area="table")

        # target_obj ------------------------------------------------------------
        ori_quat = [
            [0.707, 0.707, 0, 0],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5, -0.5],
        ]

        self.phone_id = np.random.choice(self._target_ids("office", "077_phone"))
        phone_pose = rand_pose(
            xlim = xlim2,
            ylim=ylim2,
            zlim=[self.office_info["table_height"]+0.01],
            qpos=ori_quat[self.phone_id],
            rotate_rand=True,
            rotate_lim=[0, np.pi/4, 0],
        )
        self.target_obj = create_actor(
            scene=self,
            pose=phone_pose,
            modelname="077_phone",
            convex=True,
            model_id=self.phone_id,
            is_static=False,
        )
        self.target_obj.set_mass(0.01)
        self.add_prohibit_area(self.target_obj, padding=0.01, area="table")

        self.des_obj_pose = self.des_obj.get_pose().p.tolist() + ori_quat[self.phone_id]
        self.des_obj_pose[2] += 0.02

    def play_once(self):
        # Determine which arm to use based on target_obj position (right if on right side, left otherwise)
        arm_tag = ArmTag(self.side)

        # Grasp the target_obj with the selected arm
        self.grasp_actor_from_table(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.05, grasp_dis=0.01)

        # Lift the target_obj upward by 0.1 meters in z-direction
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.02))
        self.attach_object(self.target_obj, f"{os.environ['ASSETS_ROOT']}/objects/077_phone/collision/base{self.phone_id}.glb", str(arm_tag))
        self.enable_table(enable=True)

        # Place the target_obj at the des_obj location with alignment constraint
        self.move(
            self.place_actor(
                self.target_obj,
                arm_tag=arm_tag,
                target_pose=self.des_obj_pose,
                constrain="align",
                pre_dis=0.06,
                dis=0.005,
                local_up_axis=[0,0,1]
            ))

        # Record information about the objects and arm used in the task
        # self.info["info"] = {
        #     "{A}": f"047_mouse/base{self.mouse_id}",
        #     "{B}": f"{self.color_name}",
        #     "{a}": str(arm_tag),
        # }
        # return self.info

    def check_success(self):
        end_pose_actual = self.target_obj.get_pose().p
        end_pose_desired = self.des_obj.get_pose().p
        eps1 = 0.04
        eps2 = 0.04

        return (np.all(abs(end_pose_actual[:2] - end_pose_desired[:2]) < np.array([eps1, eps2]))
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
