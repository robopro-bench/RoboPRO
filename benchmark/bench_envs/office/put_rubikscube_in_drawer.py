# from envs._base_task import Base_Task
from bench_envs.office._office_base_task import Office_base_task
from envs.utils import *
from bench_envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob
from transforms3d.euler import euler2quat


class put_rubikscube_in_drawer(Office_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)
    
    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        self.add_cabinet_collision()
        limit = self.cabinet.get_qlimits()[0]
        self.cabinet.set_qpos([limit[1],0,0])

        # target_obj ------------------------------------------------------------
        if self.arr_v == 1:
            level = np.random.choice([0,1])
        else:
            level = 0
        
        if level == 0:
            bias = 0.1
        else:
            bias = 0.025

        self.cube_id = np.random.choice(self._target_ids("office", "073_rubikscube"))
        if self.arr_v == 2:
            xlim = [self.office_info["shelf_lims"][0] + self.office_info["shelf_padding"], bias]
        else:
            xlim = [-bias, self.office_info["shelf_lims"][2] - self.office_info["shelf_padding"]]

        self.target_obj = rand_create_actor(
            self,
            xlim=xlim,
            ylim=[self.office_info["shelf_lims"][1]+0.065],
            zlim=[self.office_info["shelf_heights"][level]+0.04],
            modelname="073_rubikscube",
            qpos=euler2quat(0,0,np.pi, axes='sxyz'),
            rotate_rand=False,
            convex=True,
            model_id=self.cube_id,
            is_static=False,
        )
        self.add_prohibit_area(self.target_obj, padding=0.03, area=f"shelf{level}")


    def play_once(self):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag("left") if self.arr_v == 2 else ArmTag("right")

        self.move(self.grasp_actor(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.1, grasp_dis=0.04, contact_point_id=3))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.02, y=-0.02))
        self.attach_object(self.target_obj, f"{os.environ['ASSETS_ROOT']}/objects/073_rubikscube/collision/base{self.cube_id}.glb", str(arm_tag))

        target_pose = self.cabinet.get_functional_point(0)
        target_pose[1] -= 0.005

        _, actions = self.place_actor(
            self.target_obj,
            arm_tag=arm_tag,
            target_pose=target_pose,
            pre_dis=0.02,
            dis=0.02,
            constrain="align",
        )
        self.move((arm_tag, actions[1:]))

        # Record information about the objects and arm used in the task
        # self.info["info"] = {
        #     "{A}": f"047_mouse/base{self.mouse_id}",
        #     "{B}": f"{self.color_name}",
        #     "{a}": str(arm_tag),
        # }
        # return self.info

    def check_success(self):
        end_pose_actual = self.target_obj.get_pose().p
        end_pose_desired = self.cabinet.get_functional_point(0)
        end_pose_desired[2] = self.office_info["drawer_height"]
        eps1 = 0.05
        eps2 = 0.05
        eps3 = 0.02
        return (np.all(abs(end_pose_actual[:3] - end_pose_desired[:3]) < np.array([eps1, eps2, eps3]))
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
