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


class put_book_on_book(Office_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)
    
    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name(), self.des_obj.get_name()}

    def load_actors(self):
        ylim = [self.file_holder.get_pose().p[1]]
        zlim = [self.office_info["file_holder_heights"][1]+0.1]
        xlim = [self.office_info["file_holder_lims"][0]+0.08, self.office_info["file_holder_lims"][2]-0.08]
        model_id1 = np.random.choice(self._target_ids("office", "043_book"))
        model_id2 = 1
        self.target_obj = rand_create_actor(
            self,
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            modelname="043_book",
            rotate_rand=False,
            qpos=euler2quat(np.pi, 0, np.pi/2, axes='sxyz'),
            convex=True,
            model_id=model_id1,
            is_static=False,
            scale = self.item_info['scales']['043_book'][f'{model_id1}']
        )
        self.target_obj.set_mass(0.05)
        self.stabilize_object(self.target_obj)

        xlim = [-0.55,0.1] if self.target_obj.get_pose().p[0] < 0 else [-0.1,0.55]
        ylim = [-0.15, self.office_info["file_holder_lims"][1]-0.1]
        zlim = [0.77]
        self.des_obj = rand_create_actor(
            self,
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            modelname="043_book",
            rotate_rand=False,
            qpos=euler2quat(np.pi/2,0, 0, axes='sxyz'),
            convex=True,
            model_id=model_id2,
            is_static=False,
            scale = self.item_info['scales']['043_book'][f'{model_id2}']
        )
        self.des_obj.set_name("043_book_ref")
        self.des_obj.set_mass(0.3)
        self.add_prohibit_area(self.des_obj, padding=0.05)
        self.des_obj_pose = self.des_obj.get_pose().p.tolist() + euler2quat(np.pi, np.pi/6, np.pi/2, axes='sxyz').tolist()
        self.des_obj_pose[2]+=0.05
        self.des_obj_pose[1]+=0.015


    def play_once(self):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag("left") if self.target_obj.get_pose().p[0] < 0 else ArmTag("right")

        self.move(self.grasp_actor(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.04, grasp_dis=0.01))
        self.move(self.move_by_displacement(arm_tag=arm_tag, y=-0.01, z=0.01))
        self.attach_object(self.target_obj, f"{os.environ['ASSETS_ROOT']}/objects/043_book/collision/base0.glb", str(arm_tag))
        self.enable_table(enable=False)
        

        self.move(
            self.place_actor(
                self.target_obj,
                arm_tag=arm_tag,
                target_pose=self.des_obj_pose,
                constrain="align",
                pre_dis=0.01,
                dis=0.005,
                local_up_axis=[0,1,0]
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
        eps1 = 0.02
        eps2 = 0.04

        return (np.all(abs(end_pose_actual[:2] - end_pose_desired[:2]) < np.array([eps1, eps2]))
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
