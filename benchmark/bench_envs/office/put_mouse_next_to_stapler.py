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


class put_mouse_next_to_stapler(Office_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)
    
    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        self.side =  "left" if self.arr_v == 2 else "right"

        # set up cabinet
        self.add_cabinet_collision()
        limit = self.cabinet.get_qlimits()[0]
        self.cabinet.set_qpos([limit[1],0,0])

        # set up stapler --------------------------------------------------
        self.mouse_id = np.random.choice(self._target_ids("office", "047_mouse"))
        pose = self.cabinet.get_pose().p
        pose[1]-= 0.2
        pose[2] = self.office_info["table_height"]+0.03
        self.target_obj = rand_create_actor(
            scene=self,
            xlim = [pose[0]],
            ylim = [pose[1]],
            zlim = [pose[2]],
            rotate_rand = True,
            qpos = euler2quat(np.pi/2, 0, np.pi, axes='sxyz'),
            rotate_lim = [0, np.pi/8, 0],
            modelname="047_mouse",
            scale=self.item_info['scales']['047_mouse'].get(f'{self.mouse_id}',None),
            convex=True,
            model_id=self.mouse_id,
            is_static=False,
        )
        self.target_obj_pose = self.target_obj.get_pose().p.tolist()

        # des_obj
        if self.arr_v == 2:
            xlim = [self.office_info["table_lims"][0]+0.04, 0]
        else:
            xlim = [-0.2, self.office_info["table_lims"][2]-0.14]
        self.stapler_id = np.random.choice(self._target_ids("office", "048_stapler"))
        self.stapler = rand_create_actor(
            scene=self,
            xlim = xlim,
            ylim = [self.office_info["table_lims"][1]+0.04, -0.05],
            zlim = [self.office_info["table_height"]],
            rotate_rand = True,
            qpos = euler2quat(np.pi/2, 0, np.pi, axes='sxyz'),
            rotate_lim = [0, np.pi/4, 0],
            modelname="048_stapler",
            convex=True,
            model_id=self.stapler_id,
            is_static=False,
        )
        self.add_prohibit_area(self.stapler, padding=0.01, area="table")

        p = self.stapler.get_pose().p.tolist()
        p[0] += 0.1
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
        self.add_prohibit_area(self.des_obj, padding=0.02, area="table")

        self.des_obj_pose = self.des_obj.get_pose().p.tolist() + [0, 0, 0, 1]
        self.des_obj_pose[2] += 0.02


    def play_once(self):
        # Determine which arm to use based on target_obj position (right if on right side, left otherwise)
        arm_tag = ArmTag(self.side)

        self.move(self.grasp_actor(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.05,grasp_dis=0.02))
        self.attach_object(self.target_obj, f"{os.environ['ASSETS_ROOT']}/objects/047_mouse/collision/base{self.mouse_id}.glb", str(arm_tag))

        self.move(
            self.place_actor(
                self.target_obj,
                arm_tag=arm_tag,
                target_pose=self.des_obj_pose,
                constrain="align",
                pre_dis=0.02,
                dis=0.005,
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
