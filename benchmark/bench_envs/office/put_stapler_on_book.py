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


class put_stapler_on_book(Office_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)
    
    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name(), self.des_obj.get_name()}

    def load_actors(self):
        self.side =  "left" if self.arr_v == 2 else "right"

        # set up cabinet
        self.add_cabinet_collision()
        limit = self.cabinet.get_qlimits()[0]
        self.cabinet.set_qpos([limit[1],0,0])

        # set up stapler --------------------------------------------------
        self.stapler_id = np.random.choice(self._target_ids("office", "048_stapler"))
        pose = self.cabinet.get_pose().p
        pose[1]-= 0.2
        pose[2] = self.office_info["table_height"]+0.025
        self.target_obj = rand_create_actor(
            scene=self,
            xlim = [pose[0]],
            ylim = [pose[1]],
            zlim = [pose[2]],
            rotate_rand = True,
            qpos = euler2quat(np.pi/2, 0, np.pi, axes='sxyz'),
            rotate_lim = [0, np.pi/8, 0],
            modelname="048_stapler",
            convex=True,
            model_id=self.stapler_id,
            is_static=False,
        )

        # book --------------------------------------------------
        if self.target_obj.get_pose().p[0] < 0:
            xlim = [self.office_info["table_lims"][0]+0.07, 0.1]
        else:
            xlim = [-0.1, self.office_info["table_lims"][2]-0.07]
        success, self.des_obj = rand_create_cluttered_actor(
            scene=self.scene,
            xlim=xlim,
            ylim=[self.office_info["table_lims"][1]+0.06, self.office_info["shelf_lims"][1]-0.06],
            zlim=[self.office_info["table_height"]],
            modelname="043_book",
            modelid=1,
            scale=self.item_info['scales']['043_book']['1'],
            modeltype="glb",
            rotate_rand=False,
            rotate_lim=[0, np.pi/4, 0],
            qpos=euler2quat(np.pi/2,0, 0, axes='sxyz'),
            obj_radius=0.05,
            z_offset=0,
            z_max=0.02,
            prohibited_area=self.prohibited_area["table"],
            constrained=False,
            is_static=False,
            size_dict=dict(),
        )
        if not success:
            raise RuntimeError("Failed to load des_obj")
        self.add_prohibit_area(self.des_obj, padding=0.01, area="table")
        self.des_obj_pose = self.des_obj.get_pose().p.tolist() + euler2quat(np.pi/2, 0, np.pi, axes='sxyz').tolist()
        self.des_obj_pose[2] += 0.04


    def play_once(self):
        # Determine which arm to use based on target_obj position (right if on right side, left otherwise)
        arm_tag = ArmTag("left" if self.target_obj.get_pose().p[0] < 0 else "right")

        self.move(self.grasp_actor(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.05, contact_point_id=[0,1]))
        # self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.03))
        self.attach_object(self.target_obj, f"{os.environ['ASSETS_ROOT']}/objects/048_stapler/collision/base{self.stapler_id}.glb", str(arm_tag))

        self.move(self.place_actor(
            self.target_obj,
            arm_tag=arm_tag,
            target_pose=self.des_obj_pose,
            pre_dis=0.03,
            dis=0.0,
            constrain="align",
            local_up_axis=[0,0,1],
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
