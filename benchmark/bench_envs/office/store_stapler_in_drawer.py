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


class store_stapler_in_drawer(Office_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)
    
    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        self.side =  "left" if self.arr_v == 2 else "right"

        # set up cabinet
        self.add_cabinet_collision()

        # set up target_obj --------------------------------------------------
        self.stapler_id = np.random.choice(self._target_ids("office", "048_stapler"))
        if self.side == "left":
            xlim = [self.office_info["table_lims"][0]+0.03, 0.1]
        else:
            xlim = [-0.1, self.office_info["table_lims"][2]-0.03]

        success, self.target_obj = rand_create_cluttered_actor(
            scene=self,
            xlim=xlim,
            ylim=[self.office_info["table_lims"][1]+0.04, self.office_info["shelf_lims"][1]-0.1],
            zlim=[self.office_info["table_height"]],
            modelname="048_stapler",
            modelid=self.stapler_id,
            modeltype="glb",
            rotate_rand=True,
            rotate_lim=[0, np.pi/4, 0],
            qpos=euler2quat(np.pi/2, 0, np.pi, axes='sxyz'),
            size_dict=dict(),
            obj_radius=0.04,
            z_offset=0,
            z_max=0.04,
            prohibited_area=self.prohibited_area["table"],
            constrained=False,
            is_static=False,
        )
        if not success:
            raise RuntimeError("Failed to load target_obj")
        self.add_prohibit_area(self.target_obj, padding=0.01)
        self.target_obj.set_mass(0.01)

    def play_once(self):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag(self.side)

        # disable front panel collision while opening drawer
        self.enable_drawer(enable=False)

        self.move(self.grasp_actor(self.cabinet, arm_tag=arm_tag, pre_grasp_dis=0.05, grasp_dis=0.025))

        # Pull the drawer
        for _ in range(3):
            self.move(self.move_by_displacement(arm_tag=arm_tag, y=-0.0633))
        
        self.move(self.open_gripper(arm_tag=arm_tag))
        self.move(self.move_by_displacement(arm_tag=arm_tag, y=-0.02))

        self.enable_drawer(enable=True)

        self.grasp_actor_from_table(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.07, contact_point_id=[0,1])
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.03))
        self.attach_object(self.target_obj, f"{os.environ['ASSETS_ROOT']}/objects/048_stapler/collision/base{self.stapler_id}.glb", str(arm_tag))
        self.enable_table(enable=True)

        des_obj_pose = self.cabinet.get_functional_point(0)
        des_obj_pose[1] -= 0.005
        # des_obj_pose[3:] = euler2quat(np.pi/2,0, np.pi, axes='sxyz')
        self.move(self.place_actor(
            self.target_obj,
            arm_tag=arm_tag,
            target_pose=des_obj_pose,
            pre_dis=0.03,
            dis=0.02,
            constrain="align",
        ))

        self.detach_object(arms_tag=arm_tag)

        arm_tag, actions = self.grasp_actor(self.cabinet, arm_tag=arm_tag, pre_grasp_dis=0.05, grasp_dis=0.025)

        self.move((arm_tag, [actions[0]]))
        self.enable_drawer(enable=False)
        self.move((arm_tag, actions[1:]))

        # Pull the drawer
        for _ in range(3):
            self.move(self.move_by_displacement(arm_tag=arm_tag, y=0.0633))
        
        self.move(self.open_gripper(arm_tag=arm_tag))

        # Record information about the objects and arm used in the task
        # self.info["info"] = {
        #     "{A}": f"047_mouse/base{self.mouse_id}",
        #     "{B}": f"{self.color_name}",
        #     "{a}": str(arm_tag),
        # }
        # return self.info

    def check_success(self):
        end_pose_actual1 = self.target_obj.get_pose().p
        end_pose_desired1 = self.cabinet.get_functional_point(0)
        end_pose_desired1[2] = self.office_info["drawer_height"]
        eps1 = 0.04
        eps2 = 0.04
        eps3 = 0.02

        end_pose_actual2 = self.cabinet.get_qpos()[0]
        end_pose_desired2 = self.cabinet.get_qlimits()[0][0]
        eps4 = 0.03


        return (np.all(abs(end_pose_actual1[:3] - end_pose_desired1[:3]) < np.array([eps1, eps2, eps3]))
                and (abs(end_pose_desired2 - end_pose_actual2) < eps4)
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
