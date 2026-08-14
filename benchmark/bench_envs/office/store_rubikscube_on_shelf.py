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


class store_rubikscube_on_shelf(Office_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)
    
    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name(), self.des_obj.get_name()}

    def load_actors(self):
        
        # prepare drawer --------------------------------------------
        self.add_cabinet_collision()

        # rubikscube --------------------------------------------------
        self.cube_id = np.random.choice(self._target_ids("office", "073_rubikscube"))
        center = self.cabinet.get_pose().p
        center[1] -= 0.02
        center[2] = self.office_info["table_height"]+0.03
        self.target_obj = rand_create_actor(
            self,
            xlim=[center[0]],
            ylim=[center[1]],
            zlim=[center[2]],
            modelname="073_rubikscube",
            rotate_rand=True,
            rotate_lim=[0, np.pi/8, 0],
            qpos=[0, 0, 0.7071, 0.7071],
            convex=True,
            model_id=self.cube_id,
            is_static=False,
        )
        self.target_obj.set_mass(0.1)
        # des_obj_pose ------------------------------------------------------
        self.side = "left" if self.arr_v == 2 else "right"
        self.level = 0
        bias = 0.08
        xlim = [self.office_info["shelf_lims"][0] + self.office_info["shelf_padding"], bias] if self.side == "left" else [-bias, self.office_info["shelf_lims"][2] -self.office_info["shelf_padding"]]
        target_rand_pose = rand_pose(
            xlim=xlim,
            ylim=[self.office_info["shelf_lims"][1] + 0.055],
            zlim = [self.office_info["shelf_heights"][self.level]],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )

        half_size = [0.05, 0.05, 0.0005]
        self.des_obj = create_box(
            scene=self,
            pose=target_rand_pose,
            half_size=half_size,
            color=(0, 0, 1), # blue
            name="des_obj",
            is_static=True,
        )
        self.des_obj_pose = self.des_obj.get_pose().p.tolist() + euler2quat(np.pi/2,np.pi, 0, axes='sxyz').tolist()
        self.des_obj_pose[2] += 0.06 # raise des_obj_pose 0.02 meters
        # self.des_obj_pose[0] -= 0.02
        self.add_prohibit_area(self.des_obj, padding=0.05, area=f"shelf{self.level}")


    def play_once(self):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag("left") if self.arr_v == 2 else ArmTag("right")

        # disable front panel collision while opening drawer
        self.enable_drawer(enable=False)

        self.move(self.grasp_actor(self.cabinet, arm_tag=arm_tag, pre_grasp_dis=0.05, grasp_dis=0.025))

        # Pull the drawer
        for _ in range(3):
            self.move(self.move_by_displacement(arm_tag=arm_tag, y=-0.0633))
        
        self.move(self.open_gripper(arm_tag=arm_tag))
        self.move(self.move_by_displacement(arm_tag=arm_tag, y=-0.02))

        self.enable_drawer(enable=True)

        action = self.grasp_actor(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.1, contact_point_id=3)
        if action:
            action[1][1].target_pose[2] += 0.04 # grasp center of box
        self.move(action)

        # Lift the box upward by 0.1 meters in z-direction
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.03))

        self.attach_object(self.target_obj, f"{os.environ['ASSETS_ROOT']}/objects/073_rubikscube/collision/base{self.cube_id}.glb", str(arm_tag))

        self.move(
            self.place_actor(
                self.target_obj,
                arm_tag=arm_tag,
                target_pose=self.des_obj_pose,
                constrain="align",
                pre_dis=0.08,
                dis=-0.02,
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
        end_pose_actual1 = self.target_obj.get_pose().p[2]
        end_pose_desired1 = self.office_info["shelf_heights"][self.level] + 0.03

        end_pose_actual2 = self.cabinet.get_qpos()[0]
        end_pose_desired2 = self.cabinet.get_qlimits()[0][0]

        eps1 = 0.02
        eps2 = 0.03


        return (abs(end_pose_actual1 - end_pose_desired1) < eps1
                and abs(end_pose_desired2 - end_pose_actual2) < eps2
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
