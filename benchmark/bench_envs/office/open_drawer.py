from bench_envs.office._office_base_task import Office_base_task
from envs.utils import *
from bench_envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob
from transforms3d.euler import euler2quat


class open_drawer(Office_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)
    
    def _get_target_object_names(self) -> set[str]:
        return set()

    def load_actors(self):
        # prohibit area around opening space
        cabinet_pose = self.cabinet.get_pose().p
        cabinet_pose[1]-= 0.19  
        self.prohibited_area["table"].append([cabinet_pose[0]-0.11, cabinet_pose[1]-0.1, cabinet_pose[0]+0.11, cabinet_pose[1]+0.1])
        self.add_operating_area(cabinet_pose, width = 0.12, length = 0.4)


    def play_once(self):
        arm_tag = ArmTag("left") if self.arr_v == 2 else ArmTag("right")

        self.disable_panel()

        init_move = { k: v for k,v in zip(['x','y','z'],np.random.uniform(0.05, 0.15, size=3))}
        print_c(f"Initial move before grasping: {init_move}", "RED")
        self.move(self.move_by_displacement(arm_tag=arm_tag, **init_move))


        self.move(self.grasp_actor(self.cabinet, arm_tag=arm_tag, pre_grasp_dis=0.05, grasp_dis=0.025))

        # Pull the drawer open
        for _ in range(3):
            self.move(self.move_by_displacement(arm_tag=arm_tag, y=-0.0633))
        
        self.move(self.open_gripper(arm_tag=arm_tag))

        # Record information about the objects and arm used in the task
        self.info["info"] = {
            # "{A}": f"047_mouse/base{self.mouse_id}",
            # "{B}": f"{self.color_name}",
            # "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        end_pose_actual = self.cabinet.get_qpos()[0]
        end_pose_desired = self.cabinet.get_qlimits()[0][1]

        eps1 = 0.03

        return (abs(end_pose_desired - end_pose_actual) < eps1
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
