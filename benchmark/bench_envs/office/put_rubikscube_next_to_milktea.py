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


class put_rubikscube_next_to_milktea(Office_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)
    
    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        level = np.random.choice([0,1])
        self.cube_id = np.random.choice(self._target_ids("office", "073_rubikscube"))
        xlim = [self.office_info["shelf_lims"][0] + self.office_info["shelf_padding"], self.office_info["shelf_lims"][2] - self.office_info["shelf_padding"]]
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
        self.add_prohibit_area(self.target_obj, padding=0.04, area=f"shelf{level}")

        if self.target_obj.get_pose().p[0] < 0:
            xlim = [self.office_info["table_lims"][0] + 0.19, 0.15]
        else:
            xlim = [0.15, self.office_info["table_lims"][2] - 0.04]
        
        self.milktea_id = np.random.choice(self._target_ids("office", "101_milk-tea"))
        self.milktea = rand_create_actor(
            self,
            xlim=xlim,
            ylim=[self.office_info["table_lims"][1] + 0.04, self.office_info["shelf_lims"][1]-0.05],
            zlim=[self.office_info["table_height"]],
            modelname="101_milk-tea",
            rotate_rand=True,
            rotate_lim=[0, 1, 0],
            qpos=euler2quat(np.pi/2,0, 0, axes='sxyz'),
            convex=True,
            model_id=self.milktea_id,
            is_static=False,
        )
        self.collision_list.append({
            "actor": self.milktea,
            "collision_path": f"{os.environ['ASSETS_ROOT']}/objects/101_milk-tea/collision/base{self.milktea_id}.glb",
        })
        self.add_prohibit_area(self.milktea, padding=0, area="table")
    
        half_size = [0.03, 0.03, 0.0005]
        p = self.milktea.get_pose().p.tolist()
        p[0] -= 0.15
        p[2] = self.office_info["table_height"] - 0.001
        des_obj_pose = sapien.Pose(p=p, q=[1, 0, 0, 0])
        self.des_obj = create_box(
            scene=self,
            pose=des_obj_pose,
            half_size=half_size,
            color=(0, 0, 1),
            name="box",
            is_static=True,
        )
        self.add_prohibit_area(self.des_obj, padding=0.04, area="table")

        self.des_obj_pose = des_obj_pose.p.tolist() + [0, 0, 0.7071, 0.7071]
        self.des_obj_pose[2] += 0.02


    def play_once(self):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag("left") if self.target_obj.get_pose().p[0] < 0 else ArmTag("right")

        self.move(self.grasp_actor(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.1, grasp_dis=0.04, contact_point_id=3))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.01))
        self.attach_object(self.target_obj, f"{os.environ['ASSETS_ROOT']}/objects/073_rubikscube/collision/base{self.cube_id}.glb", str(arm_tag))

        self.move(
            self.place_actor(
                self.target_obj,
                arm_tag=arm_tag,
                target_pose=self.des_obj_pose,
                constrain="align",
                pre_dis=0.02,
                dis=0,
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
