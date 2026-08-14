# from envs._base_task import Base_Task
from bench_envs.office._office_base_task import Office_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob

from transforms3d.euler import euler2quat

class put_stapler_next_to_mouse(Office_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        # mouse ------------------------------------------------------------
        rand_pos = rand_pose(
            xlim=[self.office_info["table_lims"][0]+0.04, self.office_info["table_lims"][2]-0.04],
            ylim=[self.office_info["table_lims"][1]+0.04, self.office_info["shelf_lims"][1]-0.19],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, np.pi/6, 0],
        )

        self.mouse_id = np.random.choice(self._target_ids("office", "047_mouse"))
        self.mouse = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="047_mouse",
            convex=True,
            model_id=self.mouse_id,
            scale=self.item_info['scales']['047_mouse'].get(f'{self.mouse_id}',None),
        )
        self.mouse.set_mass(0.05)
        self.add_prohibit_area(self.mouse, padding=0.01, area="table")

        # des_obj ------------------------------------------------------------
        half_size = [0.04, 0.04, 0.0005]
        p = self.mouse.get_pose().p.tolist()
        p[1] += 0.15
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
        self.add_prohibit_area(self.des_obj, padding=0.03, area="table")
        self.des_obj_pose = self.des_obj.get_pose().p.tolist() + euler2quat(np.pi/2, 0, np.pi, axes='sxyz').tolist()
        self.des_obj_pose[2] += 0.02

        # target_obj ------------------------------------------------------------
        if self.mouse.get_pose().p[0] < 0:
            xlim = [self.office_info["table_lims"][0]+0.04, 0.1]
        else:
            xlim = [-0.1, self.office_info["table_lims"][2]-0.04]
        self.stapler_id = np.random.choice(self._target_ids("office", "048_stapler"))
        success, self.target_obj = rand_create_cluttered_actor(
            scene=self.scene,
            xlim=xlim,
            ylim=[self.office_info["table_lims"][1]+0.04, self.office_info["shelf_lims"][1]-0.04],
            zlim=[self.office_info["table_height"]],
            modelname="048_stapler",
            modelid=self.stapler_id,
            modeltype="glb",
            rotate_rand=True,
            rotate_lim=[0, np.pi/4, 0],
            qpos=euler2quat(np.pi/2, 0, np.pi, axes='sxyz'),
            obj_radius=0.03,
            z_offset=0,
            z_max=0.04,
            prohibited_area=self.prohibited_area["table"],
            constrained=False,
            is_static=False,
            size_dict=dict(),
        )
        if not success:
            raise RuntimeError("Failed to load target_obj")
        self.add_prohibit_area(self.target_obj, padding=0.02, area="table")

    def play_once(self):
        # Determine which arm to use based on target_obj position (right if on right side, left otherwise)
        arm_tag = ArmTag("right" if self.mouse.get_pose().p[0] > 0 else "left")

        self.grasp_actor_from_table(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.05, contact_point_id=[0,1])
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.03))
        self.attach_object(self.target_obj, f"{os.environ['ASSETS_ROOT']}/objects/048_stapler/collision/base{self.stapler_id}.glb", str(arm_tag))
        self.enable_table(enable=True)

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
