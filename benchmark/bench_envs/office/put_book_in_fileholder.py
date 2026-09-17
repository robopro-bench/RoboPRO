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


class put_book_in_fileholder(Office_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)
    
    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        holder_pose = self.file_holder.get_pose().p
        holder_pose[1]-= 0.19  
        self.prohibited_area["table"].append([holder_pose[0]-0.11, holder_pose[1]-0.1, holder_pose[0]+0.11, holder_pose[1]+0.1])

        model_id = np.random.choice(self._target_ids("office", "043_book"))
        
        # target_obj ------------------------------------------------------------
        ylim = [self.shelf.get_pose().p[1]-0.07]
        if self.arr_v == 0:
            xlims = [-0.15, self.office_info["shelf_lims"][2]-self.office_info["shelf_padding"]]
            level = 0
        elif self.arr_v == 2:
            xlims = [self.office_info["shelf_lims"][0]+self.office_info["shelf_padding"], 0.15]
            level = 0
        else:        
            level = np.random.choice([0,1])
            if level == 0:
                xlims = [self.office_info["shelf_lims"][0]+self.office_info["shelf_padding"], 0.15]
            else:
                xlims = [self.office_info["shelf_lims"][0]+self.office_info["shelf_padding"], 0.02]
        zlim = [self.office_info["shelf_heights"][level]+0.1]

        self.target_obj = rand_create_actor(
            self,
            xlim=xlims,
            ylim=ylim,
            zlim=zlim,
            modelname="043_book",
            rotate_rand=False,
            qpos=euler2quat(np.pi, 0, np.pi/2, axes='sxyz'),
            convex=True,
            model_id=model_id,
            is_static=False,
            scale = self.item_info['scales']['043_book'][f'{model_id}']
        )
        self.target_obj.set_mass(0.05)
        self.stabilize_object(self.target_obj)
        center_x = self.target_obj.get_pose().p[0] + 0.02
        center_y = self.target_obj.get_pose().p[1]
        self.prohibited_area[f"shelf{level}"].append([center_x-0.035, center_y-0.06, center_x+0.035, center_y+0.06])

        self.des_obj_pose = self.file_holder.get_pose().p.tolist() + euler2quat(np.pi, np.pi/12, np.pi/2, axes='sxyz').tolist()
        self.des_obj_pose[1]-= 0.06
        self.des_obj_pose[2]+=0.08

    def play_once(self):
        arm_tag2 = ArmTag("right") if self.arr_v == 0 else ArmTag("left")

        # target_obj ------------------------------------------------------------
        self.move(self.grasp_actor(self.target_obj, arm_tag=arm_tag2, pre_grasp_dis=0.04, grasp_dis=0.02))

        self.move(self.move_by_displacement(arm_tag=arm_tag2, z=0.015))
        self.attach_object(self.target_obj, f"{os.environ['ASSETS_ROOT']}/objects/043_book/collision/base0.glb", str(arm_tag2))

        self.move(
            self.place_actor(
                self.target_obj,
                arm_tag=arm_tag2,
                target_pose=self.des_obj_pose,
                constrain="align",
                pre_dis=0.05,
                local_up_axis=[0,1,0]
            ))
        
        # self.move(self.move_by_displacement(arm_tag=arm_tag2, y=-0.05))

        # Record information about the objects and arm used in the task
        # self.info["info"] = {
        #     "{A}": f"047_mouse/base{self.mouse_id}",
        #     "{B}": f"{self.color_name}",
        #     "{a}": str(arm_tag),
        # }
        # return self.info

    def check_success(self):
        end_pose_actual = self.target_obj.get_pose().p
        end_pose_desired = self.file_holder.get_pose().p
        end_pose_desired[1] -= 0.07
        end_pose_desired[2] += 0.05
        eps1 = 0.02
        eps2 = 0.05
        eps3 = 0.04

        return (np.all(abs(end_pose_actual[:3] - end_pose_desired[:3]) < np.array([eps1, eps2, eps3]))
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
