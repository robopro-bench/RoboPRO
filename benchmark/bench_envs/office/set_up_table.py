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


class set_up_table(Office_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)
    
    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj_1.get_name(), self.target_obj_2.get_name()}

    def load_actors(self):

        # set up cabinet
        self.add_cabinet_collision()
        limit = self.cabinet.get_qlimits()[0]
        self.cabinet.set_qpos([limit[1],0,0])

        # add prohibited in front of file holder
        holder_pose = self.file_holder.get_pose().p
        holder_pose[1]-= 0.19  
        self.prohibited_area["table"].append([holder_pose[0]-0.11, holder_pose[1]-0.1, holder_pose[0]+0.11, holder_pose[1]+0.1])
        
        model_id = np.random.choice(self._target_ids("office", "043_book"))
        
        # target_obj_2 ------------------------------------------------------------
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

        self.target_obj_2 = rand_create_actor(
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
        self.target_obj_2.set_mass(0.05)
        self.stabilize_object(self.target_obj_2)
        center_x = self.target_obj_2.get_pose().p[0] + 0.02
        center_y = self.target_obj_2.get_pose().p[1]
        self.prohibited_area[f"shelf{level}"].append([center_x-0.035, center_y-0.06, center_x+0.035, center_y+0.06])

        self.target = self.file_holder.get_pose().p.tolist() + euler2quat(np.pi, np.pi/12, np.pi/2, axes='sxyz').tolist()
        self.target[1]-= 0.06
        self.target[2]+=0.08
        self.add_operating_area(self.target[:3])

        # target_obj_1 ------------------------------------------------------------
        self.mouse_id = np.random.choice(self._target_ids("office", "047_mouse"))
        pose = self.cabinet.get_pose().p
        pose[1]-= 0.2
        pose[2] = self.office_info["table_height"]+0.03
        self.target_obj_1 = rand_create_actor(
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


        # target ------------------------------------------------------------
        if self.arr_v == 2:
            xlim = [self.office_info["table_lims"][0]+0.04, 0.15]
        else:
            xlim = [-0.15, self.office_info["table_lims"][2]-0.04]
        target_rand_pose = rand_pose(
            xlim=xlim,
            ylim=[self.office_info["table_lims"][1]+0.07, -0.1],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )

        colors = {
            "Red": (1, 0, 0),
            "Green": (0, 1, 0),
            "Blue": (0, 0, 1),
            "Yellow": (1, 1, 0),
            "Cyan": (0, 1, 1),
            "Magenta": (1, 0, 1),
            "Black": (0, 0, 0),
            "Gray": (0.5, 0.5, 0.5),
        }

        color_items = list(colors.items())
        color_index = np.random.choice(len(color_items))
        self.color_name, self.color_value = color_items[color_index]

        half_size = [0.06, 0.06, 0.0005]
        self.des_obj_1 = create_box(
            scene=self,
            pose=target_rand_pose,
            half_size=half_size,
            color=self.color_value,
            name="box",
            is_static=True,
        )
        self.add_prohibit_area(self.des_obj_1, padding=0.01, area="table")
        # Construct target pose with position from target object and identity orientation
        self.des_obj_pose_1 = self.des_obj_1.get_pose().p.tolist() + [0, 0, 0, 1]

    def play_once(self):
        # Determine which arm to use based on target_obj_1 position (right if on right side, left otherwise)
        arm_tag1 = ArmTag("left") if self.arr_v == 2 else ArmTag("right")
        arm_tag2 = ArmTag("right") if self.arr_v == 0 else ArmTag("left")

        # target_obj_1 ------------------------------------------------------------
        self.move(self.grasp_actor(self.target_obj_1, arm_tag=arm_tag1, pre_grasp_dis=0.05,grasp_dis=0.02))
        # self.move(self.move_by_displacement(arm_tag=arm_tag1, z=0.1))
        self.attach_object(self.target_obj_1, f"{os.environ['ASSETS_ROOT']}/objects/047_mouse/collision/base{self.mouse_id}.glb", str(arm_tag1))
        self.move(
            self.place_actor(
                self.target_obj_1,
                arm_tag=arm_tag1,
                target_pose=self.des_obj_pose_1,
                constrain="align",
                pre_dis=0.03,
                dis=0.02,
            ))
        self.detach_object(arms_tag=str(arm_tag1))

        arm_tag, actions = self.grasp_actor(self.cabinet, arm_tag=arm_tag1, pre_grasp_dis=0.05, grasp_dis=0.025)

        self.move((arm_tag1, [actions[0]]))
        self.enable_drawer(enable=False)
        self.move((arm_tag1, actions[1:]))

        # Pull the drawer
        for _ in range(3):
            self.move(self.move_by_displacement(arm_tag=arm_tag1, y=0.0633))
        
        self.move(self.open_gripper(arm_tag=arm_tag1))
        
        if arm_tag1 != arm_tag2:
            self.move(self.back_to_origin(arm_tag1))
        else:
            self.move(self.move_by_displacement(arm_tag=arm_tag1, y=-0.04))

        # target_obj_2 ------------------------------------------------------------
        self.move(self.grasp_actor(self.target_obj_2, arm_tag=arm_tag2, pre_grasp_dis=0.04, grasp_dis=0.02))

        self.move(self.move_by_displacement(arm_tag=arm_tag2, z=0.015, y=-0.04))
        self.attach_object(self.target_obj_2, f"{os.environ['ASSETS_ROOT']}/objects/043_book/collision/base0.glb", str(arm_tag2))

        self.move(
            self.place_actor(
                self.target_obj_2,
                arm_tag=arm_tag2,
                target_pose=self.target,
                constrain="align",
                pre_dis=0.05,
                local_up_axis=[0,1,0]
            ))
        
        self.move(self.move_by_displacement(arm_tag=arm_tag2, y=-0.05))

        # Record information about the objects and arm used in the task
        # self.info["info"] = {
        #     "{A}": f"047_mouse/base{self.mouse_id}",
        #     "{B}": f"{self.color_name}",
        #     "{a}": str(arm_tag),
        # }
        # return self.info

    def check_success(self):
        end_pose_actual1 = self.target_obj_1.get_pose().p
        end_pose_desired1 = self.des_obj_1.get_pose().p

        end_pose_actual2 = self.cabinet.get_qpos()[0]
        end_pose_desired2 = self.cabinet.get_qlimits()[0][0]

        end_pose_actual3 = self.target_obj_2.get_pose().p
        end_pose_desired3 = self.file_holder.get_pose().p
        end_pose_desired3[1] -= 0.07
        end_pose_desired3[2] += 0.05

        eps1 = 0.02
        eps2 = 0.02
        eps3 = 0.03
        eps4 = 0.02
        eps5 = 0.05
        eps6 = 0.04


        return (np.all(abs(end_pose_actual1[:2] - end_pose_desired1[:2]) < np.array([eps1, eps2]))
                and abs(end_pose_desired2 - end_pose_actual2) < eps3
                and np.all(abs(end_pose_actual3 - end_pose_desired3) < np.array([eps4, eps5, eps6]))
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
