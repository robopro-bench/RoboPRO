# from envs._base_task import Base_Task
from bench_envs.office._office_base_task import Office_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class put_mouse_on_pad(Office_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name(), self.des_obj.get_name()}

    def load_actors(self):
        rand_pos = rand_pose(
            xlim=[-0.45, 0.45],
            ylim=[-0.23, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, 3.14, 0],
        )
        while abs(rand_pos.p[0]) < 0.3:
            rand_pos = rand_pose(
            xlim=[-0.45, 0.45],
            ylim=[-0.23, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, 3.14, 0],
        )

        self.mouse_id = np.random.choice(self._target_ids("office", "047_mouse"))
        self.target_obj = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="047_mouse",
            convex=True,
            model_id=self.mouse_id,
            scale=self.item_info['scales']['047_mouse'].get(f'{self.mouse_id}',None),
        )
        self.target_obj.set_mass(0.05)

        xlim=[0]
        
        target_rand_pose = rand_pose(
            xlim=xlim,
            ylim=[-0.23, 0.05],
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
        self.des_obj = create_box(
            scene=self,
            pose=target_rand_pose,
            half_size=half_size,
            color=self.color_value,
            name="box",
            is_static=True,
        )
        self.add_prohibit_area(self.des_obj, padding=0.01, area="table")
        self.add_prohibit_area(self.target_obj, padding=0.02, area="table")
        # Construct des_obj pose with position from des_obj object and identity orientation
        self.des_obj_pose = self.des_obj.get_pose().p.tolist() + [0, 0, 0, 1]
        self.des_obj_pose[2] += 0.02

        if np.random.rand() > self.clean_background_rate and self.obstacle_density >0 and self.cluttered_table:
            self.obstacle_density = max(0, self.obstacle_density-1)  # Ensure non-negative density
            # ------------------------------------------------------------
            center_x = (self.target_obj.get_pose().p[0] + self.des_obj.get_pose().p[0]) / 2
            center_y = (self.target_obj.get_pose().p[1] + self.des_obj.get_pose().p[1]) / 2
            id_list = [i for i in range(4)]
            self.milk_box_id = np.random.choice(id_list)
            self.milk_box = rand_create_actor(
                self,
                xlim=[center_x],
                ylim=[center_y],
                modelname="038_milk-box",
                rotate_rand=True,
                rotate_lim=[0, 1, 0],
                qpos=[0.66, 0.66, -0.25, -0.25],
                convex=True,
                model_id=self.milk_box_id,
            )
            
            self.milk_box.set_mass(0.1)
            self.add_prohibit_area(self.milk_box, padding=0.01, area="table")
            self.collision_list.append({
                "actor": self.milk_box,
                "collision_path": f"{os.environ['ASSETS_ROOT']}/objects/038_milk-box/collision/base{self.milk_box_id}.glb",
            })

    def play_once(self):
        # Determine which arm to use based on target_obj position (right if on right side, left otherwise)
        arm_tag = ArmTag("right" if self.target_obj.get_pose().p[0] > 0 else "left")

        # Grasp the target_obj with the selected arm
        self.grasp_actor_from_table(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.1)

        # Lift the target_obj upward by 0.1 meters in z-direction
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1))

        self.attach_object(self.target_obj, f"{os.environ['ASSETS_ROOT']}/objects/047_mouse/collision/base{self.mouse_id}.glb", str(arm_tag))
        self.enable_table(enable=True)
        
        # Place the target_obj at the des_obj location with alignment constraint
        self.move(
            self.place_actor(
                self.target_obj,
                arm_tag=arm_tag,
                target_pose=self.des_obj_pose,
                constrain="align",
                pre_dis=0.05,
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
        eps1 = 0.02
        eps2 = 0.02

        return (np.all(abs(end_pose_actual[:2] - end_pose_desired[:2]) < np.array([eps1, eps2]))
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
