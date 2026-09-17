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


class organize_table(Office_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)
    
    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj_1.get_name(), self.target_obj_2.get_name(), self.target_obj_3.get_name(), self.des_obj_1.get_name(), self.des_obj_2.get_name()}

    def load_actors(self):
        # des_obj_pose_3 ------------------------------------------------------------
        target_rand_pose = rand_pose(
            xlim=[self.office_info["shelf_lims"][0]+self.office_info["shelf_padding"], self.office_info["shelf_lims"][2]-self.office_info["shelf_padding"]],
            ylim=[self.office_info["shelf_lims"][1] + 0.03],
            zlim = [self.office_info["shelf_heights"][0]],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )

        half_size = [0.025, 0.025, 0.0005]
        self.des_obj_3 = create_box(
            scene=self,
            pose=target_rand_pose,
            half_size=half_size,
            color=(0, 0, 1), # blue
            name="target",
            is_static=True,
        )
        self.des_obj_pose_3 = self.des_obj_3.get_pose().p.tolist()
        self.des_obj_pose_3[2] += 0.02 # raise target 0.02 meters
        self.add_prohibit_area(self.des_obj_3, padding=0.05, area=f"shelf0")

        # target_obj_1 ------------------------------------------------------------
        if self.des_obj_3.get_pose().p[0] < 0:
            xlim = [0.3, 0.45]
        else:
            xlim = [-0.45, -0.3]
        rand_pos = rand_pose(
            xlim=xlim,
            ylim=[-0.23, self.office_info["shelf_lims"][1]-0.11],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, 3.14, 0],
        )
        self.mouse_id = np.random.choice(self._target_ids("office", "047_mouse"))
        self.target_obj_1 = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="047_mouse",
            convex=True,
            model_id=self.mouse_id,
            scale=self.item_info['scales']['047_mouse'].get(f'{self.mouse_id}',None),
        )
        self.target_obj_1.set_mass(0.05)
        self.add_prohibit_area(self.target_obj_1, padding=0.03, area="table")

        # des_obj_pose_1 ------------------------------------------------------------
        target_rand_pose = rand_pose(
            xlim=[0],
            ylim=[-0.23, self.office_info["shelf_lims"][1]-0.14],
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
            name="des_obj_1",
            is_static=True,
        )
        self.add_prohibit_area(self.des_obj_1, padding=0.01, area="table")
        self.des_obj_pose_1 = self.des_obj_1.get_pose().p.tolist() + [0, 0, 0, 1]

        # target2 ------------------------------------------------------------
        center_x = (self.target_obj_1.get_pose().p[0] + self.des_obj_1.get_pose().p[0]) / 2
        center_y = (self.target_obj_1.get_pose().p[1] + self.des_obj_1.get_pose().p[1]) / 2
        self.stand_id = np.random.choice([1, 2], 1)[0]
        self.des_obj_2 = rand_create_actor(
            scene=self,
            xlim=[center_x],
            ylim=[center_y],
            zlim=[self.office_info["table_height"]],
            qpos=[0.7071, 0.7071, 0.0, 0.0],
            rotate_rand=True,
            rotate_lim=[0, np.pi / 6, 0],
            modelname="078_phonestand",
            convex=True,
            model_id=self.stand_id,
            is_static=False,
        )
        self.des_obj_2.set_mass(2)
        self.collision_list.append({
            "actor": self.des_obj_2,
            "collision_path": f"{os.environ['ASSETS_ROOT']}/objects/078_phonestand/collision/base{self.stand_id}.glb",
        })
        self.add_prohibit_area(self.des_obj_2, padding=0.01, area="table")

        # target_obj_2 ------------------------------------------------------------
        ori_quat = [
            [0.707, 0.707, 0, 0],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5, -0.5],
        ]
        if self.des_obj_3.get_pose().p[0] < 0:
            xlim = [-0.1,self.office_info["table_lims"][2]-0.05]
        else:
            xlim = [self.office_info["table_lims"][0]+0.05, 0.1]
        self.phone_id = np.random.choice(self._target_ids("office", "077_phone"))
        success, self.target_obj_2 = rand_create_cluttered_actor(
            scene=self.scene,
            xlim=xlim,
            ylim=[self.office_info["table_lims"][1]+0.05, self.office_info["shelf_lims"][1]-0.09],
            zlim=[self.office_info["table_height"]+0.01],
            modelname="077_phone",
            modelid=self.phone_id,
            modeltype="glb",
            rotate_rand=True,
            qpos=ori_quat[self.phone_id],
            rotate_lim=[0, 0.7, 0],
            obj_radius=0.04,
            z_offset=0,
            z_max=0.02,
            prohibited_area=self.prohibited_area["table"],
            constrained=False,
            is_static=False,
            size_dict=dict(),
        )
        if not success:
            raise RuntimeError("Failed to load target_obj_2")
        self.target_obj_2.set_mass(0.01)
        self.add_prohibit_area(self.target_obj_2, padding=0.01, area="table")

        # target_obj_3 ------------------------------------------------------------
        self.milktea_id = np.random.choice(self._target_ids("office", "101_milk-tea"))
        if self.des_obj_3.get_pose().p[0] < 0:
            xlim1 = [self.office_info["table_lims"][0]+self.target_objects_info["101_milk-tea"]["params"][f"{self.milktea_id}"]["radius"], 0.1]
        else:
            xlim1 = [-0.1, self.office_info["table_lims"][2]-self.target_objects_info["101_milk-tea"]["params"][f"{self.milktea_id}"]["radius"]]
        ylim1 = [self.office_info["table_lims"][1] + 0.3, self.office_info["shelf_lims"][1]-0.06]

        success, self.target_obj_3 = rand_create_cluttered_actor(
            scene=self.scene,
            xlim=xlim1,
            ylim=ylim1,
            zlim=[self.office_info["table_height"]],
            modelname="101_milk-tea",
            modelid=self.milktea_id,
            modeltype="glb",
            rotate_rand=False,
            rotate_lim=[0, 1, 0],
            qpos=euler2quat(np.pi/2,0, 0, axes='sxyz'),
            obj_radius=0.03,
            z_offset=0,
            z_max=0.1,
            prohibited_area=self.prohibited_area["table"],
            constrained=False,
            is_static=False,
            size_dict=dict(),
            scale = self.item_info["scales"]["101_milk-tea"]
        )
        if not success:
            raise RuntimeError("Failed to load target_obj")

        self.target_obj_3.set_mass(0.06)
        self.add_prohibit_area(self.target_obj_3, padding=0.01)
        self.add_operating_area(self.target_obj_3.get_pose().p)
        self.collision_list.append({
            "actor": self.target_obj_3,
            "collision_path": f"{os.environ['ASSETS_ROOT']}/objects/101_milk-tea/collision/base{self.milktea_id}.glb",
            "pose": self.target_obj_3.get_pose(),
        })
        self.target_obj_3_pose = self.target_obj_3.get_pose()
        self.target_obj_3_pose = np.concatenate([self.target_obj_3_pose.p, self.target_obj_3_pose.q]).tolist()

        self.des_obj_pose_3 += self.target_obj_3.get_pose().q.tolist()


    def play_once(self):
        # Determine which arm to use based on target_obj_1 position (right if on right side, left otherwise)
        armL = ArmTag("left")
        armR = ArmTag("right")
        arms = [
            armL if self.target_obj_1.get_pose().p[0] < 0 else armR,
            armL if self.des_obj_2.get_pose().p[0] < 0 else armR,
            armL if self.des_obj_3.get_pose().p[0] < 0 else armR,
        ]

        # target_obj_1 --------------------------------------------------
        self.enable_table(enable=False)
        self.move(self.grasp_actor(self.target_obj_1, arm_tag=arms[0], pre_grasp_dis=0.1, grasp_dis = 0.01))

        # Lift the target_obj_1 upward by 0.1 meters in z-direction
        self.move(self.move_by_displacement(arm_tag=arms[0], z=0.1))

        self.attach_object(self.target_obj_1, f"{os.environ['ASSETS_ROOT']}/objects/047_mouse/collision/base{self.mouse_id}.glb", str(arms[0]))
        # self.move(self.move_by_displacement(arm_tag=arms[0], x=-0.1, y=-0.1))
        
        # Place the target_obj_1 at the target location with alignment constraint
        self.move(
            self.place_actor(
                self.target_obj_1,
                arm_tag=arms[0],
                target_pose=self.des_obj_pose_1,
                constrain="align",
                pre_dis=0.03,
                dis=0.02,
            ))
        self.detach_object(arms_tag=str(arms[0]))
        self.enable_table(enable=True)

        if arms[0] != arms[1]:
            self.move(self.back_to_origin(arms[0]))

        # target_obj_2 --------------------------------------------------
        _, actions = self.grasp_actor(self.target_obj_2, arm_tag=arms[1], pre_grasp_dis=0.08, grasp_dis=0.01)
        self.move((arms[1], [actions[0]]))
        self.enable_table(enable=False)
        self.move((arms[1], actions[1:]))
        self.move(self.move_by_displacement(arm_tag=arms[1], z=0.03))
        
        self.attach_object(self.target_obj_2, f"{os.environ['ASSETS_ROOT']}/objects/077_phone/collision/base{self.phone_id}.glb", str(arms[1]))
        self.enable_table(enable=True)

        # Get des_obj_2's functional point as target for placement
        stand_func_pose = self.des_obj_2.get_functional_point(0)

        # Place the target_obj_2 onto the des_obj_2's functional point with alignment constraint
        self.move(
            self.place_actor(
                self.target_obj_2,
                arm_tag=arms[1],
                target_pose=stand_func_pose,
                functional_point_id=0,
                dis=0,
                constrain="align",
                pre_dis=0.05,
            ))
        self.detach_object(arms_tag=str(arms[1]))
        self.move(self.move_by_displacement(arm_tag=arms[1], y=-0.03))
        self.collision_list.append({
            "actor": self.target_obj_2,
            "collision_path": f"{os.environ['ASSETS_ROOT']}/objects/077_phone/collision/base{self.phone_id}.glb",
        })
        self.update_world()

        if arms[1] != arms[2]:
            self.move(self.back_to_origin(arms[1]))
    
        # target_obj_3 --------------------------------------------------
        _, actions = self.grasp_actor(self.target_obj_3, arm_tag=arms[2], pre_grasp_dis=0.05, grasp_dis=0.02, contact_point_id=2)
        actions[0].target_pose[2] += 0.04
        actions[1].target_pose[2] += 0.04
        self.move((arms[2], [actions[0]]))

        # Disable obstacle collision for the target_obj_3
        self.enable_obstacle(False, [f"101_milk-tea_{self.target_obj_3_pose}_{self.seed}"])

        self.move((arms[2], actions[1:]))

        # Lift the target_obj_1 upward by 0.1 meters in z-direction
        self.move(self.move_by_displacement(arm_tag=arms[2], z=0.01))

        self.attach_object(self.target_obj_3, f"{os.environ['ASSETS_ROOT']}/objects/101_milk-tea/collision/base{self.milktea_id}.glb", str(arms[2]))

        # Place the target_obj_1 at the target location with alignment constraint
        action = self.place_actor(
                self.target_obj_3,
                arm_tag=arms[2],
                target_pose=self.des_obj_pose_3,
                constrain="free",
                pre_dis=0.0,
                dis=0.0,
                local_up_axis=[0,0,1]
            )
        # action[1][0].target_pose[2] += 0.03
        self.move(action)

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

        end_pose_actual2 = np.array(self.target_obj_2.get_pose().p)
        end_pose_desired2 = np.array(self.des_obj_2.get_functional_point(0))
        end_pose_desired2[1] -= 0.01
        end_pose_desired2[2] += 0.05

        end_pose_actual3 = self.target_obj_3.get_pose().p[2]
        end_pose_desired3 = self.office_info["shelf_heights"][0]

        eps1 = 0.02
        eps2 = 0.02
        eps3 = 0.045
        eps4 = 0.05
        eps5 = 0.04
        eps6 = 0.02

        return (np.all(abs(end_pose_actual1[:2] - end_pose_desired1[:2]) < np.array([eps1, eps2]))
                and np.all(abs(end_pose_actual2[:3] - end_pose_desired2[:3]) < np.array([eps3, eps4, eps5]))
                and abs(end_pose_actual3 - end_pose_desired3) < eps6
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
