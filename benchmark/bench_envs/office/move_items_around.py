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


class move_items_around(Office_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)
    
    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj_1.get_name(), self.target_obj_2.get_name(), self.target_obj_3.get_name(), self.des_obj_1.get_name(), self.des_obj_3.get_name()}

    def load_actors(self):
        # target_obj_1 ------------------------------------------------------------
        self.milktea_id = np.random.choice(self._target_ids("office", "101_milk-tea"))
        self.target_obj_1 = rand_create_actor(
            self,
            xlim=[self.office_info["shelf_lims"][0]+self.office_info["shelf_padding"], self.office_info["shelf_lims"][2]-self.office_info["shelf_padding"]],
            ylim=[self.office_info["shelf_lims"][1]+0.04],
            zlim=[self.office_info["shelf_heights"][0]],
            modelname="101_milk-tea",
            rotate_rand=False,
            qpos=euler2quat(np.pi/2,0, 0, axes='sxyz'),
            convex=True,
            model_id=self.milktea_id,
            is_static=False,
        )
        self.target_obj_1.set_mass(0.06)
        self.stabilize_object(self.target_obj_1)
        self.add_prohibit_area(self.target_obj_1, padding=0.01, area="shelf0")
        
        # des_obj_pose_1 ------------------------------------------------------------
        if self.target_obj_1.get_pose().p[0] < 0:
            xlim = [self.office_info["table_lims"][0]+0.04, 0.1]
        else:
            xlim = [-0.1, self.office_info["table_lims"][2]-0.04]
        self.des_obj_1 = rand_create_actor(
            self,
            xlim=xlim,
            ylim=[0.02, self.office_info["shelf_lims"][1]-0.05],
            zlim=[self.office_info["table_height"]],
            modelname="019_coaster",
            rotate_rand=False,
            qpos=euler2quat(np.pi/2,0, np.pi/2, axes='sxyz'),
            convex=True,
            model_id=0,
            is_static=False,
        )
        self.add_prohibit_area(self.des_obj_1, padding=0.01, area="table")
        self.add_operating_area(self.des_obj_1.get_pose().p)
        self.des_obj_pose_1 = self.des_obj_1.get_pose().p.tolist() + self.target_obj_1.get_pose().q.tolist()   # wxyz
        self.des_obj_pose_1[2] += 0.03

        # des_obj_pose_2 ------------------------------------------------------------
        target_rand_pose = rand_pose(
            xlim=[self.office_info["shelf_lims"][0]+self.office_info["shelf_padding"], self.office_info["shelf_lims"][2]-self.office_info["shelf_padding"]],
            ylim=[self.office_info["shelf_lims"][1] + 0.055],
            zlim = [self.office_info["shelf_heights"][1]],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )

        half_size = [0.05, 0.05, 0.0005]
        self.des_obj_2 = create_box(
            scene=self,
            pose=target_rand_pose,
            half_size=half_size,
            color=(0, 0, 1), # blue
            name="des_obj_2",
            is_static=True,
        )
        self.des_obj_pose_2 = self.des_obj_2.get_pose().p.tolist() + euler2quat(np.pi/2,np.pi, 0, axes='sxyz').tolist()
        self.des_obj_pose_2[2] += 0.05 # raise target 0.02 meters
        self.add_prohibit_area(self.des_obj_2, padding=0.03, area=f"shelf1")

        # target_obj_2 ------------------------------------------------------------
        if self.des_obj_2.get_pose().p[0] < 0:
            xlim = [self.office_info["table_lims"][0]+0.04, 0.1]
        else:
            xlim = [-0.1, self.office_info["table_lims"][2]-0.04]

        self.rubikscube_id = np.random.choice(self._target_ids("office", "073_rubikscube"))
        success, self.target_obj_2 = rand_create_cluttered_actor(
            scene=self.scene,
            xlim=xlim,
            ylim=[self.office_info["table_lims"][1]+0.04, self.office_info["shelf_lims"][1]-0.04],
            zlim=[self.office_info["table_height"]],
            modelname="073_rubikscube",
            modelid=self.rubikscube_id,
            modeltype="glb",
            rotate_rand=True,
            rotate_lim=[0, 0.5, 0],
            qpos=[0, 0, 0.7071, 0.7071],
            obj_radius=0.025,
            z_offset=0,
            z_max=0.05,
            prohibited_area=self.prohibited_area["table"],
            constrained=False,
            is_static=False,
            size_dict=dict(),
        )
        if not success:
            raise RuntimeError("Failed to load target_obj_2")
        self.target_obj_2.set_mass(0.1)
        self.add_prohibit_area(self.target_obj_2, padding=0.03, area="table")
    
        # target3 ------------------------------------------------------------
        if self.arr_v == 0:
            xlim = [-0.05, self.office_info["table_lims"][2]-0.05]
        else:
            xlim = [self.office_info["table_lims"][0]+0.05, 0.05]


        self.target3_book_id = 1
        success, self.des_obj_3 = rand_create_cluttered_actor(
            scene=self.scene,
            xlim=xlim,
            ylim=[self.office_info["table_lims"][1]+0.15, self.office_info["shelf_lims"][1]-0.07],
            zlim=[self.office_info["table_height"]],
            modelname="043_book",
            modelid=self.target3_book_id,
            modeltype="glb",
            rotate_rand=False,
            qpos=euler2quat(np.pi/2,0, 0, axes='sxyz'),
            obj_radius=0.08,
            z_offset=0,
            z_max=0.02,
            prohibited_area=self.prohibited_area["table"],
            constrained=False,
            is_static=False,
            size_dict=dict(),
            scale=self.item_info['scales']['043_book'][f'{self.target3_book_id}'],
        )
        if not success:
            raise RuntimeError("Failed to load des_obj_3")
        self.des_obj_3.set_name("043_book_ref")
        self.des_obj_3.set_mass(0.1)
        self.add_prohibit_area(self.des_obj_3, padding=0.01, area="table")
        self.add_operating_area(self.des_obj_3.get_pose().p)
        self.target3 = self.des_obj_3.get_pose().p.tolist() + euler2quat(np.pi, np.pi/6, np.pi/2, axes='sxyz').tolist()
        self.target3[2]+=0.05
        self.target3[1]+=0.015
        
        # target_obj_3 ------------------------------------------------------------
        ylim = [self.file_holder.get_pose().p[1]]
        zlim = [self.office_info["file_holder_heights"][1]+0.1]
        xlim = [self.office_info["file_holder_lims"][0]+0.08, self.office_info["file_holder_lims"][2]-0.08]
        book_id = 0
        self.target_obj_3 = rand_create_actor(
            self,
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            modelname="043_book",
            rotate_rand=False,
            qpos=euler2quat(np.pi, 0, np.pi/2, axes='sxyz'),
            convex=True,
            model_id=book_id,
            is_static=False,
            scale = self.item_info['scales']['043_book'][f'{book_id}']
        )
        self.target_obj_3.set_mass(0.05)
        self.stabilize_object(self.target_obj_3)


    def play_once(self):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        armL = ArmTag("left")
        armR = ArmTag("right")
        arms = [
            armL if self.target_obj_1.get_pose().p[0] < 0 else armR,
            armL if self.des_obj_2.get_pose().p[0] < 0 else armR,
            armR if self.arr_v == 0 else armL,
        ]

        # target_obj_1 --------------------------------------------------
        action = self.grasp_actor(self.target_obj_1, arm_tag=arms[0], pre_grasp_dis=0.1, grasp_dis=0.02, contact_point_id=2)
        action[1][0].target_pose[2] += 0.03
        action[1][1].target_pose[2] += 0.03
        self.move(action)

        # Lift the mouse upward by 0.1 meters in z-direction
        self.move(self.move_by_displacement(arm_tag=arms[0], z=0.01, y=-0.08))

        self.attach_object(self.target_obj_1, f"{os.environ['ASSETS_ROOT']}/objects/101_milk-tea/collision/base{self.milktea_id}.glb", str(arms[0]))

        # Place the mouse at the target location with alignment constraint
        action = self.place_actor(
                self.target_obj_1,
                arm_tag=arms[0],
                target_pose=self.des_obj_pose_1,
                constrain="free",
                pre_dis=0.0,
                dis=0.0,
                local_up_axis=[0,0,1]
            )
        action[1][0].target_pose[2] += 0.03
        self.move(action)
        self.detach_object(arms_tag=str(arms[0]))
        self.move(self.move_by_displacement(arm_tag=arms[0], y=-0.03))
        self.move(self.move_by_displacement(arm_tag=arms[0], y=-0.02, z = 0.05))
        self.collision_list.append({
            "actor": self.target_obj_1,
            "collision_path": f"{os.environ['ASSETS_ROOT']}/objects/101_milk-tea/collision/base{self.milktea_id}.glb",
        })
        self.update_world()

        if arms[0] != arms[1]:
            self.move(self.back_to_origin(arms[0]))

        # rubics cube --------------------------------------------------
        action = self.grasp_actor(self.target_obj_2, arm_tag=arms[1], pre_grasp_dis=0.1, contact_point_id=3)
        if action:
            action[1][1].target_pose[2] += 0.04 # grasp center of box
        self.move(action)

        # Lift the box upward by 0.1 meters in z-direction
        self.move(self.move_by_displacement(arm_tag=arms[1], z=0.03))

        self.attach_object(self.target_obj_2, f"{os.environ['ASSETS_ROOT']}/objects/073_rubikscube/collision/base{self.rubikscube_id}.glb", str(arms[1]))

        self.move(
            self.place_actor(
                self.target_obj_2,
                arm_tag=arms[1],
                target_pose=self.des_obj_pose_2,
                constrain="align",
                pre_dis=0.08,
                dis=-0.02,
            ))

        self.detach_object(arms_tag=str(arms[1]))
        self.move(self.move_by_displacement(arm_tag=arms[0], y=-0.03))

        if arms[1] != arms[2]:
            self.move(self.back_to_origin(arms[1]))
    
        # target_obj_3 --------------------------------------------------
        self.move(self.grasp_actor(self.target_obj_3, arm_tag=arms[2], pre_grasp_dis=0.04, grasp_dis=0.01))
        self.move(self.move_by_displacement(arm_tag=arms[2], y=-0.01, z=0.01))
        self.attach_object(self.target_obj_3, f"{os.environ['ASSETS_ROOT']}/objects/043_book/collision/base0.glb", str(arms[2]))
        self.enable_table(enable=False)

        self.move(
            self.place_actor(
                self.target_obj_3,
                arm_tag=arms[2],
                target_pose=self.target3,
                constrain="align",
                pre_dis=0.01,
                dis=0.005,
                local_up_axis=[0,1,0]
            ))
        

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

        end_pose_actual2 = self.target_obj_2.get_pose().p[2]
        end_pose_desired2 = self.office_info["shelf_heights"][1]+0.03 # 0.03 for height of rubiks cube

        end_pose_actual3 = self.target_obj_3.get_pose().p
        end_pose_desired3 = self.des_obj_3.get_pose().p

        eps1 = 0.02
        eps2 = 0.02
        eps3 = 0.02
        eps4 = 0.02
        eps5 = 0.04

        return (np.all(abs(end_pose_actual1[:2] - end_pose_desired1[:2]) < np.array([eps1, eps2]))
                and abs(end_pose_actual2 - end_pose_desired2) < eps3
                and np.all(abs(end_pose_actual3[:2] - end_pose_desired3[:2]) < np.array([eps4, eps5]))
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
