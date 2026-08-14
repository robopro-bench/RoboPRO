# from envs._base_task import Base_Task
from bench_envs.office._office_base_task import Office_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob
from transforms3d.euler import euler2quat

class put_milktea_next_to_laptop(Office_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        # add table as collision
        self.side = np.random.choice(["left", "right"])
        # laptop_id = np.random.choice([9748,9912,9960,9968,9992,9996,10040,10098,10101,10125,10211])
        laptop_id = 9912
        self.target_obj_id = np.random.choice(self._target_ids("office", "101_milk-tea"))

        if self.side == "left":
            xlim1 = [self.office_info["table_lims"][0]+0.08, 0]
            xlim2 = [self.office_info["table_lims"][0]+self.target_objects_info["101_milk-tea"]["params"][f"{self.target_obj_id}"]["radius"], 0.1]
        else:
            xlim1 = [-0.3, self.office_info["table_lims"][2]-0.27]
            xlim2 = [-0.1, self.office_info["table_lims"][2]-self.target_objects_info["101_milk-tea"]["params"][f"{self.target_obj_id}"]["radius"]]
        ylim1 = [self.office_info["table_lims"][1]+0.2, self.office_info["shelf_lims"][1]-0.1]
        ylim2 = [self.office_info["table_lims"][1] + 0.2, self.office_info["shelf_lims"][1]-0.05]
        

        self.laptop: ArticulationActor = rand_create_sapien_urdf_obj(
            scene=self,
            modelname="015_laptop",
            modelid=laptop_id,
            xlim=xlim1,
            ylim=ylim1,
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 6],
            qpos=[0.7, 0, 0, 0.7],
            fix_root_link=True,
        )
        self.laptop.set_mass(0.1)
        limit = self.laptop.get_qlimits()[0]
        self.laptop.set_qpos([limit[0] + (limit[1] - limit[0]) * 0.9])
        self.add_prohibit_area(self.laptop, padding=0.01)
        self.collision_list.append({
                "actor": self.laptop,
                "collision_path": f"{os.environ['ASSETS_ROOT']}/objects/015_laptop/9912/textured_objs/",
                "link": ["link_0", "link_1"],
                "files": ["original-5.obj"],
            })

        half_size = [0.03, 0.03, 0.0005]
        p = self.laptop.get_pose().p.tolist()
        p[0] += 0.15
        p[1] -= 0.05
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
        self.add_prohibit_area(self.des_obj, padding=0.01, area="table")
        self.add_operating_area(self.des_obj.get_pose().p)

        # 25cm center-to-center from the laptop. Sampler checks obj_radius+margin
        # against this AABB, so shrink the box to keep the on-axis gap at 25cm.
        milktea_obj_radius = 0.03
        milktea_obj_margin = 0.005
        min_milktea_laptop_dist = 0.25
        keepout = min_milktea_laptop_dist - (milktea_obj_radius + milktea_obj_margin)
        lx, ly = self.laptop.get_pose().p[:2]
        self.prohibited_area["table"].append([lx - keepout, ly - keepout, lx + keepout, ly + keepout])
        
        success, self.target_obj = rand_create_cluttered_actor(
            scene=self.scene,
            xlim=xlim2,
            ylim=ylim2,
            zlim=[self.office_info["table_height"]],
            modelname="101_milk-tea",
            modelid=self.target_obj_id,
            modeltype="glb",
            rotate_rand=False,
            rotate_lim=[0, 1, 0],
            qpos=euler2quat(np.pi/2,0, 0, axes='sxyz'),
            obj_radius=milktea_obj_radius,
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
        self.target_obj.set_mass(0.06)
        # Spawn orientation is upright by construction; remember which LOCAL axis
        # points world-up so check_success can measure tilt without assuming an
        # asset axis convention.
        self._target_init_quat = np.array(self.target_obj.get_pose().q, dtype=np.float64)
        self.add_prohibit_area(self.target_obj, padding=0.01, area="table")
        self.add_operating_area(self.target_obj.get_pose().p)

        self.des_obj_pose = des_obj_pose.p.tolist() + self.target_obj.get_pose().q.tolist()
        self.des_obj_pose[2] += 0.02

    def play_once(self):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag(self.side)

        # Grasp the mouse with the selected arm
        action = self.grasp_actor(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.1, grasp_dis=0.02, contact_point_id=2)
        action[1][0].target_pose[2] += 0.04
        action[1][1].target_pose[2] += 0.04
        self.move(action)

        # Lift the mouse upward by 0.1 meters in z-direction
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.01))

        self.attach_object(self.target_obj, f"{os.environ['ASSETS_ROOT']}/objects/101_milk-tea/collision/base{self.target_obj_id}.glb", str(arm_tag))

        # Place the mouse at the des_obj location with alignment constraint
        action = self.place_actor(
                self.target_obj,
                arm_tag=arm_tag,
                target_pose=self.des_obj_pose,
                constrain="free",
                pre_dis=0.0,
                dis=0.0,
                local_up_axis=[0,0,1]
            )
        action[1][0].target_pose[2] += 0.03
        self.move(action)

        # Record information about the objects and arm used in the task
        # self.info["info"] = {
        #     "{A}": f"047_mouse/base{self.mouse_id}",
        #     "{B}": f"{self.color_name}",
        #     "{a}": str(arm_tag),
        # }
        # return self.info

    def check_success(self):
        pose = self.target_obj.get_pose()
        end_pose_actual = pose.p
        end_pose_desired = self.des_obj.get_pose().p
        eps1 = 0.04
        eps2 = 0.04

        # UPRIGHT requirement (2026-08-12): a collapsed/tipped milktea within
        # 4cm of the marker used to count as success. Require the local axis
        # that pointed world-up at spawn (upright by construction) to stay
        # within eps_tilt_deg of world-up. Yaw spins pass; tipped/fallen fails.
        # Retro-audit on the HF training rollouts: flips 11/300 successes, 9 of
        # them fully fallen (>84 deg); survivors median tilt 3.5 deg.
        eps_tilt_deg = 15.0  # NOTE: tunable strictness, same convention as d7427da
        from transforms3d.quaternions import quat2mat
        R0 = quat2mat(self._target_init_quat)
        Rn = quat2mat(np.array(pose.q, dtype=np.float64))
        up_local = R0.T @ np.array([0.0, 0.0, 1.0])   # local axis that was world-up
        up_now = Rn @ up_local
        upright = up_now[2] > np.cos(np.deg2rad(eps_tilt_deg))

        return (np.all(abs(end_pose_actual[:2] - end_pose_desired[:2]) < np.array([eps1, eps2]))
                and upright
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
