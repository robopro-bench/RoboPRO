from bench_envs.office._office_base_task import Office_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob
from transforms3d.euler import euler2quat

class put_milktea_on_shelf(Office_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name(), self.des_obj.get_name()}

    def load_actors(self):
        target_rand_pose = rand_pose(
            xlim=[self.office_info["shelf_lims"][0]+self.office_info["shelf_padding"], self.office_info["shelf_lims"][2]-self.office_info["shelf_padding"]],
            ylim=[self.office_info["shelf_lims"][1] + 0.03],
            zlim = [self.office_info["shelf_heights"][0]],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )

        half_size = [0.025, 0.025, 0.0005]
        self.des_obj = create_box(
            scene=self,
            pose=target_rand_pose,
            half_size=half_size,
            color=(0, 0, 1), # blue
            name="des_obj",
            is_static=True,
        )
        self.des_obj_pose = self.des_obj.get_pose().p.tolist()
        self.des_obj_pose[2] += 0.02 # raise des_obj 0.02 meters
        self.add_prohibit_area(self.des_obj, padding=0.05, area=f"shelf0")

        self.milktea_id = np.random.choice(self._target_ids("office", "101_milk-tea"))
        self.side = "right" if self.des_obj.get_pose().p[0] > 0 else "left"
        if self.side == "left":
            xlim1 = [self.office_info["table_lims"][0]+self.target_objects_info["101_milk-tea"]["params"][f"{self.milktea_id}"]["radius"], 0.08]
        else:
            xlim1 = [-0.08, self.office_info["table_lims"][2]-self.target_objects_info["101_milk-tea"]["params"][f"{self.milktea_id}"]["radius"]]
        ylim1 = [self.office_info["table_lims"][1] + 0.3, self.office_info["shelf_lims"][1]-0.05]

        self.target_obj = rand_create_actor(
            self,
            xlim=xlim1,
            ylim=ylim1,
            modelname="101_milk-tea",
            rotate_rand=False,
            rotate_lim=[0, 1, 0],
            qpos=euler2quat(np.pi/2,0, 0, axes='sxyz'),
            convex=True,
            model_id=self.milktea_id, 
        )
        self.target_obj.set_mass(0.06)
        # Spawn orientation is upright by construction; remember which local axis
        # points world-up so check_success can measure tilt without assuming an
        # asset axis convention.
        self._target_init_quat = np.array(self.target_obj.get_pose().q, dtype=np.float64)
        self.add_prohibit_area(self.target_obj, padding=0)
        self.add_operating_area(self.target_obj.get_pose().p)
        self.des_obj_pose += self.target_obj.get_pose().q.tolist()

    def play_once(self):
        arm_tag = ArmTag(self.side)

        action = self.grasp_actor(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.1, grasp_dis=0.02, contact_point_id=2)
        action[1][0].target_pose[2] += 0.04
        action[1][1].target_pose[2] += 0.04
        self.move(action)

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.01))

        self.attach_object(self.target_obj, f"{os.environ['ASSETS_ROOT']}/objects/101_milk-tea/collision/base{self.milktea_id}.glb", str(arm_tag))

        action = self.place_actor(
                self.target_obj,
                arm_tag=arm_tag,
                target_pose=self.des_obj_pose,
                constrain="free",
                pre_dis=0.05,
                dis=0.0,
                local_up_axis=[0,0,1]
            )
        # action[1][0].target_pose[2] += 0.03
        self.move(action)


    def check_success(self):
        pose = self.target_obj.get_pose()
        tp = pose.p
        xmin, ymin, xmax, ymax = self.office_info["shelf_lims"]
        eps = 0.02
        on_shelf_xy = (xmin <= tp[0] <= xmax) and (ymin <= tp[1] <= ymax)

        # Upright requirement, same as put_milktea_next_to_laptop: the local axis
        # that pointed world-up at spawn must stay within eps_tilt_deg of world-up.
        # Yaw spins pass; a tipped or fallen milktea fails.
        eps_tilt_deg = 15.0  # tilt tolerance (deg)
        from transforms3d.quaternions import quat2mat
        R0 = quat2mat(self._target_init_quat)
        Rn = quat2mat(np.array(pose.q, dtype=np.float64))
        up_local = R0.T @ np.array([0.0, 0.0, 1.0])   # local axis that was world-up
        up_now = Rn @ up_local
        upright = up_now[2] > np.cos(np.deg2rad(eps_tilt_deg))

        return (abs(tp[2] - self.office_info["shelf_heights"][0]) < eps
                and on_shelf_xy
                and upright
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
