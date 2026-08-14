from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class put_spoon_on_plate_ks(KitchenS_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        rand_pos = self.rand_pose_on_counter(
            xlim=[-0.45, 0.45],
            ylim=[-0.23, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, 3.14, 0],
            obj_padding=0.04,
        )
        while abs(rand_pos.p[0]) < 0.3:
            rand_pos = self.rand_pose_on_counter(
                xlim=[-0.45, 0.45],
                ylim=[-0.23, 0.05],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
                obj_padding=0.04,
            )

        self.spoon_id = 0
        self.target_obj = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="134_spoon",
            convex=True,
            model_id=self.spoon_id,
        )
        self.target_obj.set_mass(0.05)
        self.add_prohibit_area(self.target_obj, padding=0.02, area="table")

        # Center-ish plate (original was xlim=[0]) with enough x jitter that
        # scene 1 can still land left of the sink. Clip so the 8 cm footprint
        # clears the sink keepout; without that, most of [-0.2, 0.2] is invalid
        # when the sink is at x≈0.10.
        obj_padding = 0.08
        side_xlim = [-0.2, 0.2]
        sink_x = float(self.sink.get_pose().p[0])
        sink_geom = self.kitchens_info["sink_geom"]
        sink_pad = 0.03  # must match _load_sink
        clearance = sink_geom["hole_hx"] + sink_pad + obj_padding + 0.01
        sink_clear = [
            [side_xlim[0], min(side_xlim[1], sink_x - clearance)],
            [max(side_xlim[0], sink_x + clearance), side_xlim[1]],
        ]
        plate_xlims = [b for b in sink_clear if b[1] - b[0] >= 0.03]
        if not plate_xlims:
            raise RuntimeError(
                f"No sink-clear plate spawn in {side_xlim}: "
                f"sink_x={sink_x:.3f}"
            )
        plate_xlim = max(plate_xlims, key=lambda b: b[1] - b[0])

        target_rand_pose = self.rand_pose_on_counter(
            xlim=plate_xlim,
            ylim=[-0.23, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
            obj_padding=obj_padding,
        )

        self.plate_id = 0
        self.des_obj = create_actor(
            scene=self,
            pose=target_rand_pose,
            modelname="003_plate",
            convex=True,
            model_id=self.plate_id,
            is_static=True,
        )
        self.add_prohibit_area(self.des_obj, padding=0.01, area="table")

        self.des_obj_pose = self.des_obj.get_pose().p.tolist() + [0, 0, 0, 1]
        self.des_obj_pose[2] += 0.02

    def play_once(self):
        arm_tag = ArmTag("right" if self.target_obj.get_pose().p[0] > 0 else "left")

        self.grasp_actor_from_table(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.1)

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1))

        self.attach_object(
            self.target_obj,
            f"{os.environ['ASSETS_ROOT']}/objects/134_spoon/collision/base{self.spoon_id}.glb",
            str(arm_tag),
        )
        self.enable_table(enable=True)

        self.move(
            self.place_actor(
                self.target_obj,
                arm_tag=arm_tag,
                target_pose=self.des_obj_pose,
                constrain="align",
                pre_dis=0.05,
                dis=0.005,
            ))

    def check_success(self):
        end_pose_actual = self.target_obj.get_pose().p
        end_pose_desired = self.des_obj.get_pose().p
        eps1 = 0.03
        eps2 = 0.03

        return (np.all(abs(end_pose_actual[:2] - end_pose_desired[:2]) < np.array([eps1, eps2]))
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
