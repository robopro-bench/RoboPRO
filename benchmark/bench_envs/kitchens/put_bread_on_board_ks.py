from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from bench_envs.utils.scene_gen_utils import get_actor_boundingbox
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class put_bread_on_board_ks(KitchenS_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        rand_pos = self.rand_pose_on_counter(
            xlim=[-0.32, 0.32],
            ylim=[-0.15, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, np.pi / 2, 0],
            obj_padding=0.05,
        )
        while abs(rand_pos.p[0]) < 0.3:
            rand_pos = self.rand_pose_on_counter(
                xlim=[-0.4, 0.4],
                ylim=[-0.15, 0.05],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 2, 0],
                obj_padding=0.05,
            )

        # Use compact bread variants only — base3 is an elongated baguette
        # (~9.6cm long, only side-grip contact points) that frequently has
        # no feasible pre-grasp pose. base1, base2, base5 are roughly cubic
        # (~5–6cm) and grasp reliably from any side.
        self.bread_id = int(np.random.choice([1, 2, 5]))
        self.target_obj = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="075_bread",
            convex=True,
            model_id=self.bread_id,
        )
        self.target_obj.set_mass(0.05)

        # Static board may sit over the middle sink; keep the original center
        # band so either arm can place onto it.
        target_rand_pose = self.rand_pose_on_counter(
            xlim=[-0.12, 0.12],
            ylim=[-0.23, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, np.pi / 4, 0],
            obj_padding=0.10,
            allow_sink=True,
        )

        # Real cutting-board asset (104_board). Mesh y is thickness (~0.17–0.19 in
        # mesh units); qpos [0.5,0.5,0.5,0.5] rotates mesh-y → world-z so the board
        # lies flat. Scale 0.10 yields ~13–19 cm boards, 1.9 cm thick.
        self.board_id = int(np.random.choice([0, 1, 2, 3]))
        self.des_obj = create_actor(
            scene=self,
            pose=target_rand_pose,
            modelname="104_board",
            convex=True,
            model_id=self.board_id,
            scale=[0.10, 0.10, 0.10],
            is_static=True,
        )
        self.des_obj.set_name("board")
        self.add_prohibit_area(self.des_obj, padding=0.01, area="table")
        self.add_prohibit_area(self.target_obj, padding=0.02, area="table")
        self.des_obj_pose = self.des_obj.get_pose().p.tolist() + [0, 0, 0, 1]
        self.des_obj_pose[2] += 0.02


    def play_once(self):
        arm_tag = ArmTag("right" if self.target_obj.get_pose().p[0] > 0 else "left")

        self.grasp_actor_from_table(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.07)
        if not self.plan_success:
            return

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1))

        self.attach_object(
            self.target_obj,
            f"{os.environ['ASSETS_ROOT']}/objects/075_bread/collision/base{self.bread_id}.glb",
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
        # Success = bread is on the board (footprint + resting height), not near a
        # target center. Board is ~13–19 cm; a 3 cm XY gate around the origin
        # rejected valid off-center placements.
        bread_p = np.asarray(self.target_obj.get_pose().p)
        board_bb = get_actor_boundingbox(self.des_obj)
        if board_bb[0] is None:
            return False

        bread_on_board_xy = (board_bb[0][0] <= bread_p[0] <= board_bb[1][0] and
                             board_bb[0][1] <= bread_p[1] <= board_bb[1][1])
        dz = float(bread_p[2] - board_bb[1][2])
        # Bread is ~5–6 cm; its center sits a few cm above the board top.
        bread_on_board_z = 0 <= dz <= 0.06

        return bool(bread_on_board_xy and bread_on_board_z
                    and self.robot.is_left_gripper_open()
                    and self.robot.is_right_gripper_open())
