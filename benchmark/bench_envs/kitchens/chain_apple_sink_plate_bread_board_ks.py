from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from bench_envs.utils.scene_gen_utils import get_actor_boundingbox
from envs.utils import *
import sapien, math, os, glob
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy


class chain_apple_sink_plate_bread_board_ks(KitchenS_base_task):
    """
    Chain: apple (from sink) → plate, bread → board.
    """

    TOP_DOWN_Q = [-0.5, 0.5, -0.5, -0.5]
    TCP_OFFSET = 0.12

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.apple.get_name(), self.bread.get_name()}

    def load_actors(self):
        sink_p = self.sink.get_pose().p
        sg = self.kitchens_info["sink_geom"]

        # Apple inside sink basin — centered in x, front 1/3 of y (same
        # reachable band as pick_apple_from_sink_ks).
        self.apple_id = int(np.random.choice([0, 1]))
        ax = float(sink_p[0]) + np.random.uniform(-sg["inner_hx"] * 0.25, sg["inner_hx"] * 0.25)
        ay = float(sink_p[1]) + np.random.uniform(-sg["inner_hy"] * 0.9, -sg["inner_hy"] / 3)
        az = float(sink_p[2]) - 0.01
        apple_pose = sapien.Pose([ax, ay, az], [0.5, 0.5, 0.5, 0.5])
        self.apple = create_actor(
            scene=self, pose=apple_pose, modelname="035_apple",
            convex=True, model_id=self.apple_id,
        )
        self.apple.set_mass(0.05)

        # Apple always spawns on right of sink (sink x >= 0.10 in all scenes),
        # so the right arm does the apple→plate leg.
        self.apple_arm = ArmTag("right")

        # Plate on the front-right counter, not in the basin (the apple is
        # picked from the sink). allow_sink only ignores keep-out padding.
        plate_pose = self.rand_pose_on_counter(
            xlim=[0.16, 0.30], ylim=[-0.22, -0.12],
            qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=False,
            obj_padding=0.08,
            allow_sink=True,
        )
        self.plate = create_actor(
            scene=self, pose=plate_pose, modelname="003_plate",
            convex=True, model_id=0, is_static=True,
        )
        self.plate.set_mass(0.1)
        self.add_prohibit_area(self.plate, padding=0.02, area="table")

        # Front-center band (static; may overlap sink keep-out padding).
        # Keep it in front of the basin so it does not block the apple pick.
        self.board_id = int(np.random.choice([0, 1, 2, 3]))
        board_pose = self.rand_pose_on_counter(
            xlim=[-0.12, 0.12], ylim=[-0.23, -0.14],
            qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=True, rotate_lim=[0, np.pi/4, 0],
            obj_padding=0.10,
            allow_sink=True,
        )
        self.board = create_actor(
            scene=self, pose=board_pose, modelname="104_board",
            convex=True, model_id=self.board_id,
            scale=[0.10, 0.10, 0.10], is_static=True,
        )
        self.board.set_name("board")
        self.add_prohibit_area(self.board, padding=0.02, area="table")

        # Bread on either side (|x| >= 0.3), same reachability filter as
        # put_bread_on_board_ks. Left-only [-0.40, -0.25] sat against the
        # microwave and the grasp descent failed.
        bread_pose = self.rand_pose_on_counter(
            xlim=[-0.32, 0.32], ylim=[-0.23, -0.08],
            qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=True, rotate_lim=[0, np.pi/2, 0],
            obj_padding=0.05,
        )
        while abs(bread_pose.p[0]) < 0.3:
            bread_pose = self.rand_pose_on_counter(
                xlim=[-0.4, 0.4], ylim=[-0.23, -0.08],
                qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=True, rotate_lim=[0, np.pi/2, 0],
                obj_padding=0.05,
            )
        self.bread_id = int(np.random.choice([1, 2, 5]))
        self.bread = create_actor(
            scene=self, pose=bread_pose, modelname="075_bread",
            convex=True, model_id=self.bread_id,
        )
        self.bread.set_mass(0.05)
        self.add_prohibit_area(self.bread, padding=0.02, area="table")

    # -----------------------------------------------------------------
    # Step 1: apple from sink → plate (scripted top-down, mirrors
    # pick_apple_from_sink_ks, drop on plate instead of random counter pose).
    # -----------------------------------------------------------------
    def _apple_from_sink_to_plate(self):
        arm_tag = self.apple_arm
        self.enable_table(enable=False)
        # Board/plate are in the Curobo world and block the top-down dip
        # into the basin if they sit near the sink.
        self.collision_list = [
            e for e in self.collision_list
            if e.get("actor") not in (self.plate, self.board)
        ]
        self.update_world()
        self.enable_table(enable=False)
        self.move(self.open_gripper(arm_tag, pos=1.0))

        apple_p = self.apple.get_pose().p
        sink_p = self.sink.get_pose().p

        hover_tcp_z = float(sink_p[2]) + 0.08
        hover_pose = [
            float(apple_p[0]), float(apple_p[1]),
            hover_tcp_z + self.TCP_OFFSET,
        ] + self.TOP_DOWN_Q
        self.move(self.move_to_pose(arm_tag, hover_pose))

        grasp_tcp_z = float(apple_p[2]) + 0.005
        grasp_pose = [
            float(apple_p[0]), float(apple_p[1]),
            grasp_tcp_z + self.TCP_OFFSET,
        ] + self.TOP_DOWN_Q
        self.move(self.move_to_pose(arm_tag, grasp_pose))

        self.move(self.close_gripper(arm_tag, pos=0.0))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))

        self.attach_object(
            self.apple,
            f"{os.environ['ASSETS_ROOT']}/objects/035_apple/collision/base{self.apple_id}.glb",
            str(arm_tag),
        )
        self.enable_table(enable=True)

        # Pop the (static) plate from curobo so the scripted top-down descent
        # doesn't get blocked by the plate's own collision body.
        self.collision_list = [
            e for e in self.collision_list if e.get("actor") is not self.plate
        ]
        self.update_world()

        plate_p = self.plate.get_pose().p
        tgt_x, tgt_y, tgt_z = float(plate_p[0]), float(plate_p[1]), float(plate_p[2])
        hover_drop = [tgt_x, tgt_y, tgt_z + 0.15 + self.TCP_OFFSET] + self.TOP_DOWN_Q
        self.move(self.move_to_pose(arm_tag, hover_drop))
        self.move(self.open_gripper(arm_tag, pos=1.0))
        self.detach_object(str(arm_tag))
        self.move(self.back_to_origin(arm_tag))

    # -----------------------------------------------------------------
    # Step 2: bread → board
    # -----------------------------------------------------------------
    def _bread_to_board(self):
        arm = ArmTag("right" if float(self.bread.get_pose().p[0]) > 0 else "left")
        self.grasp_actor_from_table(self.bread, arm_tag=arm, pre_grasp_dis=0.07)
        if not self.plan_success:
            return
        self.move(self.move_by_displacement(arm_tag=arm, z=0.1))
        self.attach_object(
            self.bread,
            f"{os.environ['ASSETS_ROOT']}/objects/075_bread/collision/base{self.bread_id}.glb",
            str(arm),
        )
        self.enable_table(enable=True)
        board_p = self.board.get_pose().p
        board_target = [float(board_p[0]), float(board_p[1]),
                        float(board_p[2]) + 0.02, 0, 0, 0, 1]
        self.move(self.place_actor(
            self.bread, arm_tag=arm, target_pose=board_target,
            constrain="align", pre_dis=0.05, dis=0.005,
        ))
        self.move(self.back_to_origin(arm))

    def play_once(self):
        self._apple_from_sink_to_plate()
        self.plan_success = True
        self._bread_to_board()

    def check_success(self):
        apple_p = np.asarray(self.apple.get_pose().p)
        plate_p = np.asarray(self.plate.get_pose().p)
        bread_p = np.asarray(self.bread.get_pose().p)

        apple_on_plate = (abs(apple_p[0] - plate_p[0]) < 0.08
                          and abs(apple_p[1] - plate_p[1]) < 0.08
                          and apple_p[2] > plate_p[2] - 0.02)

        board_bb = get_actor_boundingbox(self.board)
        if board_bb[0] is None:
            return False
        bread_on_board_xy = (board_bb[0][0] <= bread_p[0] <= board_bb[1][0] and
                             board_bb[0][1] <= bread_p[1] <= board_bb[1][1])
        dz = float(bread_p[2] - board_bb[1][2])
        bread_on_board = bread_on_board_xy and 0 <= dz <= 0.06

        return bool(apple_on_plate and bread_on_board)
