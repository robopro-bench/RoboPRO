from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien, math, os, glob
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy


class chain_bowl_rack_apple_sink_ks(KitchenS_base_task):
    """
    Dual bowl:
      - Bowl 1 starts in the sink basin -> move to the dishrack.
      - Bowl 2 sits on the counter holding an apple -> move the apple into the sink.
    Right arm handles both legs (sink is always on the right, |sink_x| >= 0.10).
    """

    TOP_DOWN_Q = [-0.5, 0.5, -0.5, -0.5]
    TCP_OFFSET = 0.12

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.bowl1.get_name(), self.apple.get_name()}

    def load_actors(self):
        sink_p = self.sink.get_pose().p
        sg = self.kitchens_info["sink_geom"]

        # Bowl 1: centered in the sink basin. Destination = dishrack.
        bowl1_pose = sapien.Pose(
            [float(sink_p[0]), float(sink_p[1]), float(sink_p[2]) - 0.01],
            [0.5, 0.5, 0.5, 0.5],
        )
        self.bowl1_id = 3
        self.bowl1 = create_actor(
            scene=self, pose=bowl1_pose, modelname="002_bowl",
            convex=True, model_id=self.bowl1_id,
        )
        self.bowl1.set_mass(0.1)
        self.bowl1.set_name("bowl1")

        # Bowl 2: on the right counter, clear of the sink footprint so the
        # right arm can move between the two without crossing midline.
        # Large bowl variants (3/4/5) give the apple-pick gripper clearance.
        self.bowl2_id = int(np.random.choice([3, 4, 5]))
        def _sample_bowl2():
            return self.rand_pose_on_counter(
                xlim=[0.10, 0.42],
                ylim=[-0.20, -0.05],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=False,
                obj_padding=0.10,
            )
        bowl2_pose = _sample_bowl2()
        # Keep bowl 2 clear of the sink hole so the counter bowl isn't
        # inside the sink rectangle.
        while (abs(bowl2_pose.p[0] - float(sink_p[0])) < sg["hole_hx"] + 0.10
               and abs(bowl2_pose.p[1] - float(sink_p[1])) < sg["hole_hy"] + 0.10):
            bowl2_pose = _sample_bowl2()
        self.bowl2 = create_actor(
            scene=self, pose=bowl2_pose, modelname="002_bowl",
            convex=True, model_id=self.bowl2_id,
            is_static=True,
        )
        self.bowl2.set_name("bowl2")
        self.add_prohibit_area(self.bowl2, padding=0.02, area="table")

        # Apple sits just above bowl 2's rim — gravity settles it in.
        bp2 = self.bowl2.get_pose().p
        ax = float(bp2[0]) + np.random.uniform(-0.03, 0.03)
        ay = float(bp2[1]) + np.random.uniform(-0.03, 0.03)
        az = float(bp2[2]) + 0.05
        apple_pose = sapien.Pose([ax, ay, az], [0.5, 0.5, 0.5, 0.5])
        self.apple_id = int(np.random.choice([0, 1]))
        self.apple = create_actor(
            scene=self, pose=apple_pose, modelname="035_apple",
            convex=True, model_id=self.apple_id,
        )
        self.apple.set_mass(0.05)

    # -----------------------------------------------------------------
    # Step 1: bowl 1 from sink -> dishrack (side-grasp lift, scripted drop)
    # Mirrors chain_sink_bowl_bread_knife's bowl-from-sink lift and
    # place_bowl_in_dishrack's two-stage INIT_Q drop.
    # -----------------------------------------------------------------
    def _bowl_sink_to_rack(self):
        arm = ArmTag("right")
        self.enable_table(enable=False)
        self.grasp_actor_from_table(self.bowl1, arm_tag=arm, pre_grasp_dis=0.10)
        if not self.plan_success:
            return
        self.move(self.move_by_displacement(arm_tag=arm, y=-0.30, z=0.30))
        self.attach_object(
            self.bowl1,
            f"{os.environ['ASSETS_ROOT']}/objects/002_bowl/collision/base{self.bowl1_id}.glb",
            str(arm),
        )
        self.enable_table(enable=True)
        

        # Pop rack from curobo so bowl-rack overlap checks don't block planning.
        self.collision_list = [
            e for e in self.collision_list if e.get("actor") is not self.dishrack
        ]
        self.update_world()

        INIT_Q = [0.707, 0, 0, 0.707]
        rack_p = self.dishrack.get_pose().p
        drop_x = float(rack_p[0])
        drop_y = float(rack_p[1]) - 0.35

        hover_drop_pose = [drop_x, drop_y - 0.10, 1.20] + INIT_Q
        self.move(self.move_to_pose(arm, hover_drop_pose))
        drop_pose = [drop_x, drop_y, 1.00] + INIT_Q
        self.move(self.move_to_pose(arm, drop_pose))
        self.move(self.open_gripper(arm, pos=1.0))
        # Clear phantom bowl from planner before same-arm apple step.
        self.detach_object(str(arm))
        self.move(self.back_to_origin(arm))

    # -----------------------------------------------------------------
    # Step 2: apple from bowl 2 -> sink (scripted top-down, mirrors
    # pick_apple_from_bowl_ks, drops at sink instead of counter).
    # -----------------------------------------------------------------
    def _apple_bowl_to_sink(self):
        arm = ArmTag("right")
        self.enable_table(enable=False)
        self.move(self.open_gripper(arm, pos=1.0))

        apple_p = self.apple.get_pose().p
        bp2 = self.bowl2.get_pose().p

        hover_tcp_z = float(bp2[2]) + 0.12
        hover_pose = [
            float(apple_p[0]), float(apple_p[1]),
            hover_tcp_z + self.TCP_OFFSET,
        ] + self.TOP_DOWN_Q
        self.move(self.move_to_pose(arm, hover_pose))

        grasp_tcp_z = float(apple_p[2]) + 0.005
        grasp_pose = [
            float(apple_p[0]), float(apple_p[1]),
            grasp_tcp_z + self.TCP_OFFSET,
        ] + self.TOP_DOWN_Q
        self.move(self.move_to_pose(arm, grasp_pose))

        self.move(self.close_gripper(arm, pos=0.0))
        self.move(self.move_by_displacement(arm_tag=arm, z=0.10))

        self.attach_object(
            self.apple,
            f"{os.environ['ASSETS_ROOT']}/objects/035_apple/collision/base{self.apple_id}.glb",
            str(arm),
        )
        self.enable_table(enable=True)

        INIT_Q = [0.707, 0, 0, 0.707]
        sink_p = self.sink.get_pose().p
        sg = self.kitchens_info["sink_geom"]
        # Mirror put_bowl_in_sink_ks: drop toward the robot-side half of
        # the basin so the wrist stays close to the arm base.
        drop_x = float(sink_p[0]) + 0.02
        drop_y = float(sink_p[1]) - 1.15 * sg["hole_hy"]
        sink_z = float(sink_p[2])
        hover_drop = [drop_x, drop_y, sink_z + 0.25] + INIT_Q
        self.move(self.move_to_pose(arm, hover_drop))
        descend = [drop_x, drop_y, sink_z + 0.10] + INIT_Q
        self.move(self.move_to_pose(arm, descend))
        self.move(self.open_gripper(arm, pos=1.0))
        self.detach_object(str(arm))
        self.move(self.back_to_origin(arm))

    def play_once(self):
        self._bowl_sink_to_rack()
        self.plan_success = True
        self._apple_bowl_to_sink()

    def check_success(self):
        sink_p = self.sink.get_pose().p
        sg = self.kitchens_info["sink_geom"]
        rack_p = self.dishrack_center  # true rack center, not the offset entity pose
        b1 = self.bowl1.get_pose().p
        ap = self.apple.get_pose().p

        bowl_on_rack = (abs(b1[0] - rack_p[0]) < 0.12
                        and abs(b1[1] - rack_p[1]) < 0.10          # tightened toward true rack extent
                        and b1[2] > rack_p[2] - 0.08)              # elevated onto rack, not counter beside it
        apple_in_sink = (abs(ap[0] - sink_p[0]) < sg["hole_hx"]
                         and abs(ap[1] - sink_p[1]) < sg["hole_hy"]
                         and ap[2] < sink_p[2] + 0.02)
        return bowl_on_rack and apple_in_sink
