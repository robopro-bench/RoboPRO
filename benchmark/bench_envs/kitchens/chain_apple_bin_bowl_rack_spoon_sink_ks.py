from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien, math, os, glob
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
from transforms3d.euler import euler2quat


class chain_apple_bin_bowl_rack_spoon_sink_ks(KitchenS_base_task):
    """
    Chain: apple → bin, bowl → dishrack, spoon → sink.
    """

    # Scene 1 puts the sink at x=0.10. Its keep-out — x [-0.06, 0.26],
    # y [-0.15, 0.32] — covers the original centre bowl window
    # xlim=[-0.10, 0.10], ylim=[-0.20, -0.08] completely (same trap as
    # drop_apple_in_bin_ks / put_bowl_in_sink_ks). Clearing it in x needs
    # x < -0.12 or x > 0.32; clearing it in y needs y < -0.21. The rack is
    # on the right, so the bowl stays on the right arm, in the front strip
    # that actually clears the basin. Scenes 0 and 2 keep the original
    # centre band — sink is on the right there, and those seeds already
    # setup.
    SCENE1_BOWL_XLIM = [0.08, 0.18]
    SCENE1_BOWL_YLIM = [-0.30, -0.22]

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.apple.get_name(), self.bowl.get_name(), self.spoon.get_name()}

    def load_actors(self):
        rack_p = self.dishrack.get_pose().p

        # Bin: always on the left side of the counter.
        bin_pose = self.rand_pose_on_counter(
            xlim=[-0.42, -0.25], ylim=[-0.23, 0.0],
            qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=True, rotate_lim=[0, np.pi/4, 0],
            obj_padding=0.12,
        )
        self.bin_id = int(np.random.choice([0, 6]))
        self.bin = create_actor(
            scene=self, pose=bin_pose, modelname="063_tabletrashbin",
            convex=True, model_id=self.bin_id,
            scale=[0.10, 0.10, 0.10], is_static=True,
        )
        self.bin.set_name("bin")
        self.add_prohibit_area(self.bin, padding=0.02, area="table")
        bin_p = self.bin.get_pose().p

        # Apple: always mid-left (with variation).
        apple_pose = self.rand_pose_on_counter(
            xlim=[-0.22, -0.08], ylim=[-0.15, 0.00],
            qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=True, rotate_lim=[0, np.pi, 0],
            obj_padding=0.05,
        )
        self.apple_id = int(np.random.choice([0, 1]))
        self.apple = create_actor(
            scene=self, pose=apple_pose, modelname="035_apple",
            convex=True, model_id=self.apple_id,
        )
        self.apple.set_mass(0.05)
        self.add_prohibit_area(self.apple, padding=0.02, area="table")

        # Bowl: centre band on scenes 0/2; scene 1 uses the right-front
        # strip so the footprint clears the middle sink.
        scene1 = getattr(self, "scene_id", None) == 1
        self.bowl_arm = ArmTag("right" if float(rack_p[0]) > 0 else "left")
        bowl_pose = self.rand_pose_on_counter(
            xlim=self.SCENE1_BOWL_XLIM if scene1 else [-0.10, 0.10],
            ylim=self.SCENE1_BOWL_YLIM if scene1 else [-0.20, -0.08],
            qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=False,
            obj_padding=0.06,
        )
        self.bowl_id = 3
        self.bowl = create_actor(
            scene=self, pose=bowl_pose, modelname="002_bowl",
            convex=True, model_id=self.bowl_id,
        )
        self.bowl.set_mass(0.05)
        self.add_prohibit_area(self.bowl, padding=0.02, area="table")

        # Spoon: always mid-right (same xlim as put_spoon_in_sink_ks).
        self.spoon_arm = ArmTag("right")
        spoon_pose = self.rand_pose_on_counter(
            xlim=[0.30, 0.45], ylim=[-0.23, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=True, rotate_lim=[0, 3.14, 0],
            obj_padding=0.04,
        )
        self.spoon_id = 0
        self.spoon = create_actor(
            scene=self, pose=spoon_pose, modelname="134_spoon",
            convex=True, model_id=self.spoon_id,
        )
        self.spoon.set_mass(0.05)
        self.add_prohibit_area(self.spoon, padding=0.02, area="table")

        # Fixed drop targets
        self.bin_target_pose = [float(bin_p[0]), float(bin_p[1]), float(bin_p[2]) + 0.08, 0, 0, 0, 1]

    # -----------------------------------------------------------------
    # Step 1: apple → bin (grasp from counter, drop above bin)
    # -----------------------------------------------------------------
    def _apple_to_bin(self):
        arm = ArmTag("left")
        self.grasp_actor_from_table(self.apple, arm_tag=arm, pre_grasp_dis=0.07)
        if not self.plan_success:
            return
        self.move(self.move_by_displacement(arm_tag=arm, z=0.10))
        self.attach_object(
            self.apple,
            f"{os.environ['BENCH_ROOT']}/assets/objects/035_apple/collision/base{self.apple_id}.glb",
            str(arm),
        )
        self.enable_table(enable=True)
        self.move(self.place_actor(
            self.apple, arm_tag=arm, target_pose=self.bin_target_pose,
            constrain="align", pre_dis=0.08, dis=0.01,
        ))
        self.move(self.back_to_origin(arm))

    # -----------------------------------------------------------------
    # Step 2: bowl → dishrack (mirrors place_bowl_in_dishrack_ks.play_once)
    # -----------------------------------------------------------------
    def _bowl_to_dishrack(self):
        arm_tag = self.bowl_arm
        self.grasp_actor_from_table(self.bowl, arm_tag=arm_tag, pre_grasp_dis=0.10)
        if not self.plan_success:
            return
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))
        self.attach_object(
            self.bowl,
            f"{os.environ['BENCH_ROOT']}/assets/objects/002_bowl/collision/base{self.bowl_id}.glb",
            str(arm_tag),
        )
        # Pop rack from curobo so bowl-rack overlap checks don't block planning.
        self.collision_list = [
            e for e in self.collision_list if e.get("actor") is not self.dishrack
        ]
        self.update_world()

        INIT_Q = [0.707, 0, 0, 0.707]
        rack_p = self.dishrack.get_pose().p
        drop_x = float(rack_p[0])
        drop_y = float(rack_p[1]) - 0.35

        hover_drop_pose = [drop_x, drop_y - 0.20, 1.20] + INIT_Q
        self.move(self.move_to_pose(arm_tag, hover_drop_pose))
        drop_pose = [drop_x, drop_y, 1.00] + INIT_Q
        self.move(self.move_to_pose(arm_tag, drop_pose))
        self.move(self.open_gripper(arm_tag, pos=1.0))
        # Detach bowl from planner — open_gripper drops it physically but the
        # planner still thinks it's on the wrist, which blocks later planning
        # when spoon_to_sink uses the same arm.
        self.detach_object(str(arm_tag))
        self.move(self.back_to_origin(arm_tag))

    # -----------------------------------------------------------------
    # Step 3: spoon → sink (mirrors put_spoon_in_sink_ks.play_once)
    # -----------------------------------------------------------------
    def _spoon_to_sink(self):
        arm_tag = self.spoon_arm
        self.grasp_actor_from_table(
            self.spoon, arm_tag=arm_tag, pre_grasp_dis=0.1,
            contact_point_id=[0, 2],
        )
        if not self.plan_success:
            return
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1))
        self.attach_object(
            self.spoon,
            f"{os.environ.get('BENCH_ROOT', os.environ['ROBOTWIN_ROOT'] + '/../benchmark')}/assets/objects/134_spoon/collision/base{self.spoon_id}.glb",
            str(arm_tag),
        )
        self.enable_table(enable=True)

        sink_p = self.sink.get_pose().p
        sink_target = [
            float(sink_p[0]) - 0.05,
            float(sink_p[1]),
            float(sink_p[2]) + 0.04,
            0, 0, 0, 1,
        ]
        self.move(self.place_actor(
            self.spoon, arm_tag=arm_tag, target_pose=sink_target,
            constrain="free", pre_dis=0.05, dis=0.005,
        ))

    def play_once(self):
        self._apple_to_bin()
        self.plan_success = True
        self._bowl_to_dishrack()
        self.plan_success = True
        self._spoon_to_sink()

    def check_success(self):
        sg = self.kitchens_info["sink_geom"]
        sink_p = self.sink.get_pose().p
        bin_p = self.bin.get_pose().p
        rack_p = self.dishrack_center  # true rack center, not the offset entity pose

        ap = self.apple.get_pose().p
        bp = self.bowl.get_pose().p
        sp = self.spoon.get_pose().p

        apple_in_bin = (abs(ap[0] - bin_p[0]) < 0.08
                        and abs(ap[1] - bin_p[1]) < 0.07
                        and ap[2] < bin_p[2] + 0.10)
        bowl_on_rack = (abs(bp[0] - rack_p[0]) < 0.12
                        and abs(bp[1] - rack_p[1]) < 0.10          # tightened toward true rack extent
                        and bp[2] > rack_p[2] - 0.08)              # elevated onto rack, not counter beside it
        spoon_in_sink = (abs(sp[0] - sink_p[0]) < sg["hole_hx"]
                         and abs(sp[1] - sink_p[1]) < sg["hole_hy"]
                         and sp[2] < sink_p[2] + 0.01)
        # print(f"[chain_apple_bin] apple_in_bin={apple_in_bin} "
        #       f"(dx={ap[0]-bin_p[0]:.3f}, dy={ap[1]-bin_p[1]:.3f}, dz_above_bin={ap[2]-bin_p[2]:.3f}) | "
        #       f"bowl_on_rack={bowl_on_rack} (dx={bp[0]-rack_p[0]:.3f}, dy={bp[1]-rack_p[1]:.3f}) | "
        #       f"spoon_in_sink={spoon_in_sink} (dx={sp[0]-sink_p[0]:.3f}, dy={sp[1]-sink_p[1]:.3f}, "
        #       f"dz_above_rim={sp[2]-sink_p[2]:.3f}, hole_hx={sg['hole_hx']:.3f}, hole_hy={sg['hole_hy']:.3f})")
        return apple_in_bin and bowl_on_rack and spoon_in_sink
