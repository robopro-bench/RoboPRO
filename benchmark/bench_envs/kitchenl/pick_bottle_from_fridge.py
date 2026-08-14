from bench_envs.kitchenl._kitchen_base_large import Kitchen_base_large
from bench_envs.utils.scene_gen_utils import get_random_place_pose, get_actor_boundingbox, print_c
from envs.utils import *
import math
import numpy as np
import sapien
import transforms3d as t3d
import os
import yaml

class pick_bottle_from_fridge(Kitchen_base_large):
    BOTTLE_MASS = 0.1
    FRIDGE_WORLD_XY_JITTER = 0.025

    # Bottle spawn anchor in fridge base-link local coordinates.
    # Note: local axes map to world via world = base_p + base_R @ local.
    FRIDGE_BOTTLE_LOCAL = np.array([0.125, -0.17, 0.00], dtype=float)

    # Fridge interior bounds in fridge base-link local frame
    FRIDGE_X_BOUNDS = (-0.33, 0.16)
    FRIDGE_Y_BOUNDS = (-0.22, 0.22)
    FRIDGE_Z_BOUNDS = (-0.34, 0.24)
    IN_HAND_TCP_DIST_THRESHOLD = 0.18
    # Retrieval trajectory tuning
    APPROACH_DELTA = dict(x=0.1, y=0.10, z=-0.05)
    LIFT_DELTA = dict(z=0.05)
    RETREAT_DELTA = dict(y=-0.1)
    GRASP_PRE_DIS = 0.07
    GRASP_DIS = 0.01
    GRASP_CONTACT_POINT_ID = 1
    GRASP_CLOSE_POS = 0.0

    def _close_microwave_if_present(self) -> None:
        if getattr(self, "microwave_left", None) is None:
            return
        try:
            mw_qpos = np.array(self.microwave_left.get_qpos(), dtype=float)
            if mw_qpos.size > 0:
                mw_qpos[:] = 0.0
                self.microwave_left.set_qpos(mw_qpos)
        except Exception:
            pass

    def _get_target_object_names(self) -> set[str]:
        return {self.bottle.get_name()}

    def setup_demo(self, is_test: bool = False, **kwargs):
        self.bottle_modelname = "001_bottle"
        self.bottle_spawn_rot_deg = [0.0, 0.0, 90.0]
        rot_cfg = kwargs.pop("bottle_spawn_rot_deg", None)
        if rot_cfg is not None:
            self.bottle_spawn_rot_deg = [float(rot_cfg[0]), float(rot_cfg[1]), float(rot_cfg[2])]

        self.bottle_scale = 0.7
        bs = kwargs.pop("bottle_scale", None)
        if bs is not None:
            self.bottle_scale = float(bs)

        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)
        self._close_microwave_if_present()

        self.set_fridge_open()

    def check_stable(self):
        # Dynamic bottle can wobble while settling; ignore it during global stability screening.
        is_stable, unstable_list = super().check_stable()
        unstable_list = [n for n in unstable_list if n != "task_bottle"]
        return len(unstable_list) == 0, unstable_list

    def _bottle_quat_from_cfg(self) -> list[float]:
        roll_deg, pitch_deg, yaw_deg = self.bottle_spawn_rot_deg
        ax = math.radians(roll_deg)
        ay = math.radians(pitch_deg)
        az = math.radians(yaw_deg)
        qx, qy, qz, qw = t3d.euler.euler2quat(ax, ay, az)
        return [qw, qx, qy, qz]

    def _fridge_inside_spawn_pose(self) -> sapien.Pose:
        base_pose = self.fridge_left.get_link_pose("base_link")
        base_tf = base_pose.to_transformation_matrix()
        base_R = np.array(base_tf[:3, :3], dtype=float)
        base_p = np.array(base_tf[:3, 3], dtype=float)
        world_inside = base_p + base_R @ self.FRIDGE_BOTTLE_LOCAL
        # Randomize in WORLD frame: +/- jitter on world x/y only, keep world z unchanged.
        world_inside[0] += float(np.random.uniform(-self.FRIDGE_WORLD_XY_JITTER, self.FRIDGE_WORLD_XY_JITTER))
        world_inside[1] += float(np.random.uniform(-self.FRIDGE_WORLD_XY_JITTER, self.FRIDGE_WORLD_XY_JITTER))
        return sapien.Pose(world_inside.tolist(), self._bottle_quat_from_cfg())

    def _bottle_local_in_fridge(self) -> np.ndarray | None:
        if self.bottle is None or self.fridge_left is None:
            return None
        bottle_world = np.array(self.bottle.get_pose().p, dtype=float)
        base_pose = self.fridge_left.get_link_pose("base_link")
        inv_tf = np.linalg.inv(base_pose.to_transformation_matrix())
        bottle_local_h = inv_tf @ np.array([bottle_world[0], bottle_world[1], bottle_world[2], 1.0], dtype=float)
        return np.array(bottle_local_h[:3], dtype=float)

    def load_actors(self):

        if getattr(self, "fridge_closed_qpos", None) is None:
            self._init_fridge_states()
        self.set_fridge_open()
  
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r", encoding="utf-8") as f:
            task_objs = yaml.safe_load(f)

        self.bottle_model_id = np.random.choice(self._target_ids("kitchenl", self.bottle_modelname))

        spawn_pose = self._fridge_inside_spawn_pose()

        intrinsic_scale = self._get_asset_model_scale_create_actor(
            self.bottle_modelname, self.bottle_model_id
        )

        final_scale = float(intrinsic_scale) * float(self.bottle_scale)

        self.bottle = create_actor(
            scene=self.scene,
            pose=spawn_pose,
            modelname=self.bottle_modelname,
            model_id=self.bottle_model_id,
            is_static=False,
            convex=True,
            scale=final_scale,
        )
        if self.bottle is not None:
            self.bottle.set_mass(self.BOTTLE_MASS)
            self.bottle.set_name("task_bottle")
            if isinstance(self.bottle.config, dict):
                self.bottle.config["scale"] = [final_scale] * 3
            self.add_prohibit_area(self.bottle, padding=0.04, area="table")
        
       
        if self.scene_id == 1:
            ylim = [-0.15, 0.05]
        else:
            ylim = [-0.2, 0.1]
        self.des_pose = get_random_place_pose(xlim = [-0.1, 0.2], ylim=ylim,
                                        col_thr=0.15,zlim=[0.78],
                                        object_bounds={})
        self.add_prohibit_area(self.des_pose, padding=0.04, area="table")

    def _is_bottle_inside_fridge(self) -> bool:
        bottle_local = self._bottle_local_in_fridge()
        if bottle_local is None:
            return False
        x_l, y_l, z_l = bottle_local
        x_ok = (self.FRIDGE_X_BOUNDS[0] <= x_l <= self.FRIDGE_X_BOUNDS[1])
        y_ok = (self.FRIDGE_Y_BOUNDS[0] <= y_l <= self.FRIDGE_Y_BOUNDS[1])
        z_ok = (self.FRIDGE_Z_BOUNDS[0] <= z_l <= self.FRIDGE_Z_BOUNDS[1])
        return bool(x_ok and y_ok and z_ok)

    def play_once(self):
        arm_tag = ArmTag("right")
        # Approach fridge opening before grasp.
        self.move(self.move_by_displacement(arm_tag=arm_tag, **self.APPROACH_DELTA))
        self.move(
            self.grasp_actor(
                self.bottle,
                arm_tag=arm_tag,
                pre_grasp_dis=self.GRASP_PRE_DIS,
                grasp_dis=self.GRASP_DIS,
                gripper_pos=self.GRASP_CLOSE_POS,
                contact_point_id=self.GRASP_CONTACT_POINT_ID,
            )
        )
        self.attach_object(self.bottle, f"{os.environ['ASSETS_ROOT']}/objects/{self.bottle_modelname}/collision/base{self.bottle_model_id}.glb", str(arm_tag))

        self.move(self.move_by_displacement(arm_tag=arm_tag, **self.LIFT_DELTA))
        self.move(self.move_by_displacement(arm_tag=arm_tag, **self.RETREAT_DELTA))
        self.move(self.back_to_origin(arm_tag=arm_tag))
        self.add_collision()
        self.update_world()
        self.move(
            self.place_actor(
                self.bottle,
                arm_tag=arm_tag,
                target_pose= self.des_pose,
                constrain="free",
                pre_dis=0.07,
                dis=0.005,
            ))
        
        self.info["info"] = {
            "{A}": f"{self.bottle_modelname}/base{self.bottle_model_id}",
            "{a}": str(arm_tag),
        }
        return self.info
    def check_success(self):
        eps = 0.01
        b_pose = self.bottle.get_pose().p
        table_bb = get_actor_boundingbox(self.table)
        bottle_on_table = np.all((table_bb[0][:2] <= b_pose[:2])  &  (b_pose[:2] <= table_bb[1][:2]))
        bottle_on_table &= (b_pose[-1] - table_bb[1][-1]) < eps  
    
        return not self._is_bottle_inside_fridge() and bottle_on_table \
               and self.robot.is_right_gripper_open() \
               and self.robot.is_left_gripper_open()

