import mplib.planner
import mplib
import numpy as np
import pdb
import traceback
import os
import numpy as np
import toppra as ta
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld
import transforms3d as t3d
import envs._GLOBAL_CONFIGS as CONFIGS

from pathlib import Path
from typing import Optional, Sequence
import trimesh

try:
    # ********************** CuroboPlanner (optional) **********************
    from curobo.types.math import Pose as CuroboPose
    import time
    from curobo.types.robot import JointState
    from curobo.wrap.reacher.motion_gen import (
        MotionGen,
        MotionGenConfig,
        MotionGenPlanConfig,
        PoseCostMetric,
    )
    from curobo.geom.types import WorldConfig, Mesh
    from curobo.util import logger
    import torch
    import yaml
    from curobo.util import logger
    logger.setup_logger(level="error", logger_name="curobo")

    # Keep defaults moderate enough for 16 GB class GPUs, and allow per-run overrides.
    CUROBO_TRAJOPT_SEEDS = int(os.environ.get("CUROBO_TRAJOPT_SEEDS", "16"))
    CUROBO_BATCH_GRAPH_SEEDS = int(os.environ.get("CUROBO_BATCH_GRAPH_SEEDS", "1"))
    CUROBO_MAX_ATTEMPTS = int(os.environ.get("CUROBO_MAX_ATTEMPTS", "24"))
    CUROBO_BATCH_WARMUP = int(os.environ.get("CUROBO_BATCH_WARMUP", "0"))
    # Finetune-stage overrides. None keeps CuRobo's defaults (finetune_attempts=5,
    # finetune_dt_scale=0.85). More attempts or a looser dt_scale give finetune more room
    # on hard cases such as a close final descent near an obstacle.
    CUROBO_FINETUNE_ATTEMPTS = (int(os.environ["CUROBO_FINETUNE_ATTEMPTS"])
                                if os.environ.get("CUROBO_FINETUNE_ATTEMPTS") else None)
    CUROBO_FINETUNE_DT_SCALE = (float(os.environ["CUROBO_FINETUNE_DT_SCALE"])
                                if os.environ.get("CUROBO_FINETUNE_DT_SCALE") else None)
    # Radius (m) of the spheres that approximate a held object in CuRobo; 0.001 is
    # CuRobo's own default. Larger padding can make CuRobo reject a placement close to a
    # surface even when there is no physical collision.
    CUROBO_ATTACH_SPHERE_RADIUS = float(os.environ.get("CUROBO_ATTACH_SPHERE_RADIUS", "0.001"))
    CUROBO_NEAR_CONTACT_TRAJOPT_SEEDS = int(os.environ.get("CUROBO_NEAR_CONTACT_TRAJOPT_SEEDS", "4"))
    CUROBO_NEAR_CONTACT_MAX_ATTEMPTS = int(os.environ.get("CUROBO_NEAR_CONTACT_MAX_ATTEMPTS", "8"))
    CUROBO_NEAR_CONTACT_FINETUNE_ATTEMPTS = int(os.environ.get("CUROBO_NEAR_CONTACT_FINETUNE_ATTEMPTS", "8"))
    CUROBO_NEAR_CONTACT_FINETUNE_DT_SCALE = float(os.environ.get("CUROBO_NEAR_CONTACT_FINETUNE_DT_SCALE", "1.0"))
    CUROBO_NEAR_CONTACT_COLLISION_ACTIVATION = float(os.environ.get("CUROBO_NEAR_CONTACT_COLLISION_ACTIVATION", "0.005"))
    CUROBO_NEAR_CONTACT_POSITION_THRESHOLD = float(os.environ.get("CUROBO_NEAR_CONTACT_POSITION_THRESHOLD", "0.01"))
    CUROBO_NEAR_CONTACT_INTERPOLATION_DT = float(os.environ.get("CUROBO_NEAR_CONTACT_INTERPOLATION_DT", "0.01"))

    class CuroboPlanner:

        @staticmethod
        def _position_only_pose_cost_metric(tensor_args):
            """PoseCostMetric that scores position only (rotation weights zeroed), used for
            relax_orientation goals."""
            return PoseCostMetric(
                reach_partial_pose=True,
                reach_vec_weight=tensor_args.to_device([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            )

        def __init__(
            self,
            robot_origion_pose,
            active_joints_name,
            all_joints,
            yml_path=None,
            collision_cache: dict = {"mesh": 1, "obb": 1},
        ):
            super().__init__()
            ta.setup_logging("CRITICAL")  # hide logging
            logger.setup_logger(level="error", logger_name="'curobo")

            if yml_path != None:
                self.yml_path = yml_path
            else:
                raise ValueError("[Planner.py]: CuroboPlanner yml_path is None!")
            self.robot_origion_pose = robot_origion_pose
            self.active_joints_name = active_joints_name
            self.all_joints = all_joints

            # translate from baselink to arm's base
            with open(self.yml_path, "r") as f:
                yml_data = yaml.safe_load(f)
            self.frame_bias = yml_data["planner"]["frame_bias"]

            # motion generation
            if True:
                world_config = {
                    "cuboid": {
                        # "table": {
                        #     "dims": [0.7, 2, 0.04],  # x, y, z
                        #     "pose": [
                        #         self.robot_origion_pose.p[1],
                        #         0.0,
                        #         0.74 - self.robot_origion_pose.p[2],
                        #         1,
                        #         0,
                        #         0,
                        #         0.0,
                        #     ],  # x, y, z, qw, qx, qy, qz
                        # },
                    },
                    "mesh": {}
                }
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                self.yml_path,
                world_config,
                interpolation_dt=1 / 250,
                num_trajopt_seeds=CUROBO_TRAJOPT_SEEDS,
                collision_cache=collision_cache,
            )

            self.motion_gen = MotionGen(motion_gen_config)
            # warmup() pre-compiles CUDA graphs (about 6 s per planner, two per dual-arm robot).
            # Policy evaluation only calls plan_grippers, so ROBOTWIN_SKIP_CUROBO_WARMUP=1 skips
            # it; the planner still works and compiles on its first real plan.
            if os.environ.get("ROBOTWIN_SKIP_CUROBO_WARMUP") != "1":
                self.motion_gen.warmup()
            self.motion_gen_near_contact = None

            # Batch planning is memory-hungry. Defer its construction until a caller
            # actually needs plan_batch() so ordinary single-goal planning stays lighter.
            self._motion_gen_batch_config = MotionGenConfig.load_from_robot_config(
                self.yml_path,
                world_config,
                interpolation_dt=1 / 250,
                num_trajopt_seeds=CUROBO_TRAJOPT_SEEDS,
                num_graph_seeds=CUROBO_BATCH_GRAPH_SEEDS,
                collision_cache=collision_cache,
            )
            self.motion_gen_batch = None
            # World and attachment state applied so far. The batch and near-contact planners
            # are created lazily and replay this state when they are built.
            self._last_world_config = None
            self._attached_objects = []
            if CUROBO_BATCH_WARMUP:
                self._ensure_motion_gen_batch()

        def _ensure_motion_gen_batch(self):
            if self.motion_gen_batch is None:
                self.motion_gen_batch = MotionGen(self._motion_gen_batch_config)
                self.motion_gen_batch.warmup(batch=CONFIGS.ROTATE_NUM)
                if self._last_world_config is not None:
                    self.motion_gen_batch.clear_world_cache()
                    self.motion_gen_batch.update_world(self._last_world_config)
                for obstacle, joint_states in self._attached_objects:
                    self.motion_gen_batch.attach_external_objects_to_robot(
                        joint_state=joint_states,
                        external_objects=[obstacle],
                        link_name="attached_object",
                        surface_sphere_radius=CUROBO_ATTACH_SPHERE_RADIUS,
                        voxelize_method="ray",
                    )
            return self.motion_gen_batch

        def _active_motion_gens(self):
            gens = [self.motion_gen]
            if self.motion_gen_batch is not None:
                gens.append(self.motion_gen_batch)
            if self.motion_gen_near_contact is not None:
                gens.append(self.motion_gen_near_contact)
            return gens

        def _ensure_near_contact_motion_gen(self):
            if self.motion_gen_near_contact is None:
                config = MotionGenConfig.load_from_robot_config(
                    self.yml_path, {"cuboid": {}, "mesh": {}},
                    interpolation_dt=CUROBO_NEAR_CONTACT_INTERPOLATION_DT,
                    num_trajopt_seeds=CUROBO_NEAR_CONTACT_TRAJOPT_SEEDS,
                    collision_cache={"mesh": 1, "obb": 1},
                    collision_activation_distance=CUROBO_NEAR_CONTACT_COLLISION_ACTIVATION,
                    position_threshold=CUROBO_NEAR_CONTACT_POSITION_THRESHOLD,
                    use_cuda_graph=False,
                )
                self.motion_gen_near_contact = MotionGen(config)
                self.motion_gen_near_contact.warmup()
                if self._last_world_config is not None:
                    self.motion_gen_near_contact.clear_world_cache()
                    self.motion_gen_near_contact.update_world(self._last_world_config)
                for obstacle, joint_states in self._attached_objects:
                    ok = self.motion_gen_near_contact.attach_external_objects_to_robot(
                        joint_state=joint_states, external_objects=[obstacle],
                        link_name="attached_object",
                        surface_sphere_radius=CUROBO_ATTACH_SPHERE_RADIUS,
                        voxelize_method="ray",
                    )
                    assert ok
            return self.motion_gen_near_contact

        def plan_path(
            self,
            curr_joint_pos,
            target_gripper_pose,
            constraint_pose=None,
            approach_axis=None,
            arms_tag=None,
            relax_orientation=False,
            approach_offset=0.05,
            tstep_fraction=0.8,
            near_contact=False,
        ):
            world_base_pose = np.concatenate([
                np.array(self.robot_origion_pose.p),
                np.array(self.robot_origion_pose.q),
            ])
            world_target_pose = np.concatenate([np.array(target_gripper_pose.p), np.array(target_gripper_pose.q)])
            target_pose_p, target_pose_q = self._trans_from_world_to_base(world_base_pose, world_target_pose)
            if not ("aloha-agilex" in self.yml_path):
                target_pose_p[0] += self.frame_bias[0]
                target_pose_p[1] += self.frame_bias[1]
                target_pose_p[2] += self.frame_bias[2]
            else: # patch for aloha-agilex
                T_target = t3d.affines.compose(target_pose_p, t3d.quaternions.quat2mat(target_pose_q), [1, 1, 1])
                T_bias = t3d.affines.compose(self.frame_bias, np.eye(3), [1, 1, 1])

                if arms_tag == "left":
                    rot = t3d.axangles.axangle2mat([0, 0, 1], -0.02)
                elif arms_tag == "right":
                    rot = t3d.axangles.axangle2mat([0, 0, 1], -0.01)
                else:
                    raise ValueError(f"Invalid arms_tag: {arms_tag}")

                T_rot = t3d.affines.compose([0, 0, 0], rot, [1, 1, 1])
                T_new = T_rot @ T_bias @ T_target
                target_pose_p = T_new[:3, 3]
                target_pose_q = t3d.quaternions.mat2quat(T_new[:3, :3])

            goal_pose_of_ee = CuroboPose.from_list(list(target_pose_p) + list(target_pose_q))
            joint_indices = [self.all_joints.index(name) for name in self.active_joints_name if name in self.all_joints]
            joint_angles = [curr_joint_pos[index] for index in joint_indices]
            joint_angles = [round(angle, 5) for angle in joint_angles]  # avoid the precision problem
            start_joint_states = JointState.from_position(
                torch.tensor(joint_angles, dtype=torch.float32).cuda().reshape(1, -1),
                joint_names=self.active_joints_name,
            )
            # plan
            motion_gen = self._ensure_near_contact_motion_gen() if near_contact else self.motion_gen
            if near_contact:
                plan_config = MotionGenPlanConfig(
                    max_attempts=CUROBO_NEAR_CONTACT_MAX_ATTEMPTS,
                    finetune_attempts=CUROBO_NEAR_CONTACT_FINETUNE_ATTEMPTS,
                    finetune_dt_scale=CUROBO_NEAR_CONTACT_FINETUNE_DT_SCALE,
                )
            else:
                _finetune_kwargs = {}
                if CUROBO_FINETUNE_ATTEMPTS is not None:
                    _finetune_kwargs["finetune_attempts"] = CUROBO_FINETUNE_ATTEMPTS
                if CUROBO_FINETUNE_DT_SCALE is not None:
                    _finetune_kwargs["finetune_dt_scale"] = CUROBO_FINETUNE_DT_SCALE
                plan_config = MotionGenPlanConfig(max_attempts=CUROBO_MAX_ATTEMPTS, **_finetune_kwargs)
            if approach_axis is not None:
                # Unlike the hold_partial_pose metric below, create_grasp_approach_metric sets
                # offset_tstep_fraction >= 0, so CuRobo skips the start/goal orientation-match
                # check (INVALID_PARTIAL_POSE_COST_METRIC). It enforces a straight-line approach
                # along one axis over the last tstep_fraction of the trajectory only; CuRobo
                # resets the metric after plan_single.
                if os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1":
                    print(f"[plan_path] using approach_axis={approach_axis} "
                          f"approach_offset={approach_offset} tstep_fraction={tstep_fraction} "
                          f"pose_cost_metric")
                plan_config.pose_cost_metric = PoseCostMetric.create_grasp_approach_metric(
                    offset_position=approach_offset, linear_axis=approach_axis,
                    tstep_fraction=tstep_fraction,
                    tensor_args=motion_gen.tensor_args,
                )
            elif constraint_pose is not None:
                pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=motion_gen.tensor_args.to_device(constraint_pose),
                )
                plan_config.pose_cost_metric = pose_cost_metric
            elif relax_orientation:
                # Position-only goal for transit moves (carrying a grasped object around or
                # over an obstacle), where the orientation en route does not matter.
                # MotionGen resets pose_cost_metric at the end of planning.
                if os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1":
                    print("[plan_path] using relax_orientation pose_cost_metric")
                plan_config.pose_cost_metric = self._position_only_pose_cost_metric(motion_gen.tensor_args)

            result = motion_gen.plan_single(start_joint_states, goal_pose_of_ee, plan_config)

            # ------------------------------------------
            if result.success.item() == False and os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1":
                # Failures are frequent during candidate search; the reason is returned in
                # fail_reason, so print only when ROBOTWIN_LOG_MOVE=1.
                print("[Error]: CuroboPlanner plan_path failed:", result.status)
            # ------------------------------------------

            # output
            res_result = dict()
            # After trajopt, MotionGen fills position_error, rotation_error and
            # interpolated_plan on both success and failure (e.g. FINETUNE_TRAJOPT_FAIL), so a
            # caller can judge whether a failed attempt is usable as a waypoint. They are
            # None when trajopt did not run (e.g. IK_FAIL).
            if result.position_error is not None:
                res_result["position_error"] = float(result.position_error.reshape(-1)[0].item())
            if result.rotation_error is not None:
                res_result["rotation_error"] = float(result.rotation_error.reshape(-1)[0].item())
            if result.interpolated_plan is not None:
                res_result["position"] = np.array(result.interpolated_plan.position.to("cpu"))
                res_result["velocity"] = np.array(result.interpolated_plan.velocity.to("cpu"))

            if result.success.item() == False:
                res_result["status"] = "Fail"
                res_result["fail_reason"] = str(result.status)
            else:
                res_result["status"] = "Success"
            return res_result

        def plan_batch(
            self,
            curr_joint_pos,
            target_gripper_pose_list,
            constraint_pose=None,
            arms_tag=None,
        ):
            """
            Plan a batch of trajectories for multiple target poses.

            Input:
                - curr_joint_pos: List of current joint angles (1 x n)
                - target_gripper_pose_list: List of target poses [sapien.Pose, sapien.Pose, ...]

            Output:
                - result['status']: numpy array of string values indicating "Success"/"Fail" for each pose
                - result['position']: numpy array of joint positions with shape (n x m x l)
                  where n is number of target poses, m is number of waypoints, l is number of joints
                - result['velocity']: numpy array of joint velocities with same shape as position
            """

            num_poses = len(target_gripper_pose_list)
            # transformation from world to arm's base
            world_base_pose = np.concatenate([
                np.array(self.robot_origion_pose.p),
                np.array(self.robot_origion_pose.q),
            ])
            poses_list = []
            for target_gripper_pose in target_gripper_pose_list:
                world_target_pose = np.concatenate([np.array(target_gripper_pose.p), np.array(target_gripper_pose.q)])
                base_target_pose_p, base_target_pose_q = self._trans_from_world_to_base(world_base_pose, world_target_pose)

                if not ("aloha-agilex" in self.yml_path):
                    base_target_pose_p[0] += self.frame_bias[0]
                    base_target_pose_p[1] += self.frame_bias[1]
                    base_target_pose_p[2] += self.frame_bias[2]
                else: # patch for aloha-agilex
                    T_target = t3d.affines.compose(base_target_pose_p, t3d.quaternions.quat2mat(base_target_pose_q), [1, 1, 1])
                    T_bias = t3d.affines.compose(self.frame_bias, np.eye(3), [1, 1, 1])

                    if arms_tag == "left":
                        rot = t3d.axangles.axangle2mat([0, 0, 1], -0.02)
                    elif arms_tag == "right":
                        rot = t3d.axangles.axangle2mat([0, 0, 1], -0.01)
                    else:
                        raise ValueError(f"Invalid arms_tag: {arms_tag}")

                    T_rot = t3d.affines.compose([0, 0, 0], rot, [1, 1, 1])
                    T_new = T_rot @ T_bias @ T_target
                    base_target_pose_p = T_new[:3, 3]
                    base_target_pose_q = t3d.quaternions.mat2quat(T_new[:3, :3])

                base_target_pose_list = list(base_target_pose_p) + list(base_target_pose_q)
                poses_list.append(base_target_pose_list)

            poses_cuda = torch.tensor(poses_list, dtype=torch.float32).cuda()
            goal_pose_of_ee = CuroboPose(poses_cuda[:, :3], poses_cuda[:, 3:])
            joint_indices = [self.all_joints.index(name) for name in self.active_joints_name if name in self.all_joints]
            joint_angles = [curr_joint_pos[index] for index in joint_indices]
            joint_angles = [round(angle, 5) for angle in joint_angles]  # avoid the precision problem
            joint_angles_cuda = (torch.tensor(joint_angles, dtype=torch.float32).cuda().reshape(1, -1))
            joint_angles_cuda = torch.cat([joint_angles_cuda] * num_poses, dim=0)
            start_joint_states = JointState.from_position(joint_angles_cuda, joint_names=self.active_joints_name)
            # plan
            _finetune_kwargs = {}
            if CUROBO_FINETUNE_ATTEMPTS is not None:
                _finetune_kwargs["finetune_attempts"] = CUROBO_FINETUNE_ATTEMPTS
            if CUROBO_FINETUNE_DT_SCALE is not None:
                _finetune_kwargs["finetune_dt_scale"] = CUROBO_FINETUNE_DT_SCALE
            plan_config = MotionGenPlanConfig(max_attempts=CUROBO_MAX_ATTEMPTS, **_finetune_kwargs)
            if constraint_pose is not None:
                pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=self.motion_gen.tensor_args.to_device(constraint_pose),
                )
                plan_config.pose_cost_metric = pose_cost_metric

            try:
                motion_gen_batch = self._ensure_motion_gen_batch()
                result = motion_gen_batch.plan_batch(start_joint_states, goal_pose_of_ee, plan_config)
            except Exception as e:
                return {"status": ["Failure" for i in range(10)]}

            # output
            res_result = dict()
            # Convert boolean success values to "Success"/"Failure" strings
            success_array = result.success.cpu().numpy()
            status_array = np.array(["Success" if s else "Failure" for s in success_array], dtype=object)
            res_result["status"] = status_array

            if np.all(res_result["status"] == "Failure"):
                return res_result

            res_result["position"] = np.array(result.interpolated_plan.position.to("cpu"))
            res_result["velocity"] = np.array(result.interpolated_plan.velocity.to("cpu"))
            return res_result

        def check_ik_batch(self, target_gripper_pose_list, arms_tag=None, relax_orientation=False):
            """Collision-aware IK feasibility check per pose (no trajectory optimization).

            Much cheaper than plan_batch; used to map the arm's reachable envelope.
            Returns a bool array, True where a collision-free IK solution exists.
            relax_orientation=True runs position-only IK (rotation weights zeroed) to test
            whether a position is reachable under any orientation."""
            world_base_pose = np.concatenate([
                np.array(self.robot_origion_pose.p),
                np.array(self.robot_origion_pose.q),
            ])
            poses_list = []
            for target_gripper_pose in target_gripper_pose_list:
                world_target_pose = np.concatenate([np.array(target_gripper_pose.p), np.array(target_gripper_pose.q)])
                base_target_pose_p, base_target_pose_q = self._trans_from_world_to_base(world_base_pose, world_target_pose)

                if not ("aloha-agilex" in self.yml_path):
                    base_target_pose_p[0] += self.frame_bias[0]
                    base_target_pose_p[1] += self.frame_bias[1]
                    base_target_pose_p[2] += self.frame_bias[2]
                else:  # patch for aloha-agilex
                    T_target = t3d.affines.compose(base_target_pose_p, t3d.quaternions.quat2mat(base_target_pose_q), [1, 1, 1])
                    T_bias = t3d.affines.compose(self.frame_bias, np.eye(3), [1, 1, 1])

                    if arms_tag == "left":
                        rot = t3d.axangles.axangle2mat([0, 0, 1], -0.02)
                    elif arms_tag == "right":
                        rot = t3d.axangles.axangle2mat([0, 0, 1], -0.01)
                    else:
                        raise ValueError(f"Invalid arms_tag: {arms_tag}")

                    T_rot = t3d.affines.compose([0, 0, 0], rot, [1, 1, 1])
                    T_new = T_rot @ T_bias @ T_target
                    base_target_pose_p = T_new[:3, 3]
                    base_target_pose_q = t3d.quaternions.mat2quat(T_new[:3, :3])

                base_target_pose_list = list(base_target_pose_p) + list(base_target_pose_q)
                poses_list.append(base_target_pose_list)

            poses_cuda = torch.tensor(poses_list, dtype=torch.float32).cuda()
            # Solve one pose at a time: solve_batch conflicts with the batch size the
            # ik_solver's CUDA graph was fixed to by warmup() (batch=1).
            if relax_orientation:
                self.motion_gen.ik_solver.update_pose_cost_metric(
                    self._position_only_pose_cost_metric(self.motion_gen.tensor_args))
            try:
                results = []
                for i in range(poses_cuda.shape[0]):
                    goal_pose = CuroboPose(poses_cuda[i:i + 1, :3], poses_cuda[i:i + 1, 3:])
                    ik_result = self.motion_gen.ik_solver.solve_single(goal_pose)
                    results.append(bool(ik_result.success.reshape(-1)[0].item()))
            finally:
                if relax_orientation:
                    self.motion_gen.ik_solver.update_pose_cost_metric(PoseCostMetric.reset_metric())
            return np.array(results, dtype=bool)

        def plan_grippers(self, now_val, target_val):
            num_step = 200
            dis_val = target_val - now_val
            step = dis_val / num_step
            res = {}
            vals = np.linspace(now_val, target_val, num_step)
            res["num_step"] = num_step
            res["per_step"] = step
            res["result"] = vals
            return res

        def _trans_from_world_to_base(self, base_pose, target_pose):
            '''
                transform target pose from world frame to base frame
                base_pose: np.array([x, y, z, qw, qx, qy, qz])
                target_pose: np.array([x, y, z, qw, qx, qy, qz])
            '''
            base_p, base_q = base_pose[0:3], base_pose[3:]
            target_p, target_q = target_pose[0:3], target_pose[3:]
            rel_p = target_p - base_p
            wRb = t3d.quaternions.quat2mat(base_q)
            wRt = t3d.quaternions.quat2mat(target_q)
            result_p = wRb.T @ rel_p
            result_q = t3d.quaternions.mat2quat(wRb.T @ wRt)
            return result_p, result_q
        
        def update_world(self, collision_dict, arms_tag):
            """Update CuRobo Collision World Model with new collision objects"""

            collision_dict = self.collision_dict_world_to_arm_base(collision_dict, arms_tag)
            world_config = WorldConfig.from_dict(collision_dict)
            self._last_world_config = world_config

            for gen in self._active_motion_gens():
                gen.clear_world_cache()
                gen.update_world(world_config)

        def world_to_arm_base_pose(self, world_pose, arms_tag=None):
            """
            Convert a pose from world frame to arm base frame.

            Args:
                world_pose: either
                    - object with .p and .q
                    - or iterable [x, y, z, qw, qx, qy, qz]
                arms_tag: "left" or "right" (required for aloha-agilex patch)

            Returns:
                (p, q) where:
                    p: np.ndarray (3,)
                    q: np.ndarray (4,) quaternion
            """
            import numpy as np
            import transforms3d as t3d

            # --- Extract world pose ---
            if hasattr(world_pose, "p") and hasattr(world_pose, "q"):
                world_target_pose = np.concatenate([
                    np.array(world_pose.p),
                    np.array(world_pose.q),
                ])
            else:
                world_target_pose = np.array(world_pose)

            # --- World base pose ---
            world_base_pose = np.concatenate([
                np.array(self.robot_origion_pose.p),
                np.array(self.robot_origion_pose.q),
            ])

            # --- World -> robot base ---
            target_pose_p, target_pose_q = self._trans_from_world_to_base(
                world_base_pose,
                world_target_pose,
            )

            # --- Frame bias / aloha patch ---
            if "aloha-agilex" not in self.yml_path:
                target_pose_p = target_pose_p + np.array(self.frame_bias)

            else: # patch for aloha-agilex
                T_target = t3d.affines.compose(
                    target_pose_p,
                    t3d.quaternions.quat2mat(target_pose_q),
                    [1, 1, 1],
                )

                T_bias = t3d.affines.compose(
                    self.frame_bias,
                    np.eye(3),
                    [1, 1, 1],
                )

                if arms_tag == "left":
                    rot = t3d.axangles.axangle2mat([0, 0, 1], -0.02)
                elif arms_tag == "right":
                    rot = t3d.axangles.axangle2mat([0, 0, 1], -0.01)
                else:
                    raise ValueError(f"Invalid arms_tag: {arms_tag}")

                T_rot = t3d.affines.compose([0, 0, 0], rot, [1, 1, 1])
                T_new = T_rot @ T_bias @ T_target

                target_pose_p = T_new[:3, 3]
                target_pose_q = t3d.quaternions.mat2quat(T_new[:3, :3])

            return target_pose_p, target_pose_q

        def collision_dict_world_to_arm_base(self, collision_dict: dict, arms_tag: str | None = None) -> dict:
            """
            Returns a copy of collision_dict where every obstacle pose is converted from world frame
            to arm base frame using self.world_to_arm_base_pose(...).

            Expects each obstacle to have a 'pose' field as [x,y,z,qw,qx,qy,qz].
            """
            import copy
            import numpy as np

            out = copy.deepcopy(collision_dict)

            # Convert a flat [x, y, z, qw, qx, qy, qz] world pose to the arm base frame.
            def _convert_pose_list(pose_list):
                p, q = self.world_to_arm_base_pose(pose_list, arms_tag=arms_tag)
                return list(np.asarray(p).tolist()) + list(np.asarray(q).tolist())

            # meshes
            if "mesh" in out and out["mesh"]:
                for name, md in out["mesh"].items():
                    if "pose" in md and md["pose"] is not None:
                        md["pose"] = _convert_pose_list(md["pose"])

            # cuboids
            if "cuboid" in out and out["cuboid"]:
                for name, cd in out["cuboid"].items():
                    if "pose" in cd and cd["pose"] is not None:
                        cd["pose"] = _convert_pose_list(cd["pose"])

            # Other primitive types (sphere, capsule, cylinder) are copied unchanged, i.e. left in world frame.

            return out
        
        def visualize_world_config(self, world_model):
            """Visualize CuRobo Collision World Model: motion_gen.world_coll_checker.world_model"""
            from scipy.spatial.transform import Rotation

            scene = trimesh.Scene()
            
            # Add all cuboids
            for cuboid in world_model.cuboid:
                # Create box
                box = trimesh.creation.box(extents=cuboid.dims)
                
                # Create transformation matrix
                pose = cuboid.pose
                transform = np.eye(4)
                
                # Rotation from quaternion [qw, qx, qy, qz]
                quat = [pose[3], pose[4], pose[5], pose[6]]
                rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy: [qx,qy,qz,qw]
                transform[:3, :3] = rot.as_matrix()
                
                # Translation
                transform[:3, 3] = [pose[0], pose[1], pose[2]]
                
                # Apply transform
                box.apply_transform(transform)
                
                # Set color (red for collision objects)
                box.visual.face_colors = [255, 0, 0, 100]  # Red, semi-transparent
                
                # Add to scene
                scene.add_geometry(box, node_name=cuboid.name)
            
            # Add all meshes
            for mesh_obj in world_model.mesh:
                try:
                    # Load mesh from file
                    if mesh_obj.file_path:
                        mesh = trimesh.load(mesh_obj.file_path, force='mesh')
                        
                        # Apply scale
                        mesh.apply_scale(mesh_obj.scale)
                        
                        # Apply pose
                        pose = mesh_obj.pose
                        transform = np.eye(4)
                        
                        quat = [pose[3], pose[4], pose[5], pose[6]]
                        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
                        transform[:3, :3] = rot.as_matrix()
                        transform[:3, 3] = [pose[0], pose[1], pose[2]]
                        
                        mesh.apply_transform(transform)

                        # Set color (blue for meshes)
                        mesh.visual.face_colors = [0, 0, 255, 100]
                        
                        scene.add_geometry(mesh, node_name=mesh_obj.name)
                        
                except Exception as e:
                    print(f"Could not load mesh {mesh_obj.name}: {e}")
                
            return scene

        def enable_obstacle(self, enable: bool, mesh_names: list[str] = [], obb_names: list[str] = []):
            for gen in self._active_motion_gens():
                for name in mesh_names:
                    gen.world_coll_checker.enable_mesh(enable, name=name)
                for name in obb_names:
                    gen.world_coll_checker.enable_obb(enable, name=name)

        def attach_object(self, object: dict, curr_joint_pos: list, arms_tag: str):
            """
            Attach an object to the robot in Curobo Planning.
            Args:
                object: dict, the object to attach
                curr_joint_pos: list, the current joint positions
                arms_tag: str, the arm tag
            Returns:
                None
            """
            p,q = self.world_to_arm_base_pose(object["pose"], arms_tag=arms_tag)
            pose = np.concatenate([p, q]).tolist()
            # -----------------------------------------------
            obstacle = Mesh(
                name=object["name"],
                pose=pose,
                file_path=object["file_path"],
                scale=object["scale"],
            ) 
            # -----------------------------------------------
            joint_indices = [self.all_joints.index(name) for name in self.active_joints_name if name in self.all_joints]
            joint_angles = [curr_joint_pos[index] for index in joint_indices]
            joint_angles = [round(angle, 5) for angle in joint_angles]
            joint_states = JointState.from_position(
                torch.tensor(joint_angles, dtype=torch.float32).cuda().reshape(1, -1),
                joint_names=self.active_joints_name,
            )
            # -----------------------------------------------

            for mg in self._active_motion_gens():
                ok = mg.attach_external_objects_to_robot(
                    joint_state=joint_states,
                    external_objects=[obstacle],
                    link_name="attached_object",
                    surface_sphere_radius=CUROBO_ATTACH_SPHERE_RADIUS,
                    voxelize_method="ray",
                )
                assert ok
            self._attached_objects.append((obstacle, joint_states))

        def detach_object(self):
            for mg in self._active_motion_gens():
                mg.detach_object_from_robot(link_name="attached_object",)
            self._attached_objects = []

        def visualize_attached_objects(self, curr_joint_pos: list):
            world_model = self.motion_gen.world_coll_checker.world_model
            scene = self.visualize_world_config(world_model)
            # ---- Add attached object spheres ----
            kc = self.motion_gen.robot_cfg.kinematics.kinematics_config
            ss = kc.get_link_spheres("attached_object")

            valid_idx = (ss[:,3] > 0).nonzero().flatten()

            # get EE pose for correct transform
            joint_indices = [self.all_joints.index(name) for name in self.active_joints_name if name in self.all_joints]
            joint_angles = [curr_joint_pos[index] for index in joint_indices]
            joint_angles = [round(angle, 5) for angle in joint_angles]
            # -----------------------------------------------
            joint_states = JointState.from_position(
                torch.tensor(joint_angles, dtype=torch.float32).cuda().reshape(1, -1),
                joint_names=self.active_joints_name,
            )

            ks = self.motion_gen.compute_kinematics(joint_states)
            w_T_ee = ks.ee_pose
            from curobo.types.math import Pose

            for i in valid_idx.tolist():
                x, y, z, r = ss[i].tolist()

                # sphere in EE frame
                ee_T_sphere = Pose.from_list([x, y, z, 0, 0, 0, 1])
                w_T_sphere = w_T_ee.multiply(ee_T_sphere)
                center = w_T_sphere.position.squeeze().cpu().numpy()

                # create trimesh sphere
                sphere_mesh = trimesh.creation.icosphere(radius=r, subdivisions=2)
                sphere_mesh.apply_translation(center)

                # green for attached object
                sphere_mesh.visual.face_colors = [0, 255, 0, 180]

                scene.add_geometry(sphere_mesh)
            
            scene.show()

    
except Exception as e:
    print('[planner.py]: Something wrong happened when importing CuroboPlanner! Please check if Curobo is installed correctly. If the problem still exists, you can install Curobo from https://github.com/NVlabs/curobo manually.')
    print('Exception traceback:')
    traceback.print_exc()


# ********************** MplibPlanner **********************
class MplibPlanner:
    # links=None, joints=None
    def __init__(
        self,
        urdf_path,
        srdf_path,
        move_group,
        robot_origion_pose,
        robot_entity,
        planner_type="mplib_RRT",
        scene=None,
    ):
        super().__init__()
        ta.setup_logging("CRITICAL")  # hide logging

        links = [link.get_name() for link in robot_entity.get_links()]
        joints = [joint.get_name() for joint in robot_entity.get_active_joints()]

        if scene is None:
            self.planner = mplib.Planner(
                urdf=urdf_path,
                srdf=srdf_path,
                move_group=move_group,
                user_link_names=links,
                user_joint_names=joints,
                use_convex=False,
            )
            self.planner.set_base_pose(robot_origion_pose)
        else:
            planning_world = SapienPlanningWorld(scene, [robot_entity])
            self.planner = SapienPlanner(planning_world, move_group)

        self.planner_type = planner_type
        self.plan_step_lim = 2500
        self.TOPP = self.planner.TOPP

    def show_info(self):
        print("joint_limits", self.planner.joint_limits)
        print("joint_acc_limits", self.planner.joint_acc_limits)

    def plan_pose(
        self,
        now_qpos,
        target_pose,
        use_point_cloud=False,
        use_attach=False,
        arms_tag=None,
        try_times=2,
        log=True,
    ):
        result = {}
        result["status"] = "Fail"

        now_try_times = 1
        while result["status"] != "Success" and now_try_times < try_times:
            result = self.planner.plan_pose(
                goal_pose=target_pose,
                current_qpos=np.array(now_qpos),
                time_step=1 / 250,
                planning_time=5,
                # rrt_range=0.05
                # =================== mplib 0.1.1 ===================
                # use_point_cloud=use_point_cloud,
                # use_attach=use_attach,
                # planner_name="RRTConnect"
            )
            now_try_times += 1

        if result["status"] != "Success":
            if log:
                print(f"\n {arms_tag} arm planning failed ({result['status']}) !")
        else:
            n_step = result["position"].shape[0]
            if n_step > self.plan_step_lim:
                if log:
                    print(f"\n {arms_tag} arm planning wrong! (step = {n_step})")
                result["status"] = "Fail"

        return result

    def plan_screw(
        self,
        now_qpos,
        target_pose,
        use_point_cloud=False,
        use_attach=False,
        arms_tag=None,
        log=False,
    ):
        """
        Interpolative planning with screw motion.
        Will not avoid collision and will fail if the path contains collision.
        """
        result = self.planner.plan_screw(
            goal_pose=target_pose,
            current_qpos=now_qpos,
            time_step=1 / 250,
            # =================== mplib 0.1.1 ===================
            # use_point_cloud=use_point_cloud,
            # use_attach=use_attach,
        )

        # plan fail
        if result["status"] != "Success":
            if log:
                print(f"\n {arms_tag} arm planning failed ({result['status']}) !")
            # return result
        else:
            n_step = result["position"].shape[0]
            # plan step lim
            if n_step > self.plan_step_lim:
                if log:
                    print(f"\n {arms_tag} arm planning wrong! (step = {n_step})")
                result["status"] = "Fail"

        return result

    def plan_path(
        self,
        now_qpos,
        target_pose,
        use_point_cloud=False,
        use_attach=False,
        arms_tag=None,
        log=True,
    ):
        """
        Interpolative planning with screw motion.
        Will not avoid collision and will fail if the path contains collision.
        """
        if self.planner_type == "mplib_RRT":
            result = self.plan_pose(
                now_qpos,
                target_pose,
                use_point_cloud,
                use_attach,
                arms_tag,
                try_times=10,
                log=log,
            )
        elif self.planner_type == "mplib_screw":
            result = self.plan_screw(now_qpos, target_pose, use_point_cloud, use_attach, arms_tag, log)

        return result

    def plan_grippers(self, now_val, target_val):
        num_step = 200  # TODO
        dis_val = target_val - now_val
        per_step = dis_val / num_step
        res = {}
        vals = np.linspace(now_val, target_val, num_step)
        res["num_step"] = num_step
        res["per_step"] = per_step  # dis per step
        res["result"] = vals
        return res
