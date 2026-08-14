"""Diagnostic: reproduce INVALID_START_STATE_WORLD_COLLISION in kitchen-S tasks."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from setup_paths import setup_paths
setup_paths()

import numpy as np
import yaml
import torch
from pathlib import Path

bench_root = Path(os.environ["BENCH_ROOT"])
sim_root = Path(os.environ["SIM_ROOT"])
os.chdir(sim_root)

from envs import CONFIGS_PATH

def get_embodiment_config(robot_file):
    with open(os.path.join(robot_file, "config.yml"), "r") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)

def build_cfg(seed):
    config_path = bench_root / "bench_task_config" / "bench_demo_kitchens_clean.yml"
    with open(config_path, "r") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg["task_name"] = "put_spoon_on_plate_ks"
    cfg["render_freq"] = 0
    cfg["now_ep_num"] = 0
    cfg["seed"] = seed
    cfg["need_plan"] = True
    cfg["save_data"] = False

    embodiment_type = cfg.get("embodiment", ["aloha-agilex"])
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r") as f:
        etypes = yaml.load(f.read(), Loader=yaml.FullLoader)
    rf = etypes[embodiment_type[0]]["file_path"]
    cfg["left_robot_file"] = rf
    cfg["right_robot_file"] = rf
    cfg["dual_arm_embodied"] = True
    cfg["left_embodiment_config"] = get_embodiment_config(rf)
    cfg["right_embodiment_config"] = get_embodiment_config(rf)
    return cfg

def dump_planner_state(env, label):
    """Print CuRobo collision world state for both planners."""
    for side in ["left", "right"]:
        planner = getattr(env.robot, f"{side}_planner")
        for gen_name in ["motion_gen", "motion_gen_batch"]:
            gen = getattr(planner, gen_name)
            wcc = gen.world_coll_checker

            # OBB state
            n_obbs = wcc._env_n_obbs
            cube_enable = wcc._cube_tensor_list[2] if wcc._cube_tensor_list is not None else None
            obb_names = wcc._env_obbs_names

            # Mesh state
            n_mesh = wcc._env_n_mesh if hasattr(wcc, '_env_n_mesh') and wcc._env_n_mesh is not None else None
            mesh_enable = wcc._mesh_tensor_list[2] if hasattr(wcc, '_mesh_tensor_list') and wcc._mesh_tensor_list is not None else None
            mesh_names = wcc._env_mesh_names if hasattr(wcc, '_env_mesh_names') else None

            print(f"\n[{label}] {side}/{gen_name}:")
            print(f"  n_obbs={n_obbs}, cube_enable={cube_enable}")
            if obb_names is not None:
                print(f"  obb_names={obb_names}")
            print(f"  n_mesh={n_mesh}, mesh_enable={mesh_enable}")
            if mesh_names is not None:
                active = [n for names in mesh_names for n in names if n is not None]
                print(f"  mesh_names(active)={active}")

def dump_joint_state(env, label):
    """Print current joint positions."""
    qpos = env.robot.left_entity.get_qpos()
    print(f"\n[{label}] qpos = {qpos}")

def test_start_state_collision(env, label):
    """Print CuRobo active joint angles and cuda_graph_instance state for both arms."""
    qpos = env.robot.left_entity.get_qpos()
    for side in ["left", "right"]:
        planner = getattr(env.robot, f"{side}_planner")
        joint_indices = [planner.all_joints.index(n) for n in planner.active_joints_name if n in planner.all_joints]
        joint_angles = [round(float(qpos[i]), 6) for i in joint_indices]
        print(f"\n[{label}] {side} active joint angles: {joint_angles}")

        for gen_name in ["motion_gen", "motion_gen_batch"]:
            gen = getattr(planner, gen_name)
            cgi = gen.rollout_fn.cuda_graph_instance
            print(f"  {gen_name}.rollout_fn.cuda_graph_instance = {cgi}")

            # Direct collision check
            from curobo.types.robot import JointState
            start_js = JointState.from_position(
                torch.tensor(joint_angles, dtype=torch.float32).cuda().reshape(1, -1),
                joint_names=planner.active_joints_name,
            )
            valid, status = gen.check_start_state(start_js)
            print(f"  {gen_name}.check_start_state: valid={valid}, status={status}")

if __name__ == "__main__":
    from bench_envs.kitchens.put_spoon_on_plate_ks import put_spoon_on_plate_ks

    TASK_ENV = put_spoon_on_plate_ks()

    for seed in [0, 1, 2]:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        cfg = build_cfg(seed)
        try:
            TASK_ENV.setup_demo(**cfg)
        except Exception as e:
            print(f"setup_demo failed: {e}")
            try:
                TASK_ENV.close_env()
            except:
                pass
            continue

        dump_joint_state(TASK_ENV, f"seed{seed}_after_setup")
        dump_planner_state(TASK_ENV, f"seed{seed}_after_setup")
        test_start_state_collision(TASK_ENV, f"seed{seed}_after_setup")

        print(f"\n--- Running play_once for seed {seed} ---")
        try:
            TASK_ENV.play_once()
            print(f"play_once completed. plan_success={TASK_ENV.plan_success}")
        except Exception as e:
            print(f"play_once failed: {e}")

        dump_joint_state(TASK_ENV, f"seed{seed}_after_play")
        print(f"check_success: {TASK_ENV.check_success()}")
        TASK_ENV.close_env()
        print(f"Closed seed {seed}")
