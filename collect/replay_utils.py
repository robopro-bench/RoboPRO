"""Deterministic rebuild-and-replay helpers.

The primitive this module supports is: *reach a mid-episode state exactly, more
than once*. The obvious way — snapshot the physics state and restore it — does
not work, and the failure is silent. `PhysxSystem.pack()/unpack()` restores every
pose, quaternion, joint angle and joint velocity bit-exactly, but NOT the PhysX
contact/warm-start cache, so the same commanded actions replayed from a restored
state diverge and, measured over 18 states on put_milktea_next_to_laptop,
**flip the success label 38.9% of the time** — systematically toward failure,
because a restored branch tracks worse than a naturally-arrived one.

Rebuilding instead is exact. `close_env(clear_cache=True)` + `setup_demo(seed=...)`
resets the solver cache, and replaying the recorded action rows from step 0
reproduces the state bit-exactly (max|diff| 0.000e+00 across the full state
fingerprint, 0% label flips on the same states that flipped under restore).
Verified across separate processes as well, and with the policy in the loop —
ray-traced rendering feeding live inference is also bit-deterministic.

So the contract is:
    deterministic given  (scene seed, action rows from step 0)
    NOT deterministic given  (mid-episode observable state, action)

Prefix replay needs no policy — it is a replay of stored rows — which is why a
branch point is reproducible even though the branch itself involves inference.

See collect/verify_replay_determinism.py for the harness that measures all of this.
"""
import importlib
import os

import numpy as np
import yaml

from envs import CONFIGS_PATH, resolve_embodiment_path


def build_args(task_name, task_config):
    """Assemble the task kwargs exactly as the eval/collection clients do."""
    from script.task_loader import bench_config_path
    cfg_path = bench_config_path(task_config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = None

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        emb = yaml.load(f.read(), Loader=yaml.FullLoader)
    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        cam = yaml.load(f.read(), Loader=yaml.FullLoader)

    hct = args["camera"]["head_camera_type"]
    args["head_camera_h"], args["head_camera_w"] = cam[hct]["h"], cam[hct]["w"]

    et = args.get("embodiment")

    def robot_file(t):
        p = resolve_embodiment_path(emb[t]["file_path"])
        if p is None:
            raise ValueError(f"no embodiment file for {t}")
        return p

    def emb_conf(p):
        with open(os.path.join(p, "config.yml"), "r", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    if len(et) == 1:
        args["left_robot_file"] = args["right_robot_file"] = robot_file(et[0])
        args["dual_arm_embodied"] = True
    elif len(et) == 3:
        args["left_robot_file"], args["right_robot_file"] = robot_file(et[0]), robot_file(et[1])
        args["embodiment_dis"] = et[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = emb_conf(args["left_robot_file"])
    args["right_embodiment_config"] = emb_conf(args["right_robot_file"])
    args["render_freq"] = 0
    args["eval_mode"] = False      # 'seen' textures, matching the collection path
    args["save_data"] = False
    args["need_plan"] = False
    return args


def make_task(task_name):
    """Instantiate a task by name from bench_envs."""
    from script.task_loader import load_task
    return load_task(task_name)


def task_environment(task_name, bench_root=None):
    """office / study / kitchenl / kitchens, from the bench_envs layout."""
    from pathlib import Path
    root = Path(bench_root or os.environ["BENCH_ROOT"]) / "bench_envs"
    for env in ("office", "study", "kitchenl", "kitchens"):
        if (root / env / f"{task_name}.py").exists():
            return env
    return None


def state_fingerprint(task):
    """Deterministic float64 summary of the full sim state.

    Every actor pose (p, q) and every articulation (root pose, qpos, qvel).
    Two states with an equal fingerprint are the same scene configuration --
    this is the arbiter for whether a replay reproduced a trajectory.
    """
    parts = []
    for a in task.scene.get_all_actors():
        p = a.get_pose()
        parts.append(np.concatenate([np.asarray(p.p), np.asarray(p.q)]))
    for art in task.scene.get_all_articulations():
        rp = art.get_root_pose()
        parts.append(np.concatenate([
            np.asarray(rp.p), np.asarray(rp.q),
            np.asarray(art.get_qpos()), np.asarray(art.get_qvel())]))
    return np.concatenate(parts).astype(np.float64) if parts else np.zeros(0)


def replay_actions(task, rows):
    """Replay stored action rows. No policy, no observations consumed."""
    for row in rows:
        task.take_action(np.asarray(row, dtype=np.float64))


def rebuild(task, task_name, args, seed, step_lim=None, instruction=None):
    """Fresh scene from `seed`. Resets the PhysX solver cache -- the whole point.

    Pass the previous task object to reuse it (close_env + setup_demo is enough;
    a new process is NOT required), or None to construct one.
    """
    if task is None:
        task = make_task(task_name)
    else:
        task.close_env(clear_cache=True)
    task.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
    if step_lim is not None:
        task.step_lim = step_lim
    task.eval_video_path = None
    if instruction is not None:
        task.set_instruction(instruction=instruction)
    return task
