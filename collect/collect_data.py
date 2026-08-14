import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _env  # noqa: F401  — SIM_ROOT / BENCH_ROOT / DATA_ROOT + sys.path
from _env import resolve_save_path, run_gen_instructions

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import traceback
import os
import time
import h5py
from argparse import ArgumentParser

from export_scene import export_scene
from masking_resolve import finalize_grounding

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
bench_root = Path(os.environ["BENCH_ROOT"])


def class_decorator(task_name):
    from script.task_loader import load_task
    return load_task(task_name)

def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def build_task_and_args(task_name, task_config):
    """Load the task env + fully-resolved args dict (config yml + embodiment),
    with args["save_path"] rewritten to the run dir. Shared by the stock
    multi-episode entry point (main/run) and the single-seed runner
    (collect_one_episode.py) used for parallel-GPU collection."""

    task = class_decorator(task_name)

    from script.task_loader import bench_config_path
    task_config_path = bench_config_path(task_config)

    with open(task_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = resolve_embodiment_path(_embodiment_types[embodiment_type]["file_path"])
        if robot_file is None:
            raise "missing embodiment files"
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "number of embodiment config parameters should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["save_path"] = os.path.join(
        resolve_save_path(args["save_path"]), str(args["task_name"]), args["task_config"])
    return task, args


def main(task_name=None, task_config=None):
    task, args = build_task_and_args(task_name, task_config)
    run(task, args)


def _stamp_provenance_attrs(hdf5_path, args, seed=None, success=None, timestep=None,
                            generator=None):
    """Make each episode self-describing: which planner/metric regime (and seed)
    produced it.

    planner_exclude_obstacles attr: -1 = key omitted (legacy coupling to
    enable_collision_metrics), 0 = planner obstacle-AWARE, 1 = planner BLIND.
    planner_blind_to_obstacles is the resolved blindness actually in effect.
    """
    if not os.path.exists(hdf5_path):
        return
    ecm = bool(args.get("enable_collision_metrics", False))
    peo = args.get("planner_exclude_obstacles", None)
    blind = bool(peo) if peo is not None else ecm
    with h5py.File(hdf5_path, "a") as f:
        # human-readable producer label ("what made this data"). CuRobo datasets
        # derive it from the resolved planner regime; policy-rollout datasets pass
        # generator=<policy name> (e.g. "pi05") explicitly.
        f.attrs["generator"] = generator if generator is not None else (
            "curobo_collision_unaware" if blind else "curobo_collision_aware")
        f.attrs["enable_collision_metrics"] = ecm
        f.attrs["planner_exclude_obstacles"] = -1 if peo is None else int(bool(peo))
        f.attrs["planner_blind_to_obstacles"] = blind
        if seed is not None:
            f.attrs["seed"] = int(seed)
        if success is not None:
            # task check_success() at episode end. NOTE the CuRobo collector only
            # KEEPS successful episodes (failures are deleted / never claim a
            # slot), so kept files always read True here; policy-rollout datasets
            # keep failures too and record success per episode the same way.
            f.attrs["success"] = bool(success)
        if timestep is not None:
            # physics dt (s): converts recorded impulses (N*s) to avg force, F = J/dt
            f.attrs["physics_timestep"] = float(timestep)
        f.attrs["task_name"] = str(args.get("task_name", ""))
        f.attrs["task_config"] = str(args.get("task_config", ""))


def _banner(text, char="─", width=74):
    line = char * width
    print(f"\n\033[96m{line}\n{text}\n{line}\033[0m")


def run(TASK_ENV, args):
    epid, suc_num, fail_num, seed_list = 0, 0, 0, []

    # start_seed: base offset for the seed search so parallel shards (e.g. one
    # collector per GPU, each writing to its own task_config dir) explore
    # DISJOINT seed ranges instead of re-collecting the same scenes from seed 0.
    # Config key `start_seed` (default 0), overridable via COLLECT_START_SEED.
    # Resume (existing seed.txt) still wins over this.
    start_seed = int(os.getenv("COLLECT_START_SEED", args.get("start_seed", 0) or 0))
    epid = start_seed
    if start_seed:
        print(f"\033[95m[start_seed] seed search begins at {start_seed}\033[0m")

    # Debug mode: instead of iterating over seeds (reopening Sapien),
    # hold the viewer open on the first failure so targets/scene can be inspected.
    # Enable with:  RT_DEBUG_STUCK=1 python -u collect/collect_data.py ...
    debug_stuck = os.getenv("RT_DEBUG_STUCK") == "1"

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")
    if debug_stuck:
        print("\033[93m[RT_DEBUG_STUCK=1] will hold viewer on failure; no seed iteration\033[0m")

    # =========== Collect Seed ===========
    os.makedirs(args["save_path"], exist_ok=True)

    if not args["use_seed"]:
        print("\033[93m" + "[Start Seed and Pre Motion Data Collection]" + "\033[0m")
        args["need_plan"] = True

        save_seed_file = args.get("save_seed", True)

        if save_seed_file and os.path.exists(os.path.join(args["save_path"], "seed.txt")):
            with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
                seed_list = file.read().split()
                if len(seed_list) != 0:
                    seed_list = [int(i) for i in seed_list]
                    suc_num = len(seed_list)
                    epid = max(seed_list) + 1
            print(f"Exist seed file, Start from: {epid} / {suc_num}")

        while suc_num < args["episode_num"]:
            print(f"\n\033[90m── seed attempt {epid} · successes {suc_num}/{args['episode_num']} "
                  + "─" * 40 + "\033[0m")
            try:
                TASK_ENV.setup_demo(now_ep_num=suc_num, seed=epid, **args)
                if hasattr(TASK_ENV, "_maybe_apply_language_perturbation"):
                    TASK_ENV._maybe_apply_language_perturbation()
                # t=0 scene snapshot (before play_once mutates it) for later replay
                _init_state = TASK_ENV.capture_init_state() \
                    if hasattr(TASK_ENV, "capture_init_state") else None
                TASK_ENV.play_once()

                if TASK_ENV.plan_success and TASK_ENV.check_success():
                    print(f"simulate data episode {suc_num} success! (seed = {epid})")
                    seed_list.append(epid)
                    TASK_ENV.save_traj_data(suc_num)
                    if _init_state is not None:
                        TASK_ENV.save_init_state(suc_num, state=_init_state)
                    suc_num += 1
                else:
                    print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                    fail_num += 1
                    if debug_stuck:
                        reason = f"plan_success={TASK_ENV.plan_success} check_success={TASK_ENV.check_success()} seed={epid}"
                        TASK_ENV.debug_hold_viewer(reason=reason)
                        return

                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
            except UnStableError as e:
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                if debug_stuck:
                    TASK_ENV.debug_hold_viewer(reason=f"UnStableError: {e} (seed={epid})")
                    return
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(0.3)
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(stack_trace)
                print(" -------------")
                fail_num += 1
                if debug_stuck:
                    try:
                        TASK_ENV.debug_hold_viewer(reason=f"Exception: {e} (seed={epid})")
                    except Exception:
                        pass
                    return
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(1)

            epid += 1

            if save_seed_file:
                with open(os.path.join(args["save_path"], "seed.txt"), "w") as file:
                    for sed in seed_list:
                        file.write("%s " % sed)

        print(f"\nComplete simulation, failed \033[91m{fail_num}\033[0m times / {epid} tries \n")
    else:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]

    # =========== Collect Data ===========

    if args["collect_data"]:
        print("\033[93m" + "[Start Data Collection]" + "\033[0m")

        args["need_plan"] = False
        args["render_freq"] = 0
        args["save_data"] = True

        clear_cache_freq = args["clear_cache_freq"]

        st_idx = 0

        def exist_hdf5(idx):
            file_path = os.path.join(args["save_path"], 'data', f'episode{idx}.hdf5')
            return os.path.exists(file_path)

        while exist_hdf5(st_idx):
            st_idx += 1

        attempted_num, deleted_num, error_num = 0, 0, 0

        for episode_idx in range(st_idx, args["episode_num"]):
            if exist_hdf5(episode_idx):
                continue
            _banner(f"COLLECT episode {episode_idx} · seed {seed_list[episode_idx]} · "
                    f"{args['task_name']} / {args['task_config']}")
            attempted_num += 1

            try:
                TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed_list[episode_idx], **args)
                if hasattr(TASK_ENV, "_maybe_apply_language_perturbation"):
                    TASK_ENV._maybe_apply_language_perturbation()

                # export static scene geometry for offline link/sphere proximity
                try:
                    export_scene(TASK_ENV, os.path.join(args["save_path"], "scene", f"episode{episode_idx}"))
                except Exception as _e:
                    print(f"\033[93mexport_scene failed (episode {episode_idx}): {_e}\033[0m")

                traj_data = TASK_ENV.load_tran_data(episode_idx)
                args["left_joint_path"] = traj_data["left_joint_path"]
                args["right_joint_path"] = traj_data["right_joint_path"]
                TASK_ENV.set_path_lst(args)

                info_file_path = os.path.join(args["save_path"], "scene_info.json")

                if not os.path.exists(info_file_path):
                    with open(info_file_path, "w", encoding="utf-8") as file:
                        json.dump({}, file, ensure_ascii=False)

                with open(info_file_path, "r", encoding="utf-8") as file:
                    info_db = json.load(file)

                info = TASK_ENV.play_once()
                if info is None:
                    info = getattr(TASK_ENV, "info", None) or {}
                # grounding sidecar: seg-id -> name map + task-designated role names
                # (must be captured while the scene is still alive)
                if hasattr(TASK_ENV, "get_actor_id_map"):
                    info["actor_id_map"] = TASK_ENV.get_actor_id_map()
                if hasattr(TASK_ENV, "get_role_names"):
                    info["role_names"] = TASK_ENV.get_role_names()
                if hasattr(TASK_ENV, "get_object_poses"):
                    info["object_pose_ids"] = TASK_ENV.get_object_poses()[0]
                info["seed"] = seed_list[episode_idx]
                if getattr(TASK_ENV, "enable_collision_metrics", False) and hasattr(TASK_ENV, "get_collision_metrics"):
                    info["collision_metrics"] = TASK_ENV.get_collision_metrics()

                # measured while the scene is still alive (used for the keep decision
                # below — the stock code re-queried after close_env)
                success = bool(TASK_ENV.check_success())

                # only episodes with saved frames become an hdf5 (a failed plan can
                # produce no executable motion -> nothing to save, not a crash)
                cache_dir = os.path.join(args["save_path"], ".cache", f"episode{episode_idx}")
                has_frames = os.path.isdir(cache_dir) and any(
                    fn.endswith(".pkl") for fn in os.listdir(cache_dir))

                print(f"\033[96mepisode {episode_idx}: success={success} (frames={has_frames})\033[0m")
                info_db[f"episode_{episode_idx}"] = info

                with open(info_file_path, "w", encoding="utf-8") as file:
                    json.dump(info_db, file, ensure_ascii=False, indent=4, default=str)

                TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
                if has_frames:
                    TASK_ENV.merge_pkl_to_hdf5_video()
                    _stamp_provenance_attrs(
                        os.path.join(args["save_path"], "data", f"episode{episode_idx}.hdf5"),
                        args, seed=seed_list[episode_idx], success=success,
                        timestep=getattr(TASK_ENV, 'timestep', None))
                else:
                    print(f"\033[93mepisode {episode_idx}: no frames saved "
                          f"(planner produced no executable motion) — no hdf5 written\033[0m")
                TASK_ENV.remove_data_cache()

                if not success:
                    deleted_num += 1
                    print(f"\033[91mCollect Error on episode {episode_idx} "
                          f"(seed={seed_list[episode_idx]}), removing files\033[0m")
                    for ext_path in [
                        os.path.join(args["save_path"], "data", f"episode{episode_idx}.hdf5"),
                        os.path.join(args["save_path"], "video", f"episode{episode_idx}.mp4"),
                    ]:
                        if os.path.exists(ext_path):
                            os.remove(ext_path)
                    continue
                # episode kept -> derive grounding from the HDF5: masking sidecar +
                # baked target/bin masks in the HDF5 (all frames, all cameras).
                # Grounding needs the per-frame actor_bbox group, which is only
                # written when data_type.actor_bbox is on (default False, and only
                # a handful of configs enable it). Without it masking_resolve
                # raises SystemExit -- which, being a BaseException, escapes the
                # except-Exception guards below and kills the whole run after the
                # first episode. Skip the step instead when the flag is off.
                if (args.get("data_type") or {}).get("actor_bbox", False):
                    try:
                        finalize_grounding(args["save_path"], episode_idx,
                                           obj_pad=args.get("table_obj_pad"))
                    except Exception as _me:  # noqa: BLE001
                        print(f"\033[93mgrounding failed (episode {episode_idx}): {_me}\033[0m")
            except Exception as e:
                # one bad episode (CuRobo/mesh crash, missing frames, etc.) must NOT
                # kill the whole run — log it, record it, clean up, move on.
                print(f"\033[91m[episode {episode_idx}] crashed during collection "
                      f"(seed={seed_list[episode_idx]}): {e}\033[0m")
                print(traceback.format_exc())
                error_num += 1
                for _cleanup in (TASK_ENV.remove_data_cache, TASK_ENV.close_env):
                    try:
                        _cleanup()
                    except Exception:
                        pass
                for ext_path in [
                    os.path.join(args["save_path"], "data", f"episode{episode_idx}.hdf5"),
                    os.path.join(args["save_path"], "video", f"episode{episode_idx}.mp4"),
                ]:
                    if os.path.exists(ext_path):
                        try:
                            os.remove(ext_path)
                        except Exception:
                            pass
                continue

        run_gen_instructions(args['task_name'], args['task_config'], args['language_num'])

        _banner(f"COLLECTION SUMMARY · {args['task_name']} / {args['task_config']}")
        print(f"  {'episodes attempted':26s} {attempted_num}")
        print(f"  {'kept in dataset':26s} {attempted_num - deleted_num - error_num}")
        if deleted_num:
            print(f"  {'removed (task failed)':26s} {deleted_num}")
        if error_num:
            print(f"  {'errors (skipped)':26s} {error_num}")


if __name__ == "__main__":
    pass  # Skip Sapien_TEST — ray tracing uses too much VRAM for CuRobo

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser = parser.parse_args()
    task_name = parser.task_name
    task_config = parser.task_config

    main(task_name=task_name, task_config=task_config)
