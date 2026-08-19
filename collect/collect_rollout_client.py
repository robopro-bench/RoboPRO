"""
collect_rollout_client.py — policy (pi05/...) rollout data collection, FORMAT-
IDENTICAL to the CuRobo collector.

Runs in the RoboPRO conda env (client side); connects to a model server
(pi05/openpi/.venv). Episodes are recorded through the SAME env pipeline as
collect_data.py (`take_action` captures frames every save_freq substeps ->
pkl cache -> merge_pkl_to_hdf5_video), so a policy-rollout run dir is
indistinguishable from a CuRobo run dir except for the hdf5 `generator` attr
("pi05" vs "curobo_collision_{aware,unaware}") and `success` (failures are KEPT
here, stamped success=False — failed policy rollouts are collision-rich data):

    <save_path>/<task>/<task_config>/
        data/episodeN.hdf5      identical schema (obs cams/depth/seg/K/E +
                                grounding_mask, joint_action, endpose,
                                object_poses, contact/collision + impulse +
                                pairs, attrs)
        video/episodeN.mp4      countertop cam, same encoder
        scene/episodeN/         scene.npz + objects.json + scene_hash.txt
        _traj_data/             episodeN_init.json (t=0 state, replay input) +
                                episodeN.pkl (policy action sequence,
                                format=policy_actions_v1)
        masking/episodeN.json   grounding stage timeline (same sidecar the
                                CuRobo collectors write; requires a config with
                                data_type.actor_bbox on)
        scene_info.json, seed.txt, instructions/   same sidecars

Use a DEDICATED task_config (e.g. copy office_v2_blind.yml -> office_v2_pi05.yml)
so policy episodes land in their own run dir and don't interleave with CuRobo
episodes' indices.

Usage (via shell script):
    bash policy/pi05/collect_rollout.sh <task> <task_config> <train_config> \
        <model_name> <checkpoint_id> <seed> <server_gpu>[:<client_gpu>]

Env vars:
    COLLECT_NUM         — episodes to collect (default 100)
    COLLECT_START_SEED  — starting seed (default: continue seed.txt, else 0)
    COLLECT_FIXED_SEED  — if set, skip the expert solvability check
    ACTION_NOISE_VAR    — Gaussian action-noise variance (default 0 = faithful policy)
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _env  # noqa: F401
from _env import run_gen_instructions

from envs.utils.create_actor import UnStableError  # noqa: F401

import socket
import json
import time
import traceback
import argparse
import importlib
from typing import Any

import base64
import numpy as np
import yaml

import collect_data as cd
from collect_one_episode import update_scene_info
from export_scene import export_scene
from generate_episode_instructions import generate_episode_descriptions


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            dtype = str(obj.dtype) if obj.dtype not in (np.float32, np.float64, np.int32, np.int64) else str(obj.dtype)
            return {'__numpy_array__': True, 'data': base64.b64encode(obj.tobytes()).decode('ascii'),
                    'dtype': dtype, 'shape': obj.shape}
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


def numpy_to_json(data: Any) -> str:
    return json.dumps(data, cls=NumpyEncoder)


def json_to_numpy(json_str: str) -> Any:
    def object_hook(dct):
        if '__numpy_array__' in dct:
            return np.frombuffer(base64.b64decode(dct['data']), dtype=dct['dtype']).reshape(dct['shape'])
        return dct
    return json.loads(json_str, object_hook=object_hook)


# ── ModelClient (TCP proxy to model server) ────────────────────────────────

class ModelClient:
    _MIRRORED_DEFAULTS = {"observation_window": None}

    def __init__(self, host='localhost', port=9999, timeout=30, **mirrored):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None
        for k, v in self._MIRRORED_DEFAULTS.items():
            object.__setattr__(self, k, v)
        for k, v in mirrored.items():
            object.__setattr__(self, k, v)
        self._connect()

    def _connect(self):
        for attempt in range(1000):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect((self.host, self.port))
                print(f"🔗 Connected to model server at {self.host}:{self.port}")
                return
            except Exception as e:
                if self.sock: self.sock.close()
                print(f"⚠️ Connection attempt {attempt+1} failed: {e}. Retrying in 5s...")
                time.sleep(5)
        raise ConnectionError("Failed to connect to model server after 1000 attempts")

    def _send_recv(self, data):
        json_data = numpy_to_json(data).encode('utf-8')
        self.sock.sendall(len(json_data).to_bytes(4, 'big'))
        self.sock.sendall(json_data)
        return self._recv_response()

    def _recv_response(self):
        size = int.from_bytes(self.sock.recv(4), 'big')
        chunks, received = [], 0
        while received < size:
            chunk = self.sock.recv(min(size - received, 4096))
            if not chunk: raise ConnectionError("Incomplete response")
            chunks.append(chunk); received += len(chunk)
        return json_to_numpy(b''.join(chunks).decode('utf-8'))

    def call(self, func_name=None, obs=None):
        return self._send_recv({"cmd": func_name, "obs": obs})['res']

    def __getattr__(self, name):
        if name.startswith("_") or name in {"host", "port", "timeout", "sock"}:
            raise AttributeError(name)
        def _proxy(*args, **kwargs):
            obs = None if not args else (args[0] if len(args) == 1 else {"_args": list(args)})
            result = self.call(func_name=name, obs=obs)
            if name in ("reset_obsrvationwindows", "reset_model"):
                object.__setattr__(self, "observation_window", None)
            elif name == "update_observation_window":
                object.__setattr__(self, "observation_window", True)
            return result
        return _proxy

    def close(self):
        if self.sock:
            try: self.sock.close()
            except: pass
            finally: self.sock = None; print("🔌 Connection closed")


def eval_function_decorator(policy_name, model_name):
    return getattr(importlib.import_module(policy_name), model_name)


# ── Episode collection (env-pipeline recording, CuRobo-identical output) ────


def _get_camera_wh(camera_type):
    """W,H for the raw video stream (task_config/_camera_config.yml)."""
    import yaml as _yaml
    from pathlib import Path as _Path
    cfg = _Path(os.environ["SIM_ROOT"]) / "task_config" / "_camera_config.yml"
    with open(cfg, "r", encoding="utf-8") as f:
        c = _yaml.load(f.read(), Loader=_yaml.FullLoader)[camera_type]
    return c["w"], c["h"]


def _finish_cmd_video(proc):
    if proc is None:
        return
    try:
        proc.stdin.close()
        proc.wait(timeout=60)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def collect_rollouts(TASK_ENV, args, model, usr_args, collect_num, instruction_type):
    run_dir = Path(args["save_path"])            # already <save_path>/<task>/<config>
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    (run_dir / "_traj_data").mkdir(parents=True, exist_ok=True)

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    fixed_seed = bool(os.environ.get("COLLECT_FIXED_SEED"))
    noise_var = float(os.environ.get("ACTION_NOISE_VAR", usr_args.get("action_noise_var", 0.0)))
    noise_std = float(np.sqrt(noise_var)) if noise_var > 0 else 0.0

    # Resume exactly like the CuRobo collector: seed.txt position == episode idx.
    seed_file = run_dir / "seed.txt"
    seeds_done = [int(s) for s in seed_file.read_text().split()] if seed_file.exists() else []
    ep_idx = len(seeds_done)
    if os.environ.get("COLLECT_START_SEED"):
        seed = int(os.environ["COLLECT_START_SEED"])
    else:
        seed = (max(seeds_done) + 1) if seeds_done else int(usr_args.get("seed", 0))
    if ep_idx:
        print(f"\033[33m[resume] {ep_idx} episodes exist; seeds continue from {seed}\033[0m")

    n_succ = n_fail = 0
    _consec_fail = 0                      # fixed-seed: abort instead of drifting
    while ep_idx < collect_num:
        # ── expert solvability check (same role as the collector's seed search) ─
        if not fixed_seed:
            try:
                chk = dict(args); chk["need_plan"] = True; chk["save_data"] = False
                TASK_ENV.setup_demo(now_ep_num=ep_idx, seed=seed, is_test=True, **chk)
                episode_info = TASK_ENV.play_once()
                if episode_info is None:
                    episode_info = getattr(TASK_ENV, "info", {"info": {}})
                solvable = TASK_ENV.plan_success and TASK_ENV.check_success()
                # intended-contact set discovered by the expert plan (grasp targets,
                # drawer/appliance handles) — carried into the rollout so contact
                # labels have identical semantics to CuRobo-generated data
                intended_names = set(getattr(TASK_ENV, "_intended_contact_names", set()))
                TASK_ENV.close_env()
            except UnStableError:
                TASK_ENV.close_env(); seed += 1; continue
            except Exception:
                traceback.print_exc(); TASK_ENV.close_env(); seed += 1; continue
            if not solvable:
                seed += 1
                continue
        else:
            episode_info = {"info": {}}

        # ── rollout with env-pipeline recording (identical to CuRobo phase 2) ──
        roll = dict(args)
        roll["need_plan"] = False
        roll["save_data"] = True
        roll["render_freq"] = 0
        # policy commands land at ~10 Hz (vs. CuRobo's save_freq-substep cadence);
        # encode the merged episode mp4 to match unless the task_config overrides it.
        roll.setdefault("video_fps", 10)
        print(f"\033[34m[rollout] episode {ep_idx} (seed={seed}) — {policy_name}\033[0m")
        try:
            TASK_ENV.setup_demo(now_ep_num=ep_idx, seed=seed, is_test=fixed_seed, **roll)
            perturbed_instruction = None
            if hasattr(TASK_ENV, "_maybe_apply_language_perturbation"):
                perturbed_instruction = TASK_ENV._maybe_apply_language_perturbation()
            if not fixed_seed and intended_names and hasattr(TASK_ENV, "_intended_contact_names"):
                TASK_ENV._intended_contact_names |= intended_names
            # dense video still comes from merge_pkl_to_hdf5_video; additionally
            # stream an old-style 1-frame-per-COMMAND mp4 (real-speed-looking,
            # like the eval client) unless COLLECT_CMD_VIDEO=0. take_action
            # writes the frames (demo/countertop/head preference).
            TASK_ENV.eval_video_path = None
            _cmd_video = None
            if os.environ.get("COLLECT_CMD_VIDEO", "1") != "0":
                import subprocess as _sp
                _vdir = run_dir / "video"
                _vdir.mkdir(parents=True, exist_ok=True)
                _w, _h = _get_camera_wh(args["camera"]["head_camera_type"])
                _cmd_video = _sp.Popen(
                    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                     "-pixel_format", "rgb24", "-video_size", f"{_w}x{_h}",
                     "-framerate", "10", "-i", "-", "-pix_fmt", "yuv420p",
                     "-vcodec", "libx264", "-crf", "23",
                     str(_vdir / f"episode{ep_idx}_cmd.mp4")],
                    stdin=_sp.PIPE)
                TASK_ENV._set_eval_video_ffmpeg(_cmd_video)
                TASK_ENV.eval_video_path = str(_vdir)

            # t=0 snapshot + scene geometry (replay + offline proximity inputs)
            if hasattr(TASK_ENV, "capture_init_state"):
                try:
                    TASK_ENV.save_init_state(ep_idx, state=TASK_ENV.capture_init_state())
                except Exception as e:
                    print(f"[rollout] init-state save failed: {e}")
            try:
                export_scene(TASK_ENV, str(run_dir / "scene" / f"episode{ep_idx}"))
            except Exception as e:
                print(f"[rollout] export_scene failed: {e}")

            # instruction: the bank instruction (training-prompt distribution) wins
            # when language perturbation set one; generated descriptions are the fallback
            if perturbed_instruction is not None:
                instruction = perturbed_instruction
            else:
                results = generate_episode_descriptions(args["task_name"], [episode_info["info"]], collect_num)
                instruction = np.random.choice(results[0][instruction_type])
                TASK_ENV.set_instruction(instruction=instruction)

            # action logging (+ optional exploration noise) around take_action
            action_log = []
            _orig_ta = TASK_ENV.take_action

            def _logged_ta(action, *a, **kw):
                act = np.asarray(action, dtype=np.float32)
                if noise_std > 0:
                    act = act.copy()
                    act[[*range(6), *range(7, 13)]] += np.random.normal(
                        0, noise_std, size=12).astype(np.float32)
                action_log.append(act.copy())
                return _orig_ta(act, *a, **kw)
            TASK_ENV.take_action = _logged_ta

            # step limit: normally loaded only in eval_mode; we keep eval_mode
            # False (seen/training textures) so load the per-task limit here.
            if getattr(TASK_ENV, "step_lim", None) is None:
                try:
                    lim_path = os.path.join(os.environ["BENCH_ROOT"],
                                            "bench_task_config", "_bench_eval_step_limit.yml")
                    with open(lim_path, "r") as f:
                        TASK_ENV.step_lim = yaml.safe_load(f).get(args["task_name"], 1000)
                except Exception:
                    TASK_ENV.step_lim = 1000

            model.reset_model()
            while TASK_ENV.take_action_cnt < TASK_ENV.step_lim and not TASK_ENV.eval_success:
                observation = TASK_ENV.get_obs()
                eval_func(TASK_ENV, model, observation)
            TASK_ENV.take_action = _orig_ta

            success = bool(TASK_ENV.eval_success or TASK_ENV.check_success())

            # sidecar info (same block as the CuRobo collectors)
            info = dict(episode_info) if isinstance(episode_info, dict) else {}
            if hasattr(TASK_ENV, "get_actor_id_map"):
                info["actor_id_map"] = TASK_ENV.get_actor_id_map()
            if hasattr(TASK_ENV, "get_role_names"):
                info["role_names"] = TASK_ENV.get_role_names()
            if hasattr(TASK_ENV, "get_object_poses"):
                info["object_pose_ids"] = TASK_ENV.get_object_poses()[0]
            if getattr(TASK_ENV, "enable_collision_metrics", False) and hasattr(TASK_ENV, "get_collision_metrics"):
                info["collision_metrics"] = TASK_ENV.get_collision_metrics()
            info["seed"] = seed
            info["instruction"] = str(instruction)

            cache_dir = run_dir / ".cache" / f"episode{ep_idx}"
            has_frames = cache_dir.is_dir() and any(
                fn.endswith(".pkl") for fn in os.listdir(cache_dir))

            _finish_cmd_video(_cmd_video); _cmd_video = None
            TASK_ENV.close_env(clear_cache=True)
            if has_frames:
                TASK_ENV.merge_pkl_to_hdf5_video()
            TASK_ENV.remove_data_cache()

            if not has_frames:
                print(f"\033[93m[rollout] episode {ep_idx}: no frames — retry\033[0m")
                # COLLECT_FIXED_SEED contract: the seed NEVER moves. Drifting here
                # walked seeds out of their allotted blocks (observed 80000->81868)
                # and can silently collide with a neighbouring density's range.
                if not fixed_seed:
                    seed += 1
                else:
                    _consec_fail += 1
                    if _consec_fail >= 3:
                        raise RuntimeError(f"fixed seed {seed}: 3 consecutive failures — aborting")
                continue

            hdf5_path = str(run_dir / "data" / f"episode{ep_idx}.hdf5")
            cd._stamp_provenance_attrs(hdf5_path, args, seed=seed, success=success,
                                       timestep=getattr(TASK_ENV, "timestep", None),
                                       generator=str(policy_name))

            # policy action sequence -> _traj_data (presence-parity with the
            # CuRobo plan pkl; enables future policy-replay tooling)
            import pickle
            with open(run_dir / "_traj_data" / f"episode{ep_idx}.pkl", "wb") as f:
                pickle.dump({"format": "policy_actions_v1",
                             "policy_name": str(policy_name),
                             "actions": action_log,
                             "left_joint_path": [], "right_joint_path": []}, f)

            update_scene_info(run_dir, ep_idx, info)
            # grounding: masking/episode{idx}.json sidecar + target/bin masks baked into
            # the HDF5 — the same first-class per-episode output the CuRobo collectors
            # write, so a policy run dir stays format-identical. Must run AFTER
            # update_scene_info (the resolver reads role_names/actor_id_map from it).
            # Best-effort: a resolver failure must not lose an otherwise-good episode.
            if getattr(cd, "finalize_grounding", None) is not None:
                try:
                    cd.finalize_grounding(args["save_path"], ep_idx,
                                          obj_pad=args.get("table_obj_pad"))
                except (Exception, SystemExit) as _me:  # noqa: BLE001
                    # masking_resolve raises SystemExit when the hdf5 lacks an
                    # actor_bbox group (bench configs don't record it); a bare
                    # `except Exception` let that kill the process between the
                    # episode save and the seed.txt write. Same fix as the GB10
                    # collection tree that produced the HF training data.
                    print(f"\033[93m[rollout] grounding skipped (episode {ep_idx}): {_me}\033[0m")
            with open(seed_file, "a", encoding="utf-8") as f:
                f.write(f"{seed} ")

            n_succ += int(success); n_fail += int(not success)
            print(f"\033[92m[rollout] episode {ep_idx} saved (success={success})\033[0m  "
                  f"SR {n_succ}/{ep_idx + 1}")
            ep_idx += 1
            _consec_fail = 0
            if not fixed_seed:
                seed += 1            # fixed mode: same scene again (N rollouts per seed)
        except UnStableError:
            _finish_cmd_video(locals().get('_cmd_video'))
            TASK_ENV.close_env()
            if not fixed_seed:
                seed += 1
            else:
                _consec_fail += 1
                if _consec_fail >= 3:
                    raise RuntimeError(f"fixed seed {seed}: 3 consecutive UnStable failures — aborting")
        except (BrokenPipeError, ConnectionError) as e:
            # dead policy server — retrying seeds is pointless; abort loudly
            print(f"\033[91m[rollout] model server connection lost: {e} — aborting\033[0m")
            _finish_cmd_video(locals().get('_cmd_video'))
            for fn in (TASK_ENV.remove_data_cache, TASK_ENV.close_env):
                try:
                    fn()
                except Exception:
                    pass
            raise SystemExit(3)
        except Exception:
            traceback.print_exc()
            _finish_cmd_video(locals().get('_cmd_video'))
            for fn in (TASK_ENV.remove_data_cache, TASK_ENV.close_env):
                try:
                    fn()
                except Exception:
                    pass
            if not fixed_seed:
                seed += 1
            else:
                _consec_fail += 1
                if _consec_fail >= 3:
                    raise RuntimeError(f"fixed seed {seed}: 3 consecutive failures — aborting")

    # end-of-run language instructions, same as the CuRobo collector
    run_gen_instructions(args['task_name'], args['task_config'], args['language_num'])
    print(f"\nDone. {ep_idx} episodes -> {run_dir/'data'}  "
          f"(success {n_succ} / fail {n_fail})")


# ── Entry point ──────────────────────────────────────────────────────────────

def main(usr_args):
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args.get("instruction_type", "seen")
    port = usr_args["port"]
    collect_num = int(os.environ.get("COLLECT_NUM", usr_args.get("collect_num", 100)))

    # identical arg/scene/camera/save_path resolution to the CuRobo collector
    TASK_ENV, args = cd.build_task_and_args(task_name, task_config)
    args["policy_name"] = policy_name
    args["ckpt_setting"] = usr_args.get("ckpt_setting")

    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    model_mirrored = {}
    if "pi0_step" in usr_args:
        model_mirrored["pi0_step"] = usr_args["pi0_step"]
    model = ModelClient(port=port, **model_mirrored)

    collect_rollouts(TASK_ENV, args, model, usr_args, collect_num, instruction_type)


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    cli = parser.parse_args()

    with open(cli.config, "r") as f:
        config = yaml.safe_load(f)
    config["port"] = cli.port

    if cli.overrides:
        pairs = cli.overrides
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            try:
                value = eval(pairs[i + 1])
            except Exception:
                value = pairs[i + 1]
            config[key] = value
    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    main(parse_args_and_config())
