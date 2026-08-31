import sys
import os
import subprocess

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from script.bench_script.setup_paths import setup_paths
setup_paths()
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *
from script.eval_seeds import resolve_eval_seeds, resolve_test_num, resolve_instruction_bank, resolve_expert_check
from script.eval_envs import resolve_eval_envs, env_pack_label, eval_setup_kwargs


import sys
import os
import subprocess
import socket
import json
import threading
import time
import random
import traceback
import yaml
from datetime import datetime
import importlib
import argparse
from pathlib import Path
from collections import deque

import numpy as np
import json
from typing import Any

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

import numpy as np
import json
from typing import Any
import base64

class NumpyEncoder(json.JSONEncoder):
    """Enhanced json encoder for numpy types with array reconstruction info"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.dtype == np.float32:
                dtype = 'float32'
            elif obj.dtype == np.float64:
                dtype = 'float64'
            elif obj.dtype == np.int32:
                dtype = 'int32'
            elif obj.dtype == np.int64:
                dtype = 'int64'
            else:
                dtype = str(obj.dtype)
            
            return {
                '__numpy_array__': True,
                'data': base64.b64encode(obj.tobytes()).decode('ascii'),
                'dtype': dtype,
                'shape': obj.shape
            }
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def numpy_to_json(data: Any) -> str:
    """Convert numpy-containing data to JSON string with reconstruction info"""
    return json.dumps(data, cls=NumpyEncoder)

def json_to_numpy(json_str: str) -> Any:
    """Convert JSON string back to Python objects with numpy arrays"""
    def object_hook(dct):
        if '__numpy_array__' in dct:
            data = base64.b64decode(dct['data'])
            return np.frombuffer(data, dtype=dct['dtype']).reshape(dct['shape'])
        return dct
    
    return json.loads(json_str, object_hook=object_hook)

def class_decorator(task_name):
    envs_module = None
    if os.getenv("ROBOTWIN_BENCH_TASK") == "bench":
        for mod_path in [f"bench_envs.{task_name}", f"bench_envs.study.{task_name}", f"bench_envs.office.{task_name}", f"bench_envs.kitchenl.{task_name}", f"bench_envs.kitchens.{task_name}"]:
            try:
                envs_module = importlib.import_module(mod_path)
                break
            except ModuleNotFoundError:
                continue
    if envs_module is None:
        envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name, conda_env=None):
    # conda_env is abandoned
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e


def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args

class ModelClient:
    # Attributes that deploy_policy.py reads directly (not as methods).
    # Mirrored on the client side because the server's RPC dispatch only
    # supports method calls, and `model.x is None` cannot be satisfied by a
    # callable proxy.
    _MIRRORED_DEFAULTS = {"observation_window": None}

    def __init__(self, host='localhost', port=9999, timeout=30, **mirrored):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None
        # Local mirrors. `mirrored` kwargs (e.g. pi0_step) are forwarded to
        # client-side attributes so deploy_policy can read them without RPC.
        for k, v in self._MIRRORED_DEFAULTS.items():
            object.__setattr__(self, k, v)
        for k, v in mirrored.items():
            object.__setattr__(self, k, v)
        self._connect()

    def _connect(self):
        attempts = 0
        max_attempts = 1000
        retry_delay = 5
        
        while attempts < max_attempts:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect((self.host, self.port))
                print(f"🔗 Connected to model server at {self.host}:{self.port}")
                return
            except Exception as e:
                attempts += 1
                if self.sock:
                    self.sock.close()
                if attempts < max_attempts:
                    print(f"⚠️ Connection attempt {attempts} failed: {str(e)}")
                    print(f"🔄 Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise ConnectionError(
                        f"Failed to connect to server after {max_attempts} attempts: {str(e)}"
                    )

    def _send_recv(self, data):
        """Send request and receive response with numpy array support"""
        try:
            # Serialize with numpy support
            json_data = numpy_to_json(data).encode('utf-8')
            
            # Send data length and data
            self.sock.sendall(len(json_data).to_bytes(4, 'big'))
            self.sock.sendall(json_data)
            
            # Receive and deserialize response
            response = self._recv_response()
            return response
            
        except Exception as e:
            self.close()
            raise ConnectionError(f"Communication error: {str(e)}")

    def _recv_response(self):
        """Receive response with numpy array reconstruction"""
        # Read response length
        len_data = self.sock.recv(4)
        if not len_data:
            raise ConnectionError("Connection closed by server")
        
        size = int.from_bytes(len_data, 'big')
        
        # Read complete response
        chunks = []
        received = 0
        while received < size:
            chunk = self.sock.recv(min(size - received, 4096))
            if not chunk:
                raise ConnectionError("Incomplete response received")
            chunks.append(chunk)
            received += len(chunk)
        
        # Deserialize with numpy reconstruction
        return json_to_numpy(b''.join(chunks).decode('utf-8'))

    def call(self, func_name=None, obs=None):
        response = self._send_recv({"cmd": func_name, "obs": obs})
        if 'res' not in response:
            # Server caught an exception and returned {"error", "traceback"}
            # (see policy_model_server.py). Surface it instead of masking it
            # behind a KeyError: 'res'.
            err = response.get('error', '<no error field in response>')
            tb = response.get('traceback', '')
            raise RuntimeError(
                f"Server error during RPC '{func_name}':\n{err}\n--- server traceback ---\n{tb}")
        return response['res']

    def __getattr__(self, name):
        # __getattr__ only fires for missing attributes, so mirrored fields
        # set via object.__setattr__ are untouched.
        if name.startswith("_") or name in {"host", "port", "timeout", "sock"}:
            raise AttributeError(name)

        def _proxy(*args, **kwargs):
            if kwargs:
                raise TypeError("ModelClient RPC does not support kwargs")
            if len(args) == 0:
                obs = None
            elif len(args) == 1:
                obs = args[0]
            else:
                # Multi-arg → wrap in dict; server unpacks with `_args` key.
                obs = {"_args": list(args)}
            result = self.call(func_name=name, obs=obs)
            # Update client-side mirrors after methods that change state.
            if name == "reset_obsrvationwindows" or name == "reset_model":
                object.__setattr__(self, "observation_window", None)
            elif name == "update_observation_window":
                # Sentinel: any non-None value satisfies `is None` checks.
                object.__setattr__(self, "observation_window", True)
            return result
        return _proxy

    def close(self):
        """Close the connection"""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            finally:
                self.sock = None
                print("🔌 Connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    port = usr_args["port"]
    save_dir = None
    video_save_dir = None
    video_size = None

    policy_conda_env = usr_args.get("policy_conda_env", None)

    get_model = eval_function_decorator(policy_name, "get_model", conda_env=policy_conda_env)

    if os.getenv("ROBOTWIN_BENCH_TASK") == "bench":
        with open(f"{os.getenv('BENCH_ROOT')}/bench_task_config/{task_config}.yml", "r", encoding="utf-8") as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)
    else:
        with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

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
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)
    episode_log_path = os.path.join(save_dir, "_episodes.jsonl")
    print(f"Per-episode stats will be appended to {episode_log_path}")

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
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

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    env_list = resolve_eval_envs(task_name, task_config, usr_args, task_env=TASK_ENV)
    if env_list is not None:
        seed_list = None
        test_num = resolve_test_num(usr_args, env_list, default=100)
        print(f"\033[96mUsing {len(env_list)} eval env packs "
              f"(evaluating first {test_num}); ignoring eval_seeds\033[0m")
    else:
        seed_list = resolve_eval_seeds(task_name, task_config, usr_args)
        test_num = resolve_test_num(usr_args, seed_list, default=100)
        if seed_list is not None:
            print(f"\033[96mUsing {len(seed_list)} precollected eval seeds "
                  f"(evaluating first {test_num})\033[0m")
        else:
            print(f"\033[96mNo eval seed file; scanning from st_seed={st_seed} "
                  f"(test_num={test_num})\033[0m")
    expert_check = resolve_expert_check(usr_args, default=True)
    topk = 1

    # model = get_model(usr_args)
    # Mirror config-derived attributes that deploy_policy reads directly.
    model_mirrored = {}
    if "pi0_step" in usr_args:
        model_mirrored["pi0_step"] = usr_args["pi0_step"]
    model = ModelClient(port=port, **model_mirrored)
    st_seed, suc_num, used_seeds = eval_policy(
        task_name,
        TASK_ENV,
        args,
        model,
        st_seed,
        test_num=test_num,
        video_size=video_size,
        instruction_type=instruction_type,
        policy_conda_env=policy_conda_env,
        episode_log_path=episode_log_path,
        seed_list=seed_list,
        env_list=env_list,
        expert_check=expert_check,
    )
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        if env_list is not None:
            file.write(f"Eval envs: {' '.join(map(str, used_seeds))}\n\n")
        else:
            file.write(f"Eval seeds: {' '.join(map(str, used_seeds))}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                policy_conda_env=None,
                episode_log_path=None,
                seed_list=None,
                env_list=None,
                expert_check=True):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    # Honor the caller's expert_check setting, but always skip the live check for
    # precollected seeds and frozen env packs (already expert-validated).
    expert_check = expert_check and (seed_list is None) and (env_list is None)
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    # Per-episode stats are appended to episode_log_path after every episode.
    # Collision metrics are enabled on the policy rollout only — never on the
    # expert pre-check, because enable_collision_metrics also switches
    # update_world to exclude clutter obstacles from the planner, which would
    # change which seeds pass the expert check.
    collision_metrics_enabled = os.environ.get("EVAL_COLLISION_METRICS", "1") != "0"
    collision_metrics_active = False

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []
    seed_idx = 0
    env_idx = 0

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval", conda_env=policy_conda_env)

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    while succ_seed < test_num:
        eval_env = None
        if env_list is not None:
            if env_idx >= len(env_list):
                break
            eval_env = env_list[env_idx]
            env_idx += 1
            now_seed = int((eval_env.get("meta") or {}).get("episode_index", env_idx - 1))
        elif seed_list is not None:
            if seed_idx >= len(seed_list):
                break
            now_seed = seed_list[seed_idx]
            seed_idx += 1

        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(**eval_setup_kwargs(now_id, now_seed, args, eval_env))
                episode_info = TASK_ENV.play_once()
                if episode_info is None:
                    # Bench tasks don't `return self.info` from play_once;
                    # use the env's info dict directly.
                    episode_info = getattr(TASK_ENV, "info", {"info": {}})
                    if "info" not in episode_info:
                        episode_info = {"info": episode_info}
                TASK_ENV.close_env()
            except UnStableError as e:
                print(" -------------")
                print("Error: ", e)
                print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(" -------------")
                print("Error: ", stack_trace)
                print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue
        else:
            episode_info = {"info": {}}

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            if eval_env is not None:
                suc_test_seed_list.append(env_pack_label(eval_env))
            else:
                suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        if collision_metrics_enabled:
            try:
                TASK_ENV.setup_demo(**eval_setup_kwargs(
                    now_id, now_seed, args, eval_env, enable_collision_metrics=True))
            except NotImplementedError:
                # Task class lacks _get_target_object_names(); run without
                # collision metrics for the rest of the eval.
                print(f"\033[93m[collision-metrics] {args['task_name']} does not define "
                      f"_get_target_object_names(); collision column will be null\033[0m")
                collision_metrics_enabled = False
                try:
                    TASK_ENV.close_env()
                except Exception:
                    pass
                TASK_ENV.setup_demo(**eval_setup_kwargs(now_id, now_seed, args, eval_env))
        else:
            TASK_ENV.setup_demo(**eval_setup_kwargs(now_id, now_seed, args, eval_env))

        # Active only when the env actually built its tracking sets (bench
        # envs with the flag supported; legacy envs/ tasks silently skip).
        collision_metrics_active = (
            collision_metrics_enabled
            and getattr(TASK_ENV, "enable_collision_metrics", False)
            and hasattr(TASK_ENV, "get_collision_metrics")
            and hasattr(TASK_ENV, "robot_link_names")
        )

        lang_perturb = args.get("domain_randomization", {}).get("language_perturbation", {})
        if lang_perturb.get("enabled", False) and lang_perturb.get("instruction_bank"):
            bank_path = resolve_instruction_bank(lang_perturb["instruction_bank"])
            if os.path.exists(bank_path):
                with open(bank_path, "r") as f_bank:
                    bank = json.load(f_bank)
                pool = bank.get(args["task_name"], [])
                if pool:
                    instruction = np.random.choice(pool)
                else:
                    episode_info_list = [episode_info["info"]]
                    results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
                    instruction = np.random.choice(results[0][instruction_type])
            else:
                episode_info_list = [episode_info["info"]]
                results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
                instruction = np.random.choice(results[0][instruction_type])
        else:
            episode_info_list = [episode_info["info"]]
            results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
            instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        # Use the proxy (not .call directly) so the client-side
        # observation_window mirror resets to None — otherwise set_language
        # is skipped from episode 2 onward and the server raises
        # "Prompt is required" on the first get_action.
        model.reset_model()
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                break
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        if episode_log_path is not None:
            record = {
                "episode": TASK_ENV.test_num,
                "success": bool(succ),
            }
            if eval_env is not None:
                record["env_id"] = env_pack_label(eval_env)
                record["search_seed"] = (eval_env.get("meta") or {}).get("search_seed")
            else:
                record["seed"] = now_seed
            if collision_metrics_active:
                col = TASK_ENV.get_collision_metrics()
                record["collision"] = bool(col["is_collision"])
                record["hard_success"] = bool(succ) and not col["is_collision"]
                record["collision_count"] = int(col["total_collision_count"])
                record["collision_names"] = (col["robot_to_furniture_names"]
                                             + col["robot_to_static_object_names"]
                                             + col["target_to_static_object_names"])
            else:
                record["collision"] = None
            with open(episode_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current: \033[90m{suc_test_seed_list[-1] if suc_test_seed_list else now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        if seed_list is None and env_list is None:
            now_seed += 1

    return now_seed, TASK_ENV.suc, suc_test_seed_list


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config['port'] = args.port

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
