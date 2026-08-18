#!/usr/bin/env python3
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import json
import os
import sys
import jax
import numpy as np
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import cv2
from PIL import Image

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id

        # Resolve against POLICY_ROOT so the server works from any cwd (the eval
        # wrappers cd to the repo root, but slurm/direct invocations may not).
        policy_root = os.environ.get(
            "POLICY_ROOT",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        )
        ckpt_dir = os.path.join(policy_root, "pi05", "checkpoints", str(self.train_config_name),
                                str(self.model_name), str(self.checkpoint_id))
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(
                f"checkpoint dir not found: {ckpt_dir}\n"
                f"expected layout: $POLICY_ROOT/pi05/checkpoints/<train_config_name>/<model_name>/<checkpoint_id>"
            )

        specified_path = os.path.join(ckpt_dir, "assets")
        # The assets dir holds one subdir per norm-stats asset id. Ignore stray
        # dotfiles and pick deterministically instead of relying on listdir order.
        entries = sorted(e for e in os.listdir(specified_path) if not e.startswith("."))
        if not entries:
            raise FileNotFoundError(f"no norm-stats asset dir under {specified_path}")
        if len(entries) > 1:
            print(f"[pi_model] multiple asset ids in {specified_path}: {entries}; using {entries[0]!r}")
        assets_id = entries[0]

        is_pytorch_ckpt = os.path.exists(os.path.join(ckpt_dir, "model.safetensors"))

        config = _config.get_config(self.train_config_name)
        # PI05_FORCE_FP32 is a JAX/XLA-only workaround: on GB10/Blackwell bf16 inference aborts in XLA
        # (bf16->f16 lowering bug), so force float32 compute for the JAX backend. Must NOT be applied to
        # the PyTorch backend (it runs paligemma in bf16 to match training; forcing fp32 only on the
        # action head would be a precision mismatch that degrades actions). Keep PyTorch at native bf16.
        # Default OFF: fp32 doubles the resident params, which does not fit on a 16GB card
        # (12GB of params needs ~12GB resident plus a ~4.5GB transient buffer during the orbax
        # restore, against a 0.85*16GB budget -> RESOURCE_EXHAUSTED before the server ever
        # serves). Export PI05_FORCE_FP32=1 to opt in on GB10/Blackwell, or to reproduce the
        # fp32 training-data reference where memory allows.
        if not is_pytorch_ckpt and os.environ.get("PI05_FORCE_FP32", "0") == "1":
            import dataclasses as _dc
            config = _dc.replace(config, model=_dc.replace(config.model, dtype="float32"))
            print("[pi_model] PI05_FORCE_FP32=1 -> model compute dtype = float32")
        elif is_pytorch_ckpt:
            print("[pi_model] pytorch checkpoint -> native bf16 inference (PI05_FORCE_FP32 not applied)")
        self.policy = _policy_config.create_trained_policy(
            config,
            ckpt_dir,
            robotwin_repo_id=assets_id,
            )
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")

    # Server proxies methods by attribute lookup; eval client calls
    # `reset_model` whereas deploy_policy.py keeps the legacy typo. Alias.
    def reset_model(self):
        self.reset_obsrvationwindows()
