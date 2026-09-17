#!/usr/bin/env python3
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import os

import numpy as np
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

from train_configs import apply_checkpoint_assets_id
from train_configs import register as register_train_configs

register_train_configs()


class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id

        policy_root = os.environ.get(
            "POLICY_ROOT",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        )
        ckpt_dir = os.path.join(policy_root, "pi0", "checkpoints", str(self.train_config_name),
                                str(self.model_name), str(self.checkpoint_id))
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(
                f"checkpoint dir not found: {ckpt_dir}\n"
                f"expected layout: $POLICY_ROOT/pi0/checkpoints/<train_config_name>/<model_name>/<checkpoint_id>"
            )

        specified_path = os.path.join(ckpt_dir, "assets")
        entries = sorted(e for e in os.listdir(specified_path) if not e.startswith("."))
        if not entries:
            raise FileNotFoundError(f"no norm-stats asset dir under {specified_path}")
        assets_id = entries[0]

        config = apply_checkpoint_assets_id(_config.get_config(self.train_config_name), assets_id)
        self.policy = _policy_config.create_trained_policy(config, ckpt_dir)
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
