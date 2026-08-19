"""RoboPRO TrainConfig overlays registered into openpi at import time.

These are glue instantiations of openpi's public config API (not a copy of
openpi source). Call `register()` before `openpi.training.config.get_config`.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import openpi.transforms as _transforms
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.training import config as _config
from openpi.training import weight_loaders
from openpi.training.config import DataConfig
from openpi.training.config import LeRobotAlohaDataConfig
from openpi.training.config import TrainConfig

_GLUE = Path(__file__).resolve().parent
_CHECKPOINT_BASE = str(_GLUE / "checkpoints")
_REGISTERED = False


def _aloha_data(repo_id: str, *, adapt_to_pi: bool | None = None) -> LeRobotAlohaDataConfig:
    kwargs: dict = {
        "repo_id": repo_id,
        "repack_transforms": _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.cam_high",
                            "cam_left_wrist": "observation.images.cam_left_wrist",
                            "cam_right_wrist": "observation.images.cam_right_wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        ),
        "base_config": DataConfig(prompt_from_task=True),
    }
    if adapt_to_pi is not None:
        kwargs["adapt_to_pi"] = adapt_to_pi
    return LeRobotAlohaDataConfig(**kwargs)


def _configs() -> list[TrainConfig]:
    return [
        TrainConfig(
            name="pi05_robopro_cfm",
            model=pi0_config.Pi0Config(pi05=True),
            data=_aloha_data("local/office_multimodal_15task"),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            num_train_steps=20_000,
            batch_size=64,
            fsdp_devices=1,
            checkpoint_base_dir=_CHECKPOINT_BASE,
        ),
        TrainConfig(
            name="pi05_aloha_full_base",
            model=pi0_config.Pi0Config(pi05=True),
            data=_aloha_data("your_repo_id"),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            num_train_steps=20_000,
            batch_size=64,
            fsdp_devices=1,
            checkpoint_base_dir=_CHECKPOINT_BASE,
        ),
        TrainConfig(
            name="pi0_base_aloha_robotwin_lora",
            model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
            data=_aloha_data("your_repo_id", adapt_to_pi=False),
            freeze_filter=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            batch_size=32,
            weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30000,
            fsdp_devices=1,
            checkpoint_base_dir=_CHECKPOINT_BASE,
        ),
        TrainConfig(
            name="pi0_fast_aloha_robotwin_lora",
            model=pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora"),
            data=_aloha_data("your_repo_id", adapt_to_pi=False),
            freeze_filter=pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora").get_freeze_filter(),
            batch_size=32,
            weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
            num_train_steps=30000,
            fsdp_devices=2,
            checkpoint_base_dir=_CHECKPOINT_BASE,
        ),
        TrainConfig(
            name="pi0_base_aloha_robotwin_full",
            model=pi0_config.Pi0Config(),
            data=_aloha_data("your_repo_id", adapt_to_pi=False),
            freeze_filter=pi0_config.Pi0Config().get_freeze_filter(),
            batch_size=32,
            weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30000,
            fsdp_devices=4,
            checkpoint_base_dir=_CHECKPOINT_BASE,
        ),
        TrainConfig(
            name="pi0_fast_aloha_robotwin_full",
            model=pi0_fast.Pi0FASTConfig(),
            data=_aloha_data("your_repo_id", adapt_to_pi=False),
            freeze_filter=pi0_fast.Pi0FASTConfig().get_freeze_filter(),
            batch_size=32,
            weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
            num_train_steps=30000,
            fsdp_devices=1,
            checkpoint_base_dir=_CHECKPOINT_BASE,
        ),
    ]


def register() -> None:
    """Insert RoboPRO configs into openpi's global registry (idempotent)."""
    global _REGISTERED
    extras = _configs()
    names = {c.name for c in extras}
    _config._CONFIGS[:] = [c for c in _config._CONFIGS if c.name not in names] + extras
    if len({c.name for c in _config._CONFIGS}) != len(_config._CONFIGS):
        raise ValueError("Config names must be unique.")
    _config._CONFIGS_DICT.clear()
    _config._CONFIGS_DICT.update({c.name: c for c in _config._CONFIGS})
    _REGISTERED = True


def apply_checkpoint_assets_id(config: TrainConfig, assets_id: str) -> TrainConfig:
    """Point data.repo_id at the checkpoint's norm-stats subdir (used as asset_id)."""
    if not hasattr(config.data, "repo_id"):
        return config
    return dataclasses.replace(config, data=dataclasses.replace(config.data, repo_id=assets_id))
