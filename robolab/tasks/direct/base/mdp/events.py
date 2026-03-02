from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from robolab.envs.base.base_env import BaseEnv

def set_joint_position_limits(
        env: BaseEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        joint_limits: dict[str, tuple[float, float]],
):
    """Set custom joint position limits for specified joints.

    Args:
        env: The environment instance.
        env_ids: The environment indices. If None, all environments are used.
        asset_cfg: Scene entity configuration for the asset.
        joint_limits: Dictionary mapping joint name regex patterns to (lower, upper) limits.
            Example: {".*_knee_joint": (0.1, 2.0), ".*_hip_pitch_joint": (-2.5, 2.5)}
    """
    import re

    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # get current joint limits
    limits = asset.data.joint_pos_limits.clone()  # (num_envs, num_joints, 2)

    # get joint names
    joint_names = asset.joint_names

    # apply custom limits
    for pattern, (lower, upper) in joint_limits.items():
        for joint_idx, joint_name in enumerate(joint_names):
            if re.match(pattern, joint_name):
                limits[env_ids, joint_idx, 0] = lower
                limits[env_ids, joint_idx, 1] = upper
                print(f"[INFO] Set joint '{joint_name}' limits to [{lower}, {upper}]")

    # write limits to simulation
    asset.write_joint_position_limit_to_sim(limits, env_ids=env_ids)
