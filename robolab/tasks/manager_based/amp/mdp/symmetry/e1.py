"""Symmetry functions for E1 robot (12 DOF, legs only).

Isaac Lab joint order:
  0: left_hip_pitch    1: right_hip_pitch   (pitch, no negate)
  2: left_hip_roll     3: right_hip_roll    (roll,  negate)
  4: left_hip_yaw      5: right_hip_yaw     (yaw,   negate)
  6: left_knee         7: right_knee        (pitch, no negate)
  8: left_ankle_pitch  9: right_ankle_pitch (pitch, no negate)
 10: left_ankle_roll  11: right_ankle_roll  (roll,  negate)

Policy obs (history_length=1, total=45):
  [0:3]   ang_vel           sign: [-1, 1, -1]
  [3:6]   projected_gravity sign: [1, -1, 1]
  [6:9]   velocity_commands sign: [1, -1, -1]
  [9:21]  joint_pos (12)
  [21:33] joint_vel (12)
  [33:45] actions (12)

Critic obs (history_length=3, single step=48, total=144):
  [0:3]   base_lin_vel      sign: [1, -1, 1]
  [3:6]   base_ang_vel      sign: [-1, 1, -1]
  [6:9]   projected_gravity sign: [1, -1, 1]
  [9:12]  velocity_commands sign: [1, -1, -1]
  [12:24] joint_pos (12)
  [24:36] joint_vel (12)
  [36:48] actions (12)
"""

from __future__ import annotations
import torch
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]

# ── 关节左右镜像 ───────────────────────────────────────────────────────────────
# L/R 互换索引
_JOINT_SWAP_L = [0, 2, 4, 6,  8, 10]
_JOINT_SWAP_R = [1, 3, 5, 7,  9, 11]
# roll/yaw 取反
_JOINT_NEGATE = [2, 3, 4, 5, 10, 11]

# ── Policy obs 维度 ────────────────────────────────────────────────────────────
_POLICY_STEP = 45    # 单步 policy obs 维度
_POLICY_HIST = 10     # PolicyCfg.history_length

# ── Critic obs 维度 ────────────────────────────────────────────────────────────
_CRITIC_STEP = 48    # 单步 critic obs 维度
_CRITIC_HIST = 10     # CriticCfg.history_length


@lru_cache(maxsize=None)
def _sign_tensor(signs: tuple, device: str, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(signs, device=device, dtype=dtype)


def _switch_joints_left_right(x: torch.Tensor) -> torch.Tensor:
    """E1 12 关节左右镜像（原地修改传入的已克隆张量）。"""
    tmp_l = x[..., _JOINT_SWAP_L].clone()
    x[..., _JOINT_SWAP_L] = x[..., _JOINT_SWAP_R]
    x[..., _JOINT_SWAP_R] = tmp_l
    x[..., _JOINT_NEGATE] *= -1
    return x


def _transform_policy_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    """镜像 policy obs（history_length=10，共 450 维）。"""
    obs = obs.clone()
    dev, dt = obs.device, obs.dtype
    for i in range(_POLICY_HIST):
        s = i * _POLICY_STEP
        obs[:, s + 0:s + 3] *= _sign_tensor((-1, 1, -1), dev, dt)  # ang_vel
        obs[:, s + 3:s + 6] *= _sign_tensor((1, -1, 1), dev, dt)  # projected_gravity
        obs[:, s + 6:s + 9] *= _sign_tensor((1, -1, -1), dev, dt)  # velocity_commands
        obs[:, s + 9:s + 21] = _switch_joints_left_right(obs[:, s + 9:s + 21].clone())  # joint_pos
        obs[:, s + 21:s + 33] = _switch_joints_left_right(obs[:, s + 21:s + 33].clone())  # joint_vel
        obs[:, s + 33:s + 45] = _switch_joints_left_right(obs[:, s + 33:s + 45].clone())  # actions
    return obs


def _transform_critic_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    """镜像 critic obs（history_length=10，共 480 维）。

    每步 48 维独立做左右镜像，循环处理 3 步历史。
    """
    obs = obs.clone()
    dev, dt = obs.device, obs.dtype
    for i in range(_CRITIC_HIST):
        s = i * _CRITIC_STEP
        obs[:, s+0 :s+3 ] *= _sign_tensor(( 1,-1,  1), dev, dt)   # base_lin_vel
        obs[:, s+3 :s+6 ] *= _sign_tensor((-1, 1, -1), dev, dt)   # base_ang_vel
        obs[:, s+6 :s+9 ] *= _sign_tensor(( 1,-1,  1), dev, dt)   # projected_gravity
        obs[:, s+9 :s+12] *= _sign_tensor(( 1,-1, -1), dev, dt)   # velocity_commands
        obs[:, s+12:s+24] = _switch_joints_left_right(obs[:, s+12:s+24].clone())  # joint_pos
        obs[:, s+24:s+36] = _switch_joints_left_right(obs[:, s+24:s+36].clone())  # joint_vel
        obs[:, s+36:s+48] = _switch_joints_left_right(obs[:, s+36:s+48].clone())  # actions
    return obs


@torch.no_grad()
def compute_symmetric_states(env, obs=None, actions=None):
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"]
        obs_aug["policy"][batch_size:] = _transform_policy_obs_left_right(obs["policy"])
        obs_aug["critic"][:batch_size] = obs["critic"]
        obs_aug["critic"][batch_size:] = _transform_critic_obs_left_right(obs["critic"])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.empty(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions
        actions_aug[batch_size:] = _switch_joints_left_right(actions.clone())
    else:
        actions_aug = None

    return obs_aug, actions_aug