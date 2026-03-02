# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# =============================================================================
# E1 12-DOF Interrupt — observation layout (per step, history=10)
#
# Policy obs  46 dims/step  →  460 total
#   [0:3]    ang_vel
#   [3:6]    projected_gravity
#   [6:9]    commands (vx, vy, wz)
#   [9:21]   joint_pos  (12)
#   [21:33]  joint_vel  (12)
#   [33:45]  actions    (12)
#   [45]     interrupt_mask (1)
#
# Critic obs  85 dims/step  →  850 total  (flat)
#   [0:46]   policy obs (above)
#   [46:49]  lin_vel  (3)
#   [49:51]  feet_contact  L=49, R=50
#   [51:57]  feet_contact_force  L_xyz=[51,52,53], R_xyz=[54,55,56]
#   [57:59]  air_time   L=57, R=58
#   [59:61]  feet_height  L=59, R=60
#   [61:73]  joint_acc   (12)
#   [73:85]  joint_torques (12)
#
# Joint ordering assumed (IsaacLab / interleaved L-R pairs, from e1_agent_cfg):
#   0  left_hip_pitch   1  right_hip_pitch
#   2  left_hip_roll    3  right_hip_roll
#   4  left_hip_yaw     5  right_hip_yaw
#   6  left_knee        7  right_knee
#   8  left_ankle_pitch 9  right_ankle_pitch
#  10  left_ankle_roll 11  right_ankle_roll
# =============================================================================

import torch
from functools import lru_cache

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)

from robolab.tasks.direct.base import BaseAgentCfg


# ---------------------------------------------------------------------------
# Mirror helpers
# ---------------------------------------------------------------------------

def _e1_joint_mirror(start_idx):
    """Interleaved L/R pairs: swap each (L_i, R_i) → (R_i, L_i)."""
    mirror_indices = []
    for i in range(start_idx, start_idx + 12, 2):
        mirror_indices.extend([i + 1, i])
    mirror_signs = [
         1,  1,   # hip_pitch
        -1, -1,   # hip_roll
        -1, -1,   # hip_yaw
         1,  1,   # knee
         1,  1,   # ankle_pitch
        -1, -1,   # ankle_roll
    ]
    return mirror_indices, mirror_signs


# -- Policy obs mirror (46 dims / step) --
_jp_idx, _jp_sgn = _e1_joint_mirror(9)
_jv_idx, _jv_sgn = _e1_joint_mirror(21)
_ac_idx, _ac_sgn = _e1_joint_mirror(33)

policy_obs_mirror_indices = (
    [0, 1, 2,   # ang_vel:           roll(-), pitch(+), yaw(-)
     3, 4, 5,   # projected_gravity: gx(+),   gy(-),    gz(+)
     6, 7, 8]   # commands:          vx(+),   vy(-),    wz(-)
    + _jp_idx + _jv_idx + _ac_idx
    + [45]      # interrupt_mask: symmetric
)
policy_obs_mirror_signs = (
    [-1, 1, -1,
      1, -1, 1,
      1, -1, -1]
    + _jp_sgn + _jv_sgn + _ac_sgn
    + [1]
)

# -- Critic obs mirror (85 dims / step) --
_ja_idx, _ja_sgn = _e1_joint_mirror(61)
_jt_idx, _jt_sgn = _e1_joint_mirror(73)

_extra_indices = (
    [46, 47, 48]                     # lin_vel:          vx(+), vy(-), vz(+)
    + [50, 49]                        # feet_contact:     swap L/R
    + [54, 55, 56, 51, 52, 53]        # feet_force:       swap L/R blocks
    + [58, 57]                        # air_time:         swap L/R
    + [60, 59]                        # feet_height:      swap L/R
)
_extra_signs = (
    [1, -1, 1]
    + [1,  1]
    + [1, -1, 1,  1, -1, 1]
    + [1,  1]
    + [1,  1]
)

critic_obs_mirror_indices = (
    policy_obs_mirror_indices
    + _extra_indices
    + _ja_idx
    + _jt_idx
)
critic_obs_mirror_signs = (
    policy_obs_mirror_signs
    + _extra_signs
    + _ja_sgn
    + _jt_sgn
)

# -- Action mirror (12 dims) --
act_mirror_indices = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
act_mirror_signs   = [1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1]

# -- History expansion --
_POLICY_DIM = 46
_CRITIC_DIM  = 85
_HISTORY     = 10


def _expand(indices, dim):
    return [idx + i * dim for i in range(_HISTORY) for idx in indices]


policy_obs_mirror_indices_expanded = _expand(policy_obs_mirror_indices, _POLICY_DIM)
policy_obs_mirror_signs_expanded   = policy_obs_mirror_signs * _HISTORY
critic_obs_mirror_indices_expanded = _expand(critic_obs_mirror_indices, _CRITIC_DIM)
critic_obs_mirror_signs_expanded   = critic_obs_mirror_signs * _HISTORY


# ---------------------------------------------------------------------------
# Cached sign tensors
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _get_signs(signs_tuple, device, dtype):
    return torch.tensor(signs_tuple, device=device, dtype=dtype)


def mirror_policy_observation(obs):
    signs = _get_signs(tuple(policy_obs_mirror_signs_expanded), obs.device, obs.dtype)
    return obs[..., policy_obs_mirror_indices_expanded] * signs


def mirror_critic_observation(obs):
    signs = _get_signs(tuple(critic_obs_mirror_signs_expanded), obs.device, obs.dtype)
    return obs[..., critic_obs_mirror_indices_expanded] * signs


def mirror_actions(actions):
    signs = _get_signs(tuple(act_mirror_signs), actions.device, actions.dtype)
    return actions[..., act_mirror_indices] * signs


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

def data_augmentation_func(env, obs, actions):
    if obs is None:
        obs_aug = None
    else:
        obs_mirror = obs.clone()
        obs_mirror["policy"] = mirror_policy_observation(obs["policy"])
        if "critic" in obs.keys():
            obs_mirror["critic"] = mirror_critic_observation(obs["critic"])
        obs_aug = torch.cat([obs, obs_mirror], dim=0)
    if actions is None:
        actions_aug = None
    else:
        actions_aug = torch.cat((actions, mirror_actions(actions)), dim=0)
    return obs_aug, actions_aug


# ---------------------------------------------------------------------------
# Agent config
# ---------------------------------------------------------------------------

@configclass
class E1InterruptAgentCfg(BaseAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "e1_interrupt"
        self.wandb_project   = "e1_interrupt"
        self.logger          = "tensorboard"
        self.seed            = 42
        self.num_steps_per_env = 24
        self.max_iterations    = 9001
        self.save_interval     = 1000
        self.algorithm = RslRlPpoAlgorithmCfg(
            class_name="PPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            normalize_advantage_per_mini_batch=False,
            symmetry_cfg=RslRlSymmetryCfg(
                use_data_augmentation=True,
                use_mirror_loss=True,
                mirror_loss_coeff=0.2,
                data_augmentation_func=data_augmentation_func,
            ),
            rnd_cfg=None,
        )
        self.clip_actions = 100.0
