from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)

import torch
from functools import lru_cache

from robolab.tasks.direct.base import BaseAgentCfg


# =============================================================================
# E1 12-DOF 关节顺序（ISAACLAB）
#  idx  关节名                   轴     镜像符号
#   0   left_hip_pitch_joint    pitch  +1
#   1   right_hip_pitch_joint   pitch  +1
#   2   left_hip_roll_joint     roll   -1
#   3   right_hip_roll_joint    roll   -1
#   4   left_hip_yaw_joint      yaw    -1
#   5   right_hip_yaw_joint     yaw    -1
#   6   left_knee_joint         pitch  +1
#   7   right_knee_joint        pitch  +1
#   8   left_ankle_pitch_joint  pitch  +1
#   9   right_ankle_pitch_joint pitch  +1
#  10   left_ankle_roll_joint   roll   -1
#  11   right_ankle_roll_joint  roll   -1
# =============================================================================


def generate_e1_joint_mirror(start_idx):
    """生成 E1 12 关节的镜像索引和符号。

    关节两两配对 (L, R)，镜像时互换；pitch 不变，roll/yaw 取反。
    """
    mirror_indices = []
    for i in range(start_idx, start_idx + 12, 2):
        mirror_indices.extend([i + 1, i])   # (L, R) → (R, L)
    mirror_signs = [
         1,  1,   # hip_pitch
        -1, -1,   # hip_roll
        -1, -1,   # hip_yaw
         1,  1,   # knee
         1,  1,   # ankle_pitch
        -1, -1,   # ankle_roll
    ]
    return mirror_indices, mirror_signs


def generate_height_scan_mirror(start_idx, rows=11, cols=17):
    """高度图镜像：前后翻转。"""
    mirror_indices = []
    for row in range(rows):
        mirror_row = rows - 1 - row
        for col in range(cols):
            mirror_indices.append(start_idx + col + mirror_row * cols)
    return mirror_indices, [1] * (rows * cols)


# =============================================================================
# Policy obs（45 维/步，历史 10 步 → 共 450 维）
#   [0:3]   ang_vel (3)
#   [3:6]   projected_gravity (3)
#   [6:9]   commands: vx, vy, wz (3)
#   [9:21]  joint_pos (12)
#   [21:33] joint_vel (12)
#   [33:45] actions (12)
# =============================================================================

joint_pos_mirror_indices,  joint_pos_mirror_signs  = generate_e1_joint_mirror(9)
joint_vel_mirror_indices,  joint_vel_mirror_signs  = generate_e1_joint_mirror(21)
action_obs_mirror_indices, action_obs_mirror_signs = generate_e1_joint_mirror(33)

policy_obs_mirror_indices = (
    [0, 1, 2,    # ang_vel
     3, 4, 5,    # projected_gravity
     6, 7, 8]    # commands
    + joint_pos_mirror_indices
    + joint_vel_mirror_indices
    + action_obs_mirror_indices
)
policy_obs_mirror_signs = (
    [-1,  1, -1,   # ang_vel:  roll_rate(-), pitch_rate(+), yaw_rate(-)
      1, -1,  1,   # gravity:  gx(+), gy(-), gz(+)
      1, -1, -1]   # commands: vx(+), vy(-), wz(-)
    + joint_pos_mirror_signs
    + joint_vel_mirror_signs
    + action_obs_mirror_signs
)

# =============================================================================
# Critic obs（flat=84 维/步，rough=271 维/步）
#   [0:45]   policy obs
#   [45:48]  lin_vel: vx, vy, vz
#   [48:50]  feet_contact:       left=48, right=49
#   [50:56]  feet_contact_force: left=[50,51,52], right=[53,54,55]  (Fx,Fy,Fz)
#   [56:58]  air_time:           left=56, right=57
#   [58:60]  feet_height:        left=58, right=59
#   [60:72]  joint_acc (12)
#   [72:84]  joint_torques (12)
#   [84:89]  feet_pos (6)
#   [89:276] height_scan 11×17=187  （rough only）
# =============================================================================

joint_acc_mirror_indices,     joint_acc_mirror_signs     = generate_e1_joint_mirror(60)
joint_torques_mirror_indices, joint_torques_mirror_signs = generate_e1_joint_mirror(72)

_extra_indices = (
    [45, 46, 47]               # lin_vel
  + [49, 48]                   # feet_contact:      swap L/R
  + [53, 54, 55, 50, 51, 52]   # feet_contact_force: swap L/R
  + [57, 56]                   # air_time:          swap L/R
  + [59, 58]                   # feet_height:       swap L/R
)
_extra_signs = (
    [1, -1, 1]                 # lin_vel: vx(+), vy(-), vz(+)
  + [1,  1]                    # feet_contact
  + [1, -1, 1,  1, -1, 1]     # feet_force: Fx(+), Fy(-), Fz(+)  × 2 脚
  + [1,  1]                    # air_time
  + [1,  1]                    # feet_height
)
# feet_pos: left=[84,85,86], right=[87,88,89]; y轴取反
_feet_pos_indices = [87, 88, 89, 84, 85, 86]
_feet_pos_signs   = [1, -1, 1, 1, -1, 1]

# flat（84 维）
critic_obs_mirror_indices = (
    policy_obs_mirror_indices
    + _extra_indices
    + joint_acc_mirror_indices
    + joint_torques_mirror_indices
)
critic_obs_mirror_signs = (
    policy_obs_mirror_signs
    + _extra_signs
    + joint_acc_mirror_signs
    + joint_torques_mirror_signs
)

# rough（271 维 = 84 + 187）
_hs_indices, _hs_signs = generate_height_scan_mirror(84, 11, 17)
critic_obs_mirror_indices_rough = critic_obs_mirror_indices + _hs_indices
critic_obs_mirror_signs_rough   = critic_obs_mirror_signs   + _hs_signs

# 动作镜像
act_mirror_indices = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
act_mirror_signs   = [1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1]

_POLICY_DIM       = 45
_CRITIC_DIM_FLAT  = 84
_CRITIC_DIM_ROUGH = 271
_HISTORY          = 10

def _expand(indices, dim):
    return [idx + i * dim for i in range(_HISTORY) for idx in indices]

policy_obs_mirror_indices_expanded  = _expand(policy_obs_mirror_indices, _POLICY_DIM)
policy_obs_mirror_signs_expanded    = policy_obs_mirror_signs * _HISTORY
critic_obs_mirror_indices_expanded  = _expand(critic_obs_mirror_indices, _CRITIC_DIM)
critic_obs_mirror_signs_expanded    = critic_obs_mirror_signs * _HISTORY


@lru_cache(maxsize=None)
def _get_signs(signs_tuple, device, dtype):
    return torch.tensor(signs_tuple, device=device, dtype=dtype)


def mirror_policy_observation(obs):
    signs = _get_signs(tuple(policy_obs_mirror_signs_expanded), obs.device, obs.dtype)
    return obs[..., policy_obs_mirror_indices_expanded] * signs


def mirror_critic_observation(obs):
    """根据 critic obs 宽度自动区分 flat / rough。"""
    if obs.shape[-1] == _CRITIC_DIM_FLAT * _HISTORY:
        indices = critic_obs_mirror_indices_flat_expanded
        signs   = _get_signs(tuple(critic_obs_mirror_signs_flat_expanded), obs.device, obs.dtype)
    else:
        indices = critic_obs_mirror_indices_rough_expanded
        signs   = _get_signs(tuple(critic_obs_mirror_signs_rough_expanded), obs.device, obs.dtype)
    return obs[..., indices] * signs


def mirror_actions(actions):
    signs = _get_signs(tuple(act_mirror_signs), actions.device, actions.dtype)
    return actions[..., act_mirror_indices] * signs


def data_augmentation_func(env, obs, actions):
    if obs is None:
        obs_aug = None
    else:
        obs_mirror = obs.clone()
        obs_mirror["policy"] = mirror_policy_observation(obs["policy"])
        if "critic" in obs.keys():
            obs_mirror["critic"] = mirror_critic_observation(obs["critic"])
        # if "perception_a" in obs.keys():
        #     obs_mirror["perception_a"] = mirror_perception_observation(obs["perception_a"])
        # if "perception_c" in obs.keys():
        #     obs_mirror["perception_c"] = mirror_perception_observation(obs["perception_c"])
        obs_aug = torch.cat([obs, obs_mirror], dim=0)
    if actions is None:
        actions_aug = None
    else:
        actions_aug = torch.cat((actions, mirror_actions(actions)), dim=0)
    return obs_aug, actions_aug

@configclass
class E1FlatAgentCfg(BaseAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "e1_flat"
        self.wandb_project = "e1_flat"
        self.logger = "tensorboard"
        self.seed = 42
        self.num_steps_per_env = 24
        self.max_iterations = 9001
        self.save_interval = 1000
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

            # 是否加入镜像数据加强
            # symmetry_cfg=None,
            symmetry_cfg=RslRlSymmetryCfg(
                use_data_augmentation=True,
                use_mirror_loss=True,
                mirror_loss_coeff=0.2,
                data_augmentation_func=data_augmentation_func,
            ),
            rnd_cfg=None,
        )
        self.clip_actions = 100.0


@configclass
class E1RoughAgentCfg(E1FlatAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "e1_rough"
        self.wandb_project = "e1_rough"
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
            # symmetry_cfg=None,
            symmetry_cfg=RslRlSymmetryCfg(
                use_data_augmentation=True,
                use_mirror_loss=True,
                mirror_loss_coeff=0.2,
                data_augmentation_func=data_augmentation_func,
            ),
            rnd_cfg=None,
        )
