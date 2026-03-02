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

from robolab.tasks.direct.base import (  # noqa:F401
    BaseAgentCfg,
)

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
    """生成 E1 12 关节的镜像索引和符号。L↔R 互换，pitch 不变，roll/yaw 取反。"""
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


def generate_height_scan_mirror(start_idx, rows=11, cols=17):
    """高度图镜像：前后翻转（行翻转）。"""
    mirror_indices = []
    for row in range(rows):
        mirror_row = rows - 1 - row
        for col in range(cols):
            mirror_indices.append(start_idx + col + mirror_row * cols)
    return mirror_indices, [1] * (rows * cols)


# =============================================================================
# Policy obs（45 维/步）
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
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
    + joint_pos_mirror_indices
    + joint_vel_mirror_indices
    + action_obs_mirror_indices
)
policy_obs_mirror_signs = (
    [-1,  1, -1,   # ang_vel:  roll(-), pitch(+), yaw(-)
      1, -1,  1,   # gravity:  gx(+), gy(-), gz(+)
      1, -1, -1]   # commands: vx(+), vy(-), wz(-)
    + joint_pos_mirror_signs
    + joint_vel_mirror_signs
    + action_obs_mirror_signs
)

# =============================================================================
# Critic obs（90 维/步，AttnEncEnv 比 base 多 feet_pos 6 维）
#   [0:45]   policy obs
#   [45:48]  lin_vel: vx, vy, vz
#   [48:50]  feet_contact: left=48, right=49
#   [50:56]  feet_contact_force: left=[50,51,52], right=[53,54,55]
#   [56:58]  air_time: left=56, right=57
#   [58:60]  feet_height: left=58, right=59
#   [60:72]  joint_acc (12)
#   [72:84]  joint_torque (12)
#   [84:90]  feet_pos: left=[84,85,86], right=[87,88,89]
# =============================================================================

joint_acc_mirror_indices,     joint_acc_mirror_signs     = generate_e1_joint_mirror(60)
joint_torques_mirror_indices, joint_torques_mirror_signs = generate_e1_joint_mirror(72)

_extra_indices = (
    [45, 46, 47]               # lin_vel
  + [49, 48]                   # feet_contact: swap L/R
  + [53, 54, 55, 50, 51, 52]   # feet_force: swap L/R
  + [57, 56]                   # air_time: swap L/R
  + [59, 58]                   # feet_height: swap L/R
)
_extra_signs = (
    [1, -1, 1]                 # lin_vel: vx(+), vy(-), vz(+)
  + [1,  1]                    # feet_contact
  + [1, -1, 1, 1, -1, 1]       # feet_force
  + [1,  1]                    # air_time
  + [1,  1]                    # feet_height
)

# feet_pos: left=[84,85,86](x,y,z), right=[87,88,89]; 镜像时 y 取反
_feet_pos_indices = [87, 88, 89, 84, 85, 86]
_feet_pos_signs   = [1, -1, 1, 1, -1, 1]

critic_obs_mirror_indices = (
    policy_obs_mirror_indices
    + _extra_indices
    + joint_acc_mirror_indices
    + joint_torques_mirror_indices
    + _feet_pos_indices
)
critic_obs_mirror_signs = (
    policy_obs_mirror_signs
    + _extra_signs
    + joint_acc_mirror_signs
    + joint_torques_mirror_signs
    + _feet_pos_signs
)

# 动作镜像（12 维）
act_mirror_indices = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
act_mirror_signs   = [1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1]

# 高度图镜像（187 维，perception obs 直接为一帧高度图）
map_scan_mirror_indices, map_scan_mirror_signs = generate_height_scan_mirror(0, 11, 17)

# =============================================================================
# policy obs: 5 步 × 45 维 = 225；critic obs: 5 步 × 90 维 = 450
# =============================================================================
_POLICY_DIM = 45
_CRITIC_DIM = 90
_HISTORY    = 5


def _expand(indices, dim):
    return [idx + i * dim for i in range(_HISTORY) for idx in indices]


policy_obs_mirror_indices_expanded = _expand(policy_obs_mirror_indices, _POLICY_DIM)
policy_obs_mirror_signs_expanded   = policy_obs_mirror_signs * _HISTORY

critic_obs_mirror_indices_expanded = _expand(critic_obs_mirror_indices, _CRITIC_DIM)
critic_obs_mirror_signs_expanded   = critic_obs_mirror_signs * _HISTORY


@lru_cache(maxsize=None)
def get_policy_obs_mirror_signs_tensor(device, dtype):
    return torch.tensor(policy_obs_mirror_signs_expanded, device=device, dtype=dtype)


def mirror_policy_observation(policy_obs):
    mirrored = policy_obs[..., policy_obs_mirror_indices_expanded]
    signs = get_policy_obs_mirror_signs_tensor(device=policy_obs.device, dtype=policy_obs.dtype)
    return mirrored * signs


@lru_cache(maxsize=None)
def get_critic_obs_mirror_signs_tensor(device, dtype):
    return torch.tensor(critic_obs_mirror_signs_expanded, device=device, dtype=dtype)


def mirror_critic_observation(critic_obs):
    mirrored = critic_obs[..., critic_obs_mirror_indices_expanded]
    signs = get_critic_obs_mirror_signs_tensor(device=critic_obs.device, dtype=critic_obs.dtype)
    return mirrored * signs


@lru_cache(maxsize=None)
def get_act_mirror_signs_tensor(device, dtype):
    return torch.tensor(act_mirror_signs, device=device, dtype=dtype)


def mirror_actions(actions):
    mirrored = actions[..., act_mirror_indices]
    signs = get_act_mirror_signs_tensor(device=actions.device, dtype=actions.dtype)
    return mirrored * signs


@lru_cache(maxsize=None)
def get_map_scan_mirror_signs_tensor(device, dtype):
    return torch.tensor(map_scan_mirror_signs, device=device, dtype=dtype)


def mirror_perception_observation(perception_obs):
    mirrored = perception_obs[..., map_scan_mirror_indices]
    signs = get_map_scan_mirror_signs_tensor(device=perception_obs.device, dtype=perception_obs.dtype)
    return mirrored * signs


def data_augmentation_func(env, obs, actions):
    if obs is None:
        obs_aug = None
    else:
        obs_mirror = obs.clone()
        obs_mirror["policy"] = mirror_policy_observation(obs["policy"])
        if "critic" in obs.keys():
            obs_mirror["critic"] = mirror_critic_observation(obs["critic"])
        if "perception_a" in obs.keys():
            obs_mirror["perception_a"] = mirror_perception_observation(obs["perception_a"])
        if "perception_c" in obs.keys():
            obs_mirror["perception_c"] = mirror_perception_observation(obs["perception_c"])
        obs_aug = torch.cat([obs, obs_mirror], dim=0)
    if actions is None:
        actions_aug = None
    else:
        actions_aug = torch.cat((actions, mirror_actions(actions)), dim=0)
    return obs_aug, actions_aug


@configclass
class RslRlPpoEncActorCriticCfg(RslRlPpoActorCriticCfg):
    embedding_dim: int = 64
    head_num: int = 8
    map_size: tuple = (17, 11)
    map_resolution: float = 0.1
    actor_history_length: int = 5
    critic_history_length: int = 1
    enable_critic_estimation: bool = False
    estimation_slice: list = [45, 46, 47]
    estimaiton_hidden_dims: list = [256, 64]
    enable_obs_encoder: bool = False
    obs_encoder_hidden_dims: list = [256, 64]
    latent_dim: int = 16


@configclass
class RslRlPpoEncAlgorithmCfg(RslRlPpoAlgorithmCfg):
    enable_aux_loss: bool = False
    aux_loss_coef: float = 1.0


@configclass
class E1AttnEncAgentCfg(BaseAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "e1_attn_enc"
        self.wandb_project = "e1_attn_enc"
        self.logger = "tensorboard"
        self.seed = 42
        self.obs_groups = {"policy": ["policy"], "critic": ["critic"], "perception": ["perception_a", "perception_c"]}
        self.num_steps_per_env = 24
        self.max_iterations = 9001
        self.save_interval = 1000
        self.policy = RslRlPpoEncActorCriticCfg(
            class_name="ActorCriticAttnEnc",
            init_noise_std=1.0,
            noise_std_type="scalar",
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            embedding_dim=32,
            head_num=4,
            map_size=(17, 11),
            map_resolution=0.1,
            actor_history_length=5,
            critic_history_length=5,
            enable_critic_estimation=True,
            estimation_slice=[45, 46, 47],
            estimaiton_hidden_dims=[256, 64],
            enable_obs_encoder=True,
            latent_dim=32,
            obs_encoder_hidden_dims=[256, 128],
        )
        self.algorithm = RslRlPpoEncAlgorithmCfg(
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
            enable_aux_loss=True,
            aux_loss_coef=0.05,
            normalize_advantage_per_mini_batch=False,
            symmetry_cfg=RslRlSymmetryCfg(
                use_data_augmentation=True,
                use_mirror_loss=True,
                mirror_loss_coeff=0.1,
                data_augmentation_func=data_augmentation_func,
            ),
            rnd_cfg=None,
        )
        self.clip_actions = 100.0
