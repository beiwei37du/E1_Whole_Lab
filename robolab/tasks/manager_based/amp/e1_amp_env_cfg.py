import os
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robolab.tasks.manager_based.amp.mdp as mdp
from robolab.tasks.manager_based.amp.amp_env_cfg import AmpEnvCfg
from robolab.assets.robots.droidrobot import E1_12DOF_CFG
from robolab import ROBOLAB_ROOT_DIR

# Isaac Lab 顺序
ISAACLAB_JOINT_ORDER = [
    'left_hip_pitch_joint',
    'right_hip_pitch_joint',
    'left_hip_roll_joint',
    'right_hip_roll_joint',
    'left_hip_yaw_joint',
    'right_hip_yaw_joint',
    'left_knee_joint',
    'right_knee_joint',
    'left_ankle_pitch_joint',
    'right_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_ankle_roll_joint',
]

# 运动数据集中的关节顺序（需与你的 retarget 脚本一致）
DATASET_JOINT_ORDER = [
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
]

# E1_12dof 无手臂，关键身体只有脚
KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]

ANIMATION_TERM_NAME = "animation"
AMP_NUM_STEPS = 3


@configclass
class E1AmpRewards:
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    alive              = RewTerm(func=mdp.is_alive,          weight=0)
    lin_vel_z_l2       = RewTerm(func=mdp.lin_vel_z_l2,      weight=0)
    ang_vel_xy_l2      = RewTerm(func=mdp.ang_vel_xy_l2,     weight=0)
    flat_orientation_l2= RewTerm(func=mdp.flat_orientation_l2,weight=0)
    joint_vel_l2       = RewTerm(func=mdp.joint_vel_l2,      weight=0)
    joint_acc_l2       = RewTerm(func=mdp.joint_acc_l2,      weight=0)
    action_rate_l2     = RewTerm(func=mdp.action_rate_l2,    weight=0)
    smoothness_1       = RewTerm(func=mdp.smoothness_1,      weight=0)
    joint_pos_limits   = RewTerm(func=mdp.joint_pos_limits,  weight=0)
    joint_energy       = RewTerm(func=mdp.joint_energy,      weight=0)
    joint_regularization = RewTerm(func=mdp.joint_deviation_l1, weight=0)
    joint_torques_l2   = RewTerm(func=mdp.joint_torques_l2,  weight=0)
    low_speed_sway_penalty = RewTerm(
        func=mdp.low_speed_sway_penalty, weight=0,
        params={"command_name": "base_velocity", "command_threshold": 0.1},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide, weight=0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg":  SceneEntityCfg("robot",          body_names=".*_ankle_roll_link"),
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble, weight=0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )
    feet_air_time_positive_biped = RewTerm(
        func=mdp.feet_air_time_positive_biped, weight=0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    sound_suppression = RewTerm(
        func=mdp.sound_suppression_acc_per_foot, weight=0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts, weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class E1AmpEnvCfg(AmpEnvCfg):
    rewards: E1AmpRewards = E1AmpRewards()

    def __post_init__(self):
        super().__post_init__()

        # ---- Scene ----
        self.scene.robot = E1_12DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # ---- 随机化 ----
        self.events.add_base_mass.params["asset_cfg"].body_names = "pelvis"
        self.events.add_base_mass.params["mass_distribution_params"] = [-2.8, 2.95]
        self.events.randomize_rigid_body_com.params["asset_cfg"].body_names = ["pelvis"]
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "pelvis"

        # ---- 运动数据（需提前准备 E1 的 .pkl 动作文件）----
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions", "e1_lab"
        )
        self.motion_data.motion_dataset.motion_data_weights = {
            # # # walk 8
            "suibu": 1,
            # "e1_stand": 1,
        }

        # ---- 动画 ----
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS

        # ---- 判别器观测 ----
        self.observations.disc.history_length = AMP_NUM_STEPS
        self.observations.disc_demo.ref_root_local_rot_tan_norm.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_root_lin_vel_b.params["animation"]          = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_root_ang_vel_b.params["animation"]          = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_pos.params["animation"]               = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_vel.params["animation"]               = ANIMATION_TERM_NAME

        # ---- 奖励权重 ----
        self.rewards.track_lin_vel_xy_exp.weight       =  1.25
        self.rewards.track_ang_vel_z_exp.weight        =  1.25
        self.rewards.alive.weight                      =  0.15
        self.rewards.ang_vel_xy_l2.weight              = -0.1
        self.rewards.flat_orientation_l2.weight        = -1.0
        self.rewards.joint_vel_l2.weight               = -2e-4
        self.rewards.joint_acc_l2.weight               = -2.5e-7
        self.rewards.action_rate_l2.weight             = -0.01
        self.rewards.joint_pos_limits.weight           = -1.0
        self.rewards.joint_energy.weight               = -1e-4
        self.rewards.joint_torques_l2.weight           = -1e-5
        self.rewards.joint_regularization.weight       = -1e-3
        self.rewards.low_speed_sway_penalty.weight     = -1e-2
        self.rewards.feet_slide.weight                 = -0.1
        self.rewards.feet_stumble.weight               = -0.1
        self.rewards.sound_suppression.weight          = -5e-5
        self.rewards.feet_air_time_positive_biped.weight = 1.0
        self.rewards.undesired_contacts.weight         = -1.0

        # ---- 速度指令范围 ----
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # ---- 终止条件 ----
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "pelvis",
            # "torso_link",
            ".*_hip_yaw_link",
            ".*_hip_roll_link",
        ]

        if self.__class__.__name__ == "E1AmpEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class E1AmpEnvCfg_PLAY(E1AmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None

