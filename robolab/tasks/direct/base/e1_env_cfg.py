# observation_space = 3(ang_vel) + 3(gravity) + 3(cmd) + 12(joint_pos) + 12(joint_vel) + 12(action) = 45
# state_space(flat)  = 45 + 3(lin_vel) + 2(feet_contact) + 6(feet_force) + 2(air_time) + 2(feet_height) + 12(joint_acc) + 12(joint_torque) = 84
# state_space(rough) = 84 + 11×17(height_scan) = 271

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass
from robolab.tasks.direct.base import mdp

from robolab.assets.robots import E1_12DOF_CFG  # 改这里

from robolab.tasks.direct.base import (  # noqa:F401
    BaseAgentCfg,
    BaseEnvCfg,
    RewardCfg,
    HeightScannerCfg,
    SceneContextCfg,
    RobotCfg,
    ObsScalesCfg,
    NormalizationCfg,
    CommandRangesCfg,
    CommandsCfg,
    NoiseScalesCfg,
    NoiseCfg,
    EventCfg,
    GRAVEL_TERRAINS_CFG,
    ROUGH_TERRAINS_CFG,
    ROUGH_HARD_TERRAINS_CFG,
    SceneCfg
)

@configclass
class E1RewardCfg(RewardCfg):
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    upward = RewTerm(func=mdp.upward, weight=0.4)
    energy = RewTerm(func=mdp.energy, weight=-1e-4)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-2e-2)
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=-2e-2)

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*ankle_roll.*).*")},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 0.4},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide, weight=-0.3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "min": 0.16, "max": 0.50},
    )
    knee_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_knee.*"]), "min": 0.18, "max": 0.35},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])},
    )
    feet_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"])},
    )

    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.03,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*"])},
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch.*", ".*_knee.*", ".*_ankle_pitch.*"])},
    )
    joint_deviation_feets = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.03,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_roll.*"])},
    )
    stand_still = RewTerm(
        func=mdp.stand_still, weight=-0.2,
        params={
            "pos_cfg": SceneEntityCfg("robot", joint_names=[".*_hip.*", ".*_knee.*", ".*_ankle.*"]),
            "vel_cfg": SceneEntityCfg("robot", joint_names=[".*_hip.*", ".*_knee.*", ".*_ankle.*"]),
            "pos_weight": 1.0, "vel_weight": 0.04,
        },
    )

    feet_height = RewTerm(
            func=mdp.feet_height,
            weight=0.2,
            params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
                    "sensor_cfg1": SceneEntityCfg("left_feet_scanner"),
                    "sensor_cfg2": SceneEntityCfg("right_feet_scanner"),
                    "ankle_height":0.06,"threshold":0.02})


@configclass
class E1FlatEnvCfg(BaseEnvCfg):
    reward = E1RewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.action_space = 12
        self.observation_space = 45     # ← 3+3+3+12+12+12
        self.state_space = 84           # ← 45+3+2+6+2+2+12+12

        self.scene_context.robot = E1_12DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene_context.height_scanner.prim_body_name = "pelvis"

        self.scene_context.terrain_type = "generator"
        self.scene_context.terrain_generator = GRAVEL_TERRAINS_CFG
        self.scene_context.height_scanner.enable_height_scan = False
        self.scene = SceneCfg(config=self.scene_context, physics_dt=self.sim.dt, step_dt=self.decimation * self.sim.dt)

        # 接触终止 body names
        self.robot.terminate_contacts_body_names = ["pelvis", ".*_hip_yaw_link", ".*_hip_roll_link"]
        self.robot.feet_body_names = [".*ankle_roll.*"]

        # 随机化配置
        self.events.add_base_mass.params["asset_cfg"].body_names = ["pelvis"]
        self.events.randomize_rigid_body_com.params["asset_cfg"].body_names = ["pelvis"]
        self.events.scale_link_mass.params["asset_cfg"].body_names = ["left_.*_link", "right_.*_link"]
        self.events.scale_actuator_gains.params["asset_cfg"].joint_names = [".*_joint"]
        self.events.scale_joint_parameters.params["asset_cfg"].joint_names = [".*_joint"]
        self.events.set_joint_limits.params["joint_limits"] = {
            "L_hip_pitch_joint":   (-1.57, 1.57),  # 左髋pitch
            "L_hip_roll_joint":    ( -0.3, 1.57),  # 左髋roll
            "L_hip_yaw_joint":     (-1.57, 1.57),  # 左髋yaw
            "L_knee_joint":        (  0.0, 2.45),  # 左膝关节
            "L_ankle_pitch_joint": (-0.45, 0.45),  # 左踝pitch
            "L_ankle_roll_joint":  (-0.26, 0.26),  # 左踝roll

            "R_hip_pitch_joint":   (-1.57, 1.57),  # 右髋pitch
            "R_hip_roll_joint":    (-1.57,  0.3),  # 右髋roll
            "R_hip_yaw_joint":     (-1.57, 1.57),  # 右髋yaw
            "R_knee_joint":        (  0.0, 2.45),  # 右膝关节
            "R_ankle_pitch_joint": (-0.45, 0.45),  # 右踝pitch
            "R_ankle_roll_joint":  (-0.26, 0.26),  # 右踝roll
        }

        # scales & noise
        self.robot.action_scale = 0.25
        self.noise.noise_scales.joint_vel = 1.75
        self.noise.noise_scales.joint_pos = 0.03


@configclass
class E1RoughEnvCfg(E1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.state_space = 271          # ← 84 + 11×17(高度扫描)
        self.scene_context.height_scanner.enable_height_scan = True
        self.scene_context.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene = SceneCfg(config=self.scene_context, physics_dt=self.sim.dt, step_dt=self.decimation * self.sim.dt)
        self.sim.physx.gpu_collision_stack_size = 2**29
