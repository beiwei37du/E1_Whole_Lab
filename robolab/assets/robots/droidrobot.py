import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robolab.assets import ISAAC_DATA_DIR

E1_12DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ISAAC_DATA_DIR}/robots/droidrobot/E1/E1_12dof.urdf",
        fix_base=False,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.715),
        joint_pos={
            "left_hip_pitch_joint": -0.1,
            "left_hip_roll_joint": 0,
            "left_hip_yaw_joint": 0,
            "left_knee_joint": 0.2,
            "left_ankle_pitch_joint": -0.1,

            "right_hip_pitch_joint": -0.1,
            "right_hip_roll_joint": 0,
            "right_hip_yaw_joint": 0,
            "right_knee_joint": 0.2,
            "right_ankle_pitch_joint": -0.1,

            # "left_shoulder_pitch_joint": 0.18,
            # "left_shoulder_roll_joint": 0.06,
            # "left_shoulder_yaw_joint": 0.06,
            # "left_elbow_joint": 0.78,
            # "right_arm_pitch_joint": 0.18,
            # "right_arm_roll_joint": -0.06,
            # "right_elbow_joint": 0.78,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                # ".*torso.*",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint":   36.0,
                ".*_hip_roll_joint":  60.0,
                ".*_hip_pitch_joint": 60.0,
                ".*_knee_joint":      59.3,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint":   50.27,
                ".*_hip_roll_joint":  20.42,
                ".*_hip_pitch_joint": 20.42,
                ".*_knee_joint":      30.65,
            },
            stiffness={
                ".*_hip_yaw_joint":   100.0,
                ".*_hip_roll_joint":  150.0,
                ".*_hip_pitch_joint": 150.0,
                ".*_knee_joint":      150.0,
            },
            damping={
                ".*_hip_yaw_joint":   3,
                ".*_hip_roll_joint":  3,
                ".*_hip_pitch_joint": 3,
                ".*_knee_joint":      5,
            },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint"
            ],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 59.3,
                ".*_ankle_roll_joint":  14.0,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 30.65,
                ".*_ankle_roll_joint":  32.99,
            },
            stiffness={
                ".*_ankle_pitch_joint": 20,
                ".*_ankle_roll_joint":  20,
            },
            damping=2.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        # "shoulders": DelayedPDActuatorCfg(
        #     joint_names_expr=[
        #         ".*_shoulder_pitch_joint",
        #         ".*_shoulder_roll_joint",
        #         ".*_shoulder_yaw_joint",
        #         ".*_elbow_pitch_joint",
        #     ],
        #     effort_limit_sim={
        #         ".*_shoulder_pitch_joint": 36,
        #         ".*_shoulder_roll_joint":  36,
        #         ".*_shoulder_yaw_joint":   36,
        #         ".*_elbow_pitch_joint":    36,
        #     },
        #     velocity_limit_sim={
        #         ".*_shoulder_pitch_joint": 3,
        #         ".*_shoulder_roll_joint":  3,
        #         ".*_shoulder_yaw_joint":   3,
        #         ".*_elbow_pitch_joint":    5,
        #     },
        #     stiffness={
        #         ".*_shoulder_pitch_joint": 3,
        #         ".*_shoulder_roll_joint":  3,
        #         ".*_shoulder_yaw_joint":   3,
        #         ".*_elbow_pitch_joint":    3,
        #     },
        #     damping=2.0,
        #     armature=0.01,
        #     min_delay=0,
        #     max_delay=2,
        # ),
    },
)

E1_21DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ISAAC_DATA_DIR}/robots/droidrobot/E1/E1_21dof.urdf",
        fix_base=False,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.715),
        joint_pos={
            "left_hip_pitch_joint": -0.1,
            "left_hip_roll_joint": 0,
            "left_hip_yaw_joint": 0,
            "left_knee_joint": 0.2,
            "left_ankle_pitch_joint": -0.1,

            "right_hip_pitch_joint": -0.1,
            "right_hip_roll_joint": 0,
            "right_hip_yaw_joint": 0,
            "right_knee_joint": 0.2,
            "right_ankle_pitch_joint": -0.1,

            "left_shoulder_pitch_joint": 0.18,
            "left_shoulder_roll_joint": 0.06,
            "left_shoulder_yaw_joint": 0.06,
            "left_elbow_joint": 0.78,
            "right_shoulder_pitch_joint": 0.18,
            "right_shoulder_roll_joint": 0.06,
            "right_shoulder_yaw_joint": 0.06,
            "right_elbow_joint": 0.78,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "waist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint":   36.0,
                ".*_hip_roll_joint":  60.0,
                ".*_hip_pitch_joint": 60.0,
                ".*_knee_joint":      59.3,
                "waist_yaw_joint":    60.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint":   50.27,
                ".*_hip_roll_joint":  20.42,
                ".*_hip_pitch_joint": 20.42,
                ".*_knee_joint":      30.65,
                "waist_yaw_joint":    20.42
            },
            stiffness={
                ".*_hip_yaw_joint":   100.0,
                ".*_hip_roll_joint":  150.0,
                ".*_hip_pitch_joint": 150.0,
                ".*_knee_joint":      150.0,
                "waist_yaw_joint":    100.0,
            },
            damping={
                ".*_hip_yaw_joint":   3,
                ".*_hip_roll_joint":  3,
                ".*_hip_pitch_joint": 3,
                ".*_knee_joint":      5,
                "waist_yaw_joint":    3,
            },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint"
            ],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 59.3,
                ".*_ankle_roll_joint":  14.0,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 30.65,
                ".*_ankle_roll_joint":  32.99,
            },
            stiffness={
                ".*_ankle_pitch_joint": 20,
                ".*_ankle_roll_joint":  20,
            },
            damping=2.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "shoulders": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 36,
                ".*_shoulder_roll_joint":  36,
                ".*_shoulder_yaw_joint":   14,
                ".*_elbow_joint":    36,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 3,
                ".*_shoulder_roll_joint":  3,
                ".*_shoulder_yaw_joint":   3,
                ".*_elbow_joint":    5,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 40,
                ".*_shoulder_roll_joint":  40,
                ".*_shoulder_yaw_joint":   40,
                ".*_elbow_joint":    40,
            },
            damping=2.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
    },
)