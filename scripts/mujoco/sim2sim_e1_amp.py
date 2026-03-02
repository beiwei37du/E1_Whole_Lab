# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import onnxruntime as ort
import os
import time
import cv2
import matplotlib.pyplot as plt
from robolab.assets import ISAAC_DATA_DIR
from scripts.mujoco.keyboard import KeyboardCommand, print_keyboard_instructions


def get_obs(data):
    """从 MuJoCo data 中提取观测量。"""
    q     = data.qpos.astype(np.double)
    dq    = data.qvel.astype(np.double)
    quat  = data.sensor('bq').data[[1, 2, 3, 0]].astype(np.double)
    r     = R.from_quat(quat)
    v     = r.apply(data.qvel[:3], inverse=True).astype(np.double)   # 机体系线速度
    omega = data.sensor('gyro').data.astype(np.double)
    gvec  = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD 力矩计算。"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def run_mujoco(policy, cfg, headless=False):
    print_keyboard_instructions()
    cmd = KeyboardCommand(
        min_vx=cfg.cmd_config.min_vx, max_vx=cfg.cmd_config.max_vx,
        min_vy=cfg.cmd_config.min_vy, max_vy=cfg.cmd_config.max_vy,
        min_dyaw=cfg.cmd_config.min_dyaw, max_dyaw=cfg.cmd_config.max_dyaw,
    )
    cmd.start()

    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data  = mujoco.MjData(model)
    data.qpos[-cfg.robot_config.num_actions:] = cfg.robot_config.default_pos
    mujoco.mj_step(model, data)
    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()

    os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
    os.environ['MUJOCO_GL'] = 'glfw'

    if headless:
        renderer = mujoco.Renderer(model, width=1920, height=1080)
        fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
        cam = mujoco.MjvCamera()
        cam.distance  = 3.0
        cam.azimuth   = 45.0
        cam.elevation = -20.0
        cam.lookat    = [0, 0, 0.8]
        out = cv2.VideoWriter(
            'simulation_e1_amp.mp4', fourcc,
            1.0 / cfg.sim_config.dt / cfg.sim_config.decimation,
            (1920, 1080)
        )
    else:
        viewer = mujoco_viewer.MujocoViewer(model, data, mode='window', width=1920, height=1080)
        viewer.cam.distance  = 3.0
        viewer.cam.azimuth   = 45.0
        viewer.cam.elevation = -20.0
        viewer.cam.lookat    = [0, 0, 0.8]

    num_actions = cfg.robot_config.num_actions
    target_pos  = np.zeros(num_actions, dtype=np.double)
    action      = np.zeros(num_actions, dtype=np.double)
    tau         = np.zeros(num_actions, dtype=np.double)

    # 历史观测缓冲：IsaacLab 按 term 分组堆叠，每个 term 独立维护历史
    # buffer 形状 (frame_stack, term_dim)，dim0=0 最旧，dim0=-1 最新
    H = cfg.robot_config.frame_stack
    hist_ang_vel    = np.zeros((H,  3), dtype=np.float32)   # base_ang_vel
    hist_proj_grav  = np.zeros((H,  3), dtype=np.float32)   # projected_gravity
    hist_vel_cmd    = np.zeros((H,  3), dtype=np.float32)   # velocity_commands
    hist_joint_pos  = np.zeros((H, num_actions), dtype=np.float32)  # joint_pos_rel
    hist_joint_vel  = np.zeros((H, num_actions), dtype=np.float32)  # joint_vel_rel
    hist_actions    = np.zeros((H, num_actions), dtype=np.float32)  # last_action
    is_first_frame = True
    count_lowlevel = 0

    # 绘图数据收集
    time_data                = []
    commanded_joint_pos_data = []
    actual_joint_pos_data    = []
    tau_data                 = []
    cmd_vx_data, cmd_vy_data, cmd_dyaw_data = [], [], []
    actual_lin_vel_data      = []
    actual_ang_vel_data      = []

    start_time = time.time()

    for step in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
        # 重置请求
        if cmd.reset_requested:
            data.qpos[:] = initial_qpos
            data.qvel[:] = initial_qvel
            cmd.reset()
            hist_ang_vel.fill(0.0); hist_proj_grav.fill(0.0); hist_vel_cmd.fill(0.0)
            hist_joint_pos.fill(0.0); hist_joint_vel.fill(0.0); hist_actions.fill(0.0)
            is_first_frame = True
            mujoco.mj_forward(model, data)
            cmd.reset_requested = False

        q, dq, quat, v, omega, gvec = get_obs(data)
        q  = q[-num_actions:]    # MuJoCo 关节位置
        dq = dq[-num_actions:]   # MuJoCo 关节速度

        # 策略更新：200 Hz 物理 → 50 Hz 策略（decimation=4，与 Isaac Lab AMP 一致）
        if count_lowlevel % cfg.sim_config.decimation == 0:
            # ── MuJoCo 顺序 → Isaac Lab 顺序 ─────────────────────────────────
            q_obs  = np.zeros(num_actions, dtype=np.double)
            dq_obs = np.zeros(num_actions, dtype=np.double)
            q_rel  = q - cfg.robot_config.default_pos
            for isaac_idx, mujoco_idx in enumerate(cfg.robot_config.usd2urdf):
                q_obs[isaac_idx]  = q_rel[mujoco_idx]
                dq_obs[isaac_idx] = dq[mujoco_idx]

            # ── 当前帧各 term 观测 ────────────────────────────────────────────
            cur_ang_vel   = omega.astype(np.float32)
            cur_proj_grav = gvec.astype(np.float32)
            cur_vel_cmd   = np.array([cmd.vx, cmd.vy, cmd.dyaw], dtype=np.float32)
            cur_joint_pos = q_obs.astype(np.float32)
            cur_joint_vel = dq_obs.astype(np.float32)
            cur_action    = action.astype(np.float32)

            # ── 更新各 term 历史缓冲（oldest→newest，滑动窗口）────────────────
            # 首帧用当前值填满所有历史（对应 CircularBuffer 首次 append 的行为）
            if is_first_frame:
                hist_ang_vel[:]   = cur_ang_vel
                hist_proj_grav[:] = cur_proj_grav
                hist_vel_cmd[:]   = cur_vel_cmd
                hist_joint_pos[:] = cur_joint_pos
                hist_joint_vel[:] = cur_joint_vel
                hist_actions[:]   = cur_action
                is_first_frame = False
            else:
                hist_ang_vel   = np.roll(hist_ang_vel,   -1, axis=0); hist_ang_vel[-1]   = cur_ang_vel
                hist_proj_grav = np.roll(hist_proj_grav, -1, axis=0); hist_proj_grav[-1] = cur_proj_grav
                hist_vel_cmd   = np.roll(hist_vel_cmd,   -1, axis=0); hist_vel_cmd[-1]   = cur_vel_cmd
                hist_joint_pos = np.roll(hist_joint_pos, -1, axis=0); hist_joint_pos[-1] = cur_joint_pos
                hist_joint_vel = np.roll(hist_joint_vel, -1, axis=0); hist_joint_vel[-1] = cur_joint_vel
                hist_actions   = np.roll(hist_actions,   -1, axis=0); hist_actions[-1]   = cur_action

            # ── 按 term 分组拼接（与 IsaacLab ObsGroup flatten_history_dim 一致）
            # 每个 term: (H, dim) → (H*dim,) 展平，oldest 在前 newest 在后
            # 最终: [ang_vel×H(30), proj_grav×H(30), vel_cmd×H(30),
            #        joint_pos×H(120), joint_vel×H(120), actions×H(120)] = 450
            policy_input = np.concatenate([
                hist_ang_vel.reshape(-1),
                hist_proj_grav.reshape(-1),
                hist_vel_cmd.reshape(-1),
                hist_joint_pos.reshape(-1),
                hist_joint_vel.reshape(-1),
                hist_actions.reshape(-1),
            ]).reshape(1, -1).astype(np.float32)

            # ── 策略推理（ONNX Runtime）───────────────────────────────────────
            input_name = policy.get_inputs()[0].name
            action[:] = policy.run(None, {input_name: policy_input})[0][0]

            # ── Isaac 顺序 action → MuJoCo 顺序目标位置 ──────────────────────
            target_q = action * cfg.robot_config.action_scale
            for isaac_idx, mujoco_idx in enumerate(cfg.robot_config.usd2urdf):
                target_pos[mujoco_idx] = target_q[isaac_idx]
            target_pos_abs = target_pos + cfg.robot_config.default_pos

            print(
                f"cmd: vx={cmd.vx:.2f} vy={cmd.vy:.2f} dyaw={cmd.dyaw:.2f} | "
                f"act: vx={v[0]:.2f} vy={v[1]:.2f} wz={omega[2]:.2f}"
            )

            # 收集绘图数据
            time_data.append(step * cfg.sim_config.dt)
            commanded_joint_pos_data.append(target_pos_abs.copy())
            actual_joint_pos_data.append(q.copy())
            tau_data.append(tau.copy())
            cmd_vx_data.append(cmd.vx)
            cmd_vy_data.append(cmd.vy)
            cmd_dyaw_data.append(cmd.dyaw)
            actual_lin_vel_data.append(v[:2].copy())
            actual_ang_vel_data.append(omega[2].copy())

            # 渲染
            if headless:
                renderer.update_scene(data, camera=cam)
                base_pos = data.qpos[0:3].tolist()
                cam.lookat = [float(base_pos[0]), float(base_pos[1]), float(base_pos[2])]
                out.write(renderer.render())
            else:
                base_pos = data.qpos[0:3].tolist()
                viewer.cam.lookat = [float(base_pos[0]), float(base_pos[1]), float(base_pos[2])]
                viewer.render()
        else:
            target_pos_abs = target_pos + cfg.robot_config.default_pos

        # ── PD 控制 ───────────────────────────────────────────────────────────
        target_vel = np.zeros(num_actions, dtype=np.double)
        tau = pd_control(
            target_pos_abs, q, cfg.robot_config.kps,
            target_vel,     dq, cfg.robot_config.kds
        )
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
        data.ctrl = tau
        mujoco.mj_step(model, data)
        count_lowlevel += 1

        # 实时同步
        elapsed = time.time() - start_time
        target_t = (step + 1) * cfg.sim_config.dt
        if elapsed < target_t:
            time.sleep(target_t - elapsed)

    if headless:
        out.release()
    else:
        viewer.close()
    cmd.stop()

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    print("Simulation finished. Generating plots...")

    time_data                = np.array(time_data)
    commanded_joint_pos_data = np.array(commanded_joint_pos_data)
    actual_joint_pos_data    = np.array(actual_joint_pos_data)
    tau_data                 = np.array(tau_data)
    actual_lin_vel_data      = np.array(actual_lin_vel_data)
    actual_ang_vel_data      = np.array(actual_ang_vel_data)

    # E1 关节名称（MuJoCo 顺序）
    joint_names = [
        'L_hip_pitch', 'L_hip_roll', 'L_hip_yaw', 'L_knee', 'L_ank_pitch', 'L_ank_roll',
        'R_hip_pitch', 'R_hip_roll', 'R_hip_yaw', 'R_knee', 'R_ank_pitch', 'R_ank_roll',
    ]

    # Plot 1: 关节位置（指令 vs 实际）
    n_cols = 4
    n_rows = (num_actions + n_cols - 1) // n_cols
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharex=True)
    axes1 = axes1.flatten()
    for i in range(num_actions):
        axes1[i].plot(time_data, commanded_joint_pos_data[:, i], '--', label='Cmd')
        axes1[i].plot(time_data, actual_joint_pos_data[:, i],        label='Act')
        axes1[i].set_title(joint_names[i])
        axes1[i].set_xlabel("Time [s]")
        axes1[i].set_ylabel("Pos [rad]")
        axes1[i].legend()
        axes1[i].grid(True)
    for i in range(num_actions, len(axes1)):
        fig1.delaxes(axes1[i])
    fig1.suptitle("E1 AMP — Commanded vs Actual Joint Positions", fontsize=16)
    plt.tight_layout()
    fig1.savefig("e1_amp_joint_positions.png")

    # Plot 2: 底座速度（指令 vs 实际）
    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axes2[0].plot(time_data, cmd_vx_data,               '--', label='Cmd Vx')
    axes2[0].plot(time_data, actual_lin_vel_data[:, 0],       label='Act Vx')
    axes2[0].set_title("Base Linear Velocity X")
    axes2[0].set_ylabel("m/s"); axes2[0].legend(); axes2[0].grid(True)

    axes2[1].plot(time_data, cmd_vy_data,               '--', label='Cmd Vy')
    axes2[1].plot(time_data, actual_lin_vel_data[:, 1],       label='Act Vy')
    axes2[1].set_title("Base Linear Velocity Y")
    axes2[1].set_ylabel("m/s"); axes2[1].legend(); axes2[1].grid(True)

    axes2[2].plot(time_data, cmd_dyaw_data,             '--', label='Cmd Dyaw')
    axes2[2].plot(time_data, actual_ang_vel_data,              label='Act Dyaw')
    axes2[2].set_title("Base Angular Velocity Z")
    axes2[2].set_xlabel("Time [s]")
    axes2[2].set_ylabel("rad/s"); axes2[2].legend(); axes2[2].grid(True)

    fig2.suptitle("E1 AMP — Commanded vs Actual Base Velocities", fontsize=16)
    plt.tight_layout()
    fig2.savefig("e1_amp_base_velocities.png")
    print("Plots saved: e1_amp_joint_positions.png, e1_amp_base_velocities.png")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='E1 AMP Sim2Sim deployment script.')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to exported policy.onnx (ONNX Runtime)')
    parser.add_argument('--headless', action='store_true',
                        help='Run without GUI and save video')
    args = parser.parse_args()

    class Sim2simCfg:

        class sim_config:
            # E1_12dof.xml：含 gyro / bq 传感器，12 个关节电机
            mujoco_model_path = f'{ISAAC_DATA_DIR}/robots/droidrobot/E1/E1_12dof.xml'
            sim_duration = 2000000.0
            dt           = 0.005   # 物理步长 200 Hz（与 Isaac Lab AMP 一致）
            decimation   = 4       # 策略频率 50 Hz（decimation=4，与 Isaac Lab AMP 一致）

        class robot_config:
            # ── PD 增益（MuJoCo 顺序：左腿全部 → 右腿全部）────────────────────
            # 顺序: L_pitch L_roll L_yaw L_knee L_ank_pitch L_ank_roll  (×2)
            kps = np.array([
                150, 150, 100, 150, 20, 20,   # 左腿
                150, 150, 100, 150, 20, 20,   # 右腿
            ], dtype=np.double)
            kds = np.array([
                3,   3,   3,   5,   2,  2,    # 左腿
                3,   3,   3,   5,   2,  2,    # 右腿
            ], dtype=np.double)

            # ── 默认关节位置（MuJoCo 顺序，与 Isaac Lab init_state 一致）────────
            default_pos = np.array([
                -0.1, 0.0, 0.0, 0.2, -0.1, 0.0,   # 左腿
                -0.1, 0.0, 0.0, 0.2, -0.1, 0.0,   # 右腿
            ], dtype=np.double)

            # ── 力矩限幅（MuJoCo 顺序，来自 URDF actuatorfrcrange）───────────────
            tau_limit = np.array([
                60, 60, 36, 59.3, 59.3, 14,   # 左腿
                60, 60, 36, 59.3, 59.3, 14,   # 右腿
            ], dtype=np.double)

            # ── Isaac Lab → MuJoCo 关节索引映射 ──────────────────────────────────
            # Isaac Lab (ISAACLAB_JOINT_ORDER):
            #   [L_pitch, R_pitch, L_roll, R_roll, L_yaw, R_yaw,
            #    L_knee,  R_knee,  L_ank_pitch, R_ank_pitch, L_ank_roll, R_ank_roll]
            # MuJoCo (actuator 顺序):
            #   [L_pitch, L_roll, L_yaw, L_knee, L_ank_pitch, L_ank_roll,
            #    R_pitch, R_roll, R_yaw, R_knee, R_ank_pitch, R_ank_roll]
            # usd2urdf[isaac_idx] = mujoco_idx
            usd2urdf = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]

            # ── 观测维度（与 AmpEnvCfg.PolicyCfg 完全对应）──────────────────────
            # ang_vel(3) + projected_gravity(3) + velocity_commands(3)
            # + joint_pos_rel(12) + joint_vel_rel(12) + last_action(12) = 45
            # policy 输入 = 45 × history_length = 45 × 10 = 450
            frame_stack    = 1   # AMP PolicyCfg.history_length = 10
            num_single_obs = 45
            num_actions    = 12
            action_scale   = 0.25

        class cmd_config:
            min_vx   = -0.8
            max_vx   =  2.5
            min_vy   = -0.8
            max_vy   =  0.8
            min_dyaw = -1.0
            max_dyaw =  1.0

    policy_model = args.load_model
    if policy_model is None:
        policy_model = os.path.join(ISAAC_DATA_DIR, 'policies', 'amp', 'policy.onnx')

    policy = ort.InferenceSession(policy_model)
    run_mujoco(policy, Sim2simCfg(), args.headless)
