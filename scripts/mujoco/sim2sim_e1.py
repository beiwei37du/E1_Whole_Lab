import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from robolab.assets import ISAAC_DATA_DIR
import onnxruntime as ort
import os
import cv2
import matplotlib.pyplot as plt
from scripts.mujoco.keyboard import KeyboardCommand, print_keyboard_instructions
from scripts.tools.foxshow import FoxShow

"""

"""

def get_obs(data):
    """从 MuJoCo data 中提取观测量。"""
    q    = data.qpos.astype(np.double)
    dq   = data.qvel.astype(np.double)
    quat = data.sensor('bq').data[[1, 2, 3, 0]].astype(np.double)
    r    = R.from_quat(quat)
    v    = r.apply(data.qvel[:3], inverse=True).astype(np.double)   # 机体系线速度
    omega = data.sensor('gyro').data.astype(np.double)
    gvec  = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD 力矩计算。"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def run_mujoco(policy, cfg, headless=False, foxshow=False):
    print_keyboard_instructions()
    cmd = KeyboardCommand()
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
            'simulation_e1.mp4', fourcc,
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

    if foxshow:
        urdf_path = cfg.sim_config.mujoco_model_path.replace('.xml', '.urdf')
        fox = FoxShow(urdf_path)
        fox_joint_names = fox.get_joint_names()
        mujoco_joint_names = [model.joint(i).name for i in range(1, 1 + num_actions)]
        urdf_to_mujoco = [mujoco_joint_names.index(name) for name in fox_joint_names]

    target_pos  = np.zeros(num_actions, dtype=np.double)
    action      = np.zeros(num_actions, dtype=np.double)
    tau         = np.zeros(num_actions, dtype=np.double)

    # 历史观测缓冲（frame_stack × num_single_obs）
    hist_obs = np.zeros(
        (cfg.robot_config.frame_stack, cfg.robot_config.num_single_obs),
        dtype=np.float32
    )
    is_first_frame = True
    count_lowlevel = 0

    # 绘图数据收集
    time_data               = []
    commanded_joint_pos_data = []
    actual_joint_pos_data    = []
    tau_data                 = []
    cmd_vx_data, cmd_vy_data, cmd_dyaw_data = [], [], []
    actual_lin_vel_data      = []
    actual_ang_vel_data      = []

    for step in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
        if cmd.reset_requested:
            data.qpos[:] = initial_qpos
            data.qvel[:] = initial_qvel
            cmd.reset()
            mujoco.mj_forward(model, data)
            cmd.reset_requested = False

        q, dq, quat, v, omega, gvec = get_obs(data)
        q  = q[-num_actions:]    # 取关节位置（MuJoCo 顺序）
        dq = dq[-num_actions:]   # 取关节速度（MuJoCo 顺序）

        # 策略频率：1000Hz → 50Hz（decimation=20）
        if count_lowlevel % cfg.sim_config.decimation == 0:
            # ── 将 MuJoCo 顺序转换为 Isaac Lab（USD）顺序 ──────────────────
            q_obs  = np.zeros(num_actions, dtype=np.double)
            dq_obs = np.zeros(num_actions, dtype=np.double)
            q_rel  = q - cfg.robot_config.default_pos   # 相对于默认位姿的偏差
            for isaac_idx, mujoco_idx in enumerate(cfg.robot_config.usd2urdf):
                q_obs[isaac_idx]  = q_rel[mujoco_idx]
                dq_obs[isaac_idx] = dq[mujoco_idx]

            # ── 组装单步观测（45 维）────────────────────────────────────────
            obs = np.zeros((1, cfg.robot_config.num_single_obs), dtype=np.float32)
            obs[0, 0:3]  = omega         # ang_vel
            obs[0, 3:6]  = gvec          # projected_gravity
            obs[0, 6]    = cmd.vx
            obs[0, 7]    = cmd.vy
            obs[0, 8]    = cmd.dyaw
            obs[0, 9:21]  = q_obs        # joint_pos  (12, Isaac 顺序)
            obs[0, 21:33] = dq_obs       # joint_vel  (12, Isaac 顺序)
            obs[0, 33:45] = action       # last action (12, Isaac 顺序)

            # ── 更新历史缓冲 ─────────────────────────────────────────────────
            if is_first_frame:
                hist_obs       = np.tile(obs, (cfg.robot_config.frame_stack, 1))
                is_first_frame = False
            else:
                hist_obs = np.concatenate((hist_obs[1:], obs.reshape(1, -1)), axis=0)

            # ── 策略推理 ─────────────────────────────────────────────────────
            policy_input = hist_obs.reshape(1, -1).astype(np.float32)
            input_name = policy.get_inputs()[0].name
            action[:] = policy.run(None, {input_name: policy_input})[0][0]

            # ── 将 Isaac 顺序的 action 转换为 MuJoCo 顺序的目标位置 ─────────
            target_q = action * cfg.robot_config.action_scale
            for isaac_idx, mujoco_idx in enumerate(cfg.robot_config.usd2urdf):
                target_pos[mujoco_idx] = target_q[isaac_idx]
            target_pos_abs = target_pos + cfg.robot_config.default_pos

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

            if foxshow:
                tic = int(step * cfg.sim_config.dt * 1000)
                fox.update_robot_state(
                    tic=tic,
                    pos=q[urdf_to_mujoco],
                    vel=dq[urdf_to_mujoco],
                    trans=data.xpos[model.body('pelvis').id].tolist(),
                    quat=quat,
                )
                fox.update_robot_target(tic=tic, pos=target_pos_abs[urdf_to_mujoco])

            if headless:
                renderer.update_scene(data, camera=cam)
                out.write(renderer.render())
            else:
                viewer.render()
        else:
            target_pos_abs = target_pos + cfg.robot_config.default_pos

        # ── PD 控制 ──────────────────────────────────────────────────────────
        target_vel = np.zeros(num_actions, dtype=np.double)
        tau = pd_control(
            target_pos_abs, q, cfg.robot_config.kps,
            target_vel,     dq, cfg.robot_config.kds
        )
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
        data.ctrl = tau
        mujoco.mj_step(model, data)
        count_lowlevel += 1

    if headless:
        out.release()
    else:
        viewer.close()
    cmd.stop()
    if foxshow:
        fox.stop_server()

    # ── 绘图 ─────────────────────────────────────────────────────────────────
    print("Simulation finished. Generating plots...")

    time_data                = np.array(time_data)
    commanded_joint_pos_data = np.array(commanded_joint_pos_data)
    actual_joint_pos_data    = np.array(actual_joint_pos_data)
    tau_data                 = np.array(tau_data)
    actual_lin_vel_data      = np.array(actual_lin_vel_data)
    actual_ang_vel_data      = np.array(actual_ang_vel_data)

    # E1 关节名称（MuJoCo 顺序）
    joint_names_mujoco = [
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
        axes1[i].set_title(joint_names_mujoco[i])
        axes1[i].set_xlabel("Time [s]")
        axes1[i].set_ylabel("Pos [rad]")
        axes1[i].legend()
        axes1[i].grid(True)
    for i in range(num_actions, len(axes1)):
        fig1.delaxes(axes1[i])
    fig1.suptitle("E1 — Commanded vs Actual Joint Positions", fontsize=16)
    plt.tight_layout()
    fig1.savefig("e1_joint_positions.png")

    # Plot 2: 底座速度（指令 vs 实际）
    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axes2[0].plot(time_data, cmd_vx_data,                  '--', label='Cmd Vx')
    axes2[0].plot(time_data, actual_lin_vel_data[:, 0],          label='Act Vx')
    axes2[0].set_title("Base Linear Velocity X")
    axes2[0].set_ylabel("m/s"); axes2[0].legend(); axes2[0].grid(True)

    axes2[1].plot(time_data, cmd_vy_data,                  '--', label='Cmd Vy')
    axes2[1].plot(time_data, actual_lin_vel_data[:, 1],          label='Act Vy')
    axes2[1].set_title("Base Linear Velocity Y")
    axes2[1].set_ylabel("m/s"); axes2[1].legend(); axes2[1].grid(True)

    axes2[2].plot(time_data, cmd_dyaw_data,                '--', label='Cmd Dyaw')
    axes2[2].plot(time_data, actual_ang_vel_data,                label='Act Dyaw')
    axes2[2].set_title("Base Angular Velocity Z")
    axes2[2].set_xlabel("Time [s]")
    axes2[2].set_ylabel("rad/s"); axes2[2].legend(); axes2[2].grid(True)

    fig2.suptitle("E1 — Commanded vs Actual Base Velocities", fontsize=16)
    plt.tight_layout()
    fig2.savefig("e1_base_velocities.png")
    print("Plots saved: e1_joint_positions.png, e1_base_velocities.png")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='E1 Sim2Sim deployment script.')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to exported policy.onnx')
    parser.add_argument('--headless',   action='store_true', help='Run without GUI and save video')
    parser.add_argument('--foxshow',    action='store_true', help='Enable Foxglove real-time visualization')
    args = parser.parse_args()

    class Sim2simCfg:

        class sim_config:
            mujoco_model_path = f'{ISAAC_DATA_DIR}/robots/droidrobot/E1/E1_12dof.xml'
            sim_duration = 2000000
            dt           = 0.001     # MuJoCo 物理步长 1000 Hz
            decimation   = 20        # 策略频率 50 Hz（与 Isaac Lab 一致）

        class robot_config:
            # ── PD 增益（MuJoCo 关节顺序：左腿全部 → 右腿全部）────────────
            # 顺序: L_pitch L_roll L_yaw L_knee L_ank_pitch L_ank_roll  (×2)
            kps = np.array([
                150, 150, 100, 150, 20, 20,   # 左腿
                150, 150, 100, 150, 20, 20,   # 右腿
            ], dtype=np.double)
            kds = np.array([
                3,   3,   3,   5,   2,  2,    # 左腿
                3,   3,   3,   5,   2,  2,    # 右腿
            ], dtype=np.double)

            # ── 默认关节位置（MuJoCo 顺序）──────────────────────────────────
            default_pos = np.array([
                -0.1, 0.0, 0.0, 0.2, -0.1, 0.0,   # 左腿
                -0.1, 0.0, 0.0, 0.2, -0.1, 0.0,   # 右腿
            ], dtype=np.double)

            # ── 力矩限幅（MuJoCo 顺序）──────────────────────────────────────
            tau_limit = np.array([
                60, 60, 36, 60, 60, 14,   # 左腿
                60, 60, 36, 60, 60, 14,   # 右腿
            ], dtype=np.double)

            # ── Isaac Lab → MuJoCo 关节索引映射 ─────────────────────────────
            # Isaac Lab : [L_pitch, R_pitch, L_roll, R_roll, ...]
            # MuJoCo    : [L_pitch, L_roll,  L_yaw,  L_knee, ...]
            # usd2urdf[isaac_idx] = mujoco_idx
            usd2urdf = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]

            # ── 观测维度 ────────────────────────────────────────────────────
            frame_stack    = 10
            num_single_obs = 45   # 3(ω) + 3(g) + 3(cmd) + 12(q) + 12(dq) + 12(a)
            num_actions    = 12
            action_scale   = 0.25

    policy_model = args.load_model
    if policy_model is None:
        policy_model = os.path.join(ISAAC_DATA_DIR, 'policies', 'direct', 'policy.onnx')

    policy = ort.InferenceSession(policy_model)
    run_mujoco(policy, Sim2simCfg(), args.headless, args.foxshow)
