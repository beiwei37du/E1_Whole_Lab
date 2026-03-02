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

"""
E1 AttnEnc Sim2Sim deployment script.

Usage:
  python scripts/mujoco/sim2sim_e1_attn_enc.py
    or
  python scripts/mujoco/sim2sim_e1_attn_enc.py \
      --load_model logs/rsl_rl/e1_attn_enc/<run>/exported/policy.onnx
      --headless

Observation (per step, 45-dim):
  [0:3]   ang_vel (3)
  [3:6]   projected_gravity (3)
  [6:9]   commands: vx, vy, wz (3)
  [9:21]  joint_pos relative to default (12, Isaac Lab order)
  [21:33] joint_vel (12, Isaac Lab order)
  [33:45] last action (12, Isaac Lab order)

Policy input = concat(history_obs, perception_obs)
  history_obs   : frame_stack(5) × 45 = 225
  perception_obs: map_size[0](17) × map_size[1](11) = 187
  Total         : 225 + 187 = 412
"""

import numpy as np
import mujoco
import mujoco_viewer
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import onnxruntime as ort
import os
import cv2

from robolab.assets import ISAAC_DATA_DIR
from scripts.mujoco.keyboard import KeyboardCommand, print_keyboard_instructions


def get_obs(data):
    """Extract observations from MuJoCo data."""
    q     = data.qpos.astype(np.double)
    dq    = data.qvel.astype(np.double)
    quat  = data.sensor('bq').data[[1, 2, 3, 0]].astype(np.double)
    r     = R.from_quat(quat)
    v     = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    omega = data.sensor('gyro').data.astype(np.double)
    gvec  = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD torque calculation."""
    return (target_q - q) * kp + (target_dq - dq) * kd


def get_rays(model, data, pos, num_points, offset):
    """Cast downward rays to measure terrain height below each offset point."""
    dist    = np.zeros(num_points, dtype=np.float64)
    ray_vec = np.array([0, 0, -1], dtype=np.float64)
    geomgroup = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    geomid  = np.zeros(1, dtype=np.int32)
    for i in range(num_points):
        pt = pos.copy()
        pt[:2] += offset[i]
        pt[2]  += 20.0          # cast from well above
        dist[i] = mujoco.mj_ray(model, data, pt, ray_vec, geomgroup, 1, -1, geomid)
        dist[i] -= 20.0         # convert to height relative to robot base
    return dist


def draw_rays(scene, pos, offset, dists):
    """Visualise height-scan points in the viewer as small spheres."""
    if scene is None:
        return
    for i in range(len(offset)):
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.02, 0.02, 0.02],
            pos=[pos[0] + offset[i][0], pos[1] + offset[i][1], pos[2] - dists[i]],
            mat=np.eye(3).flatten(),
            rgba=[1, 0, 0, 1],
        )
        scene.ngeom += 1
        if scene.ngeom >= scene.maxgeom:
            break


def run_mujoco(policy, cfg, headless=False):
    """Run MuJoCo simulation with E1 AttnEnc policy."""
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
            'simulation_e1_attn_enc.mp4', fourcc,
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

    # Observation history buffer: (frame_stack × num_single_obs)
    hist_obs = np.zeros(
        (cfg.robot_config.frame_stack, cfg.robot_config.num_single_obs),
        dtype=np.float32
    )
    is_first_frame = True
    count_lowlevel = 0

    # Pre-compute body-frame ray offsets (map_size[0] × map_size[1] grid)
    nx, ny    = cfg.robot_config.map_size        # [17, 11]
    res       = cfg.robot_config.resolution      # 0.1
    num_pts   = nx * ny
    offset    = np.zeros((num_pts, 2), dtype=np.float64)
    start_x   = -(nx - 1) / 2.0 * res
    start_y   = -(ny - 1) / 2.0 * res
    for row in range(ny):
        for col in range(nx):
            offset[row * nx + col] = [start_x + col * res, start_y + row * res]

    for step in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
        # Reset on request
        if cmd.reset_requested:
            data.qpos[:] = initial_qpos
            data.qvel[:] = initial_qvel
            data.ctrl[:] = 0.0
            mujoco.mj_forward(model, data)
            cmd.reset()
            cmd.reset_requested = False
            hist_obs.fill(0.0)
            is_first_frame = True

        q, dq, quat, v, omega, gvec = get_obs(data)
        q  = q[-num_actions:]
        dq = dq[-num_actions:]

        # Policy step at control frequency
        if count_lowlevel % cfg.sim_config.decimation == 0:
            # MuJoCo order → Isaac Lab order
            q_obs  = np.zeros(num_actions, dtype=np.double)
            dq_obs = np.zeros(num_actions, dtype=np.double)
            q_rel  = q - cfg.robot_config.default_pos
            for isaac_idx, mujoco_idx in enumerate(cfg.robot_config.usd2urdf):
                q_obs[isaac_idx]  = q_rel[mujoco_idx]
                dq_obs[isaac_idx] = dq[mujoco_idx]

            # Build single-step observation (45-dim)
            obs = np.zeros((1, cfg.robot_config.num_single_obs), dtype=np.float32)
            obs[0, 0:3]  = omega          # ang_vel
            obs[0, 3:6]  = gvec           # projected_gravity
            obs[0, 6]    = cmd.vx
            obs[0, 7]    = cmd.vy
            obs[0, 8]    = cmd.dyaw
            obs[0, 9:21]  = q_obs         # joint_pos  (12, Isaac order)
            obs[0, 21:33] = dq_obs        # joint_vel  (12, Isaac order)
            obs[0, 33:45] = action        # last action (12, Isaac order)

            # Update history
            if is_first_frame:
                hist_obs       = np.tile(obs, (cfg.robot_config.frame_stack, 1))
                is_first_frame = False
            else:
                hist_obs = np.concatenate((hist_obs[1:], obs.reshape(1, -1)), axis=0)

            # Height scan — rotate offsets to world frame using current yaw
            pos = data.qpos[0:3].copy()
            r_rot  = R.from_quat(quat)
            yaw    = r_rot.as_euler('zyx')[0]
            cy, sy = np.cos(yaw), np.sin(yaw)
            rot_mat = np.array([[cy, -sy], [sy, cy]])
            world_offset = offset @ rot_mat.T
            dist = get_rays(model, data, pos, num_pts, world_offset)
            perception_obs = np.clip(
                dist - cfg.robot_config.height_scan_offset, -1.0, 1.0
            ).astype(np.float32)

            # Concatenate policy history + perception as ONNX input
            policy_input = np.concatenate(
                (hist_obs.reshape(1, -1), perception_obs.reshape(1, -1)), axis=1
            ).astype(np.float32)
            input_name  = policy.get_inputs()[0].name
            action[:]   = policy.run(None, {input_name: policy_input})[0][0]

            # Isaac order → MuJoCo order target positions
            target_q = action * cfg.robot_config.action_scale
            for isaac_idx, mujoco_idx in enumerate(cfg.robot_config.usd2urdf):
                target_pos[mujoco_idx] = target_q[isaac_idx]
            target_pos_abs = target_pos + cfg.robot_config.default_pos

            if headless:
                renderer.update_scene(data, camera=cam)
                if cmd.camera_follow:
                    cam.lookat = pos.tolist()
                draw_rays(renderer.scene, pos, world_offset, dist)
                out.write(renderer.render())
            else:
                if cmd.camera_follow:
                    viewer.cam.lookat = pos.tolist()
                viewer.render()
        else:
            target_pos_abs = target_pos + cfg.robot_config.default_pos

        # PD control
        target_vel = np.zeros(num_actions, dtype=np.double)
        tau = pd_control(
            target_pos_abs, q, cfg.robot_config.kps,
            target_vel,     dq, cfg.robot_config.kds,
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
    print("Simulation finished.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='E1 AttnEnc Sim2Sim deployment script.')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to exported policy.onnx '
                             '(e.g. logs/rsl_rl/e1_attn_enc/<run>/exported/policy.onnx)')
    parser.add_argument('--headless', action='store_true',
                        help='Render offscreen and save simulation_e1_attn_enc.mp4')
    args = parser.parse_args()

    policy_model = args.load_model
    if policy_model is None:
        policy_model = os.path.join(ISAAC_DATA_DIR, 'policies', 'direct', 'policy.onnx')

    class Sim2simCfg:

        class sim_config:
            mujoco_model_path = f'{ISAAC_DATA_DIR}/robots/droidrobot/E1/E1_12dof.xml'
            sim_duration = 2000000.0
            dt           = 0.001   # 1000 Hz physics
            decimation   = 20      # 50 Hz policy (matches Isaac Lab decimation=4 × dt=0.005)

        class robot_config:
            # ── PD gains (MuJoCo order: L then R, pitch/roll/yaw/knee/ank_p/ank_r) ──
            kps = np.array([
                150, 150, 100, 150, 20, 20,   # L leg
                150, 150, 100, 150, 20, 20,   # R leg
            ], dtype=np.double)
            kds = np.array([
                3,   3,   3,   5,   2,  2,    # L leg
                3,   3,   3,   5,   2,  2,    # R leg
            ], dtype=np.double)

            # ── Default joint positions (MuJoCo order) ───────────────────────────────
            default_pos = np.array([
                -0.1, 0.0, 0.0, 0.2, -0.1, 0.0,   # L leg
                -0.1, 0.0, 0.0, 0.2, -0.1, 0.0,   # R leg
            ], dtype=np.double)

            # ── Torque limits (MuJoCo order) ─────────────────────────────────────────
            tau_limit = np.array([
                60, 60, 36, 60, 60, 14,   # L leg
                60, 60, 36, 60, 60, 14,   # R leg
            ], dtype=np.double)

            # ── Isaac Lab (BFS) order → MuJoCo (DFS) order ──────────────────────────
            # Isaac Lab:
            #  0:L_hip_pitch  1:R_hip_pitch  2:L_hip_roll   3:R_hip_roll
            #  4:L_hip_yaw    5:R_hip_yaw    6:L_knee       7:R_knee
            #  8:L_ank_pitch  9:R_ank_pitch  10:L_ank_roll  11:R_ank_roll
            # MuJoCo:
            #  0:L_hip_pitch  1:L_hip_roll   2:L_hip_yaw    3:L_knee
            #  4:L_ank_pitch  5:L_ank_roll   6:R_hip_pitch  7:R_hip_roll
            #  8:R_hip_yaw    9:R_knee       10:R_ank_pitch 11:R_ank_roll
            # usd2urdf[isaac_idx] = mujoco_idx
            usd2urdf = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]

            num_actions    = 12
            action_scale   = 0.25
            frame_stack    = 5    # actor_obs_history_length = 5 (from E1AttnEncEnvCfg)
            num_single_obs = 45   # 3(ω) + 3(g) + 3(cmd) + 12(q) + 12(dq) + 12(a)

            # Height scan parameters (must match E1AttnEncEnvCfg)
            map_size           = [17, 11]   # cols × rows
            resolution         = 0.1
            height_scan_offset = 0.715      # E1 standing height (normalization.height_scan_offset)

    policy = ort.InferenceSession(policy_model)
    run_mujoco(policy, Sim2simCfg(), args.headless)
