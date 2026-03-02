# RoboLab — E1 双足机器人强化学习框架

基于 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 与 [RSL-RL](https://github.com/leggedrobotics/rsl_rl) 的 DroidRobot E1 双足机器人强化学习训练与部署框架，支持多种学习范式（直接 RL、AMP、运动模仿）及 MuJoCo Sim2Sim 迁移。

---

## 目录

- [项目结构](#项目结构)
- [环境安装](#环境安装)
- [整体工作流](#整体工作流)
- [训练 Train](#训练-train)
- [推理 Play](#推理-play)
- [Sim2Sim 部署](#sim2sim-部署)
- [Foxglove 可视化](#foxglove-可视化)
- [动作重定向工具](#动作重定向工具)
- [可用任务一览](#可用任务一览)
- [机器人规格](#机器人规格)

---

## 项目结构

```
E1_Whole_Lab/
├── robolab/                         # 主包
│   ├── assets/robots/               # 机器人资产配置（droidrobot.py）
│   ├── tasks/
│   │   ├── direct/                  # 直接 RL 任务
│   │   │   ├── base/                #   E1-Flat / E1-Rough
│   │   │   ├── attn_enc/            #   E1-AttnEnc
│   │   │   └── interrupt/           #   E1-Interrupt
│   │   └── manager_based/           # Manager-Based 任务
│   │       ├── amp/                 #   E1-AMP
│   │       └── beyondmimic/         #   E1-BeyondMimic
│   └── utils/keyboard.py            # Isaac Lab 键盘控制
├── scripts/
│   ├── rsl_rl/                      # Isaac Lab 训练与推理
│   │   ├── train.py                 #   训练入口
│   │   ├── play.py                  #   直接RL / AttnEnc / Interrupt 推理
│   │   ├── play_amp.py              #   AMP 推理
│   │   ├── play_bm.py               #   BeyondMimic 推理
│   │   └── cli_args.py              #   公共 CLI 参数
│   ├── mujoco/                      # Sim2Sim 部署
│   │   ├── sim2sim_e1.py            #   直接 RL 策略
│   │   ├── sim2sim_e1_amp.py        #   AMP 策略
│   │   ├── sim2sim_e1_bm.py         #   BeyondMimic 策略
│   │   ├── sim2sim_e1_attn_enc.py   #   AttnEnc 策略
│   │   ├── sim2sim_e1_interrupt.py  #   Interrupt 策略
│   │   ├── keyboard.py              #   数字键盘控制
│   │   └── foxshow_data/            #   Foxglove 录制数据（.mcap）
│   └── tools/
│       ├── foxshow.py               #   Foxglove 可视化后端
│       ├── list_envs.py             #   列出所有注册环境
│       ├── fix_pkl_numpy.py         #   修复动捕数据 numpy 兼容性
│       └── retarget/                #   GMR→Lab 动作重定向
│           ├── single_retarget.py   #     单文件转换
│           ├── dataset_retarget.py  #     批量转换
│           └── config/
│               ├── e1_12dof.yaml    #     E1 12-DOF 配置
│               └── e1_21dof.yaml    #     E1 21-DOF 配置
├── data/
│   ├── robots/droidrobot/E1/        # URDF / MuJoCo XML / STL 网格
│   │   ├── E1_12dof.urdf / .xml     #   12-DOF 腿部模型
│   │   └── E1_21dof.urdf / .xml     #   21-DOF 含手臂模型
│   ├── policies/                    # 预训练 / 导出的 ONNX 策略
│   │   ├── direct/policy.onnx
│   │   ├── amp/policy.onnx
│   │   └── bm/policy.onnx
│   └── motions/                     # 动捕数据
│       ├── e1_gmr/                  #   GMR 原始格式
│       ├── e1_lab/                  #   Lab 格式（AMP 使用）
│       └── e1_bm/                   #   BeyondMimic 格式（.npz）
├── rsl_rl/                          # RSL-RL 子模块
└── setup.py
```

---

## 环境安装

### 1. 前置依赖

请先按官方文档安装：

- **NVIDIA Isaac Sim** (≥ 4.5)：[安装指南](https://docs.isaacsim.omniverse.nvidia.com/)
- **Isaac Lab**：[安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/)

安装完成后验证：

```bash
python -c "import isaaclab; print('Isaac Lab OK')"
```

### 2. 克隆仓库

```bash
git clone <repo_url> E1_Whole_Lab
cd E1_Whole_Lab
```

### 3. 安装本项目

```bash
# 在项目根目录执行
pip install -e .
```

安装的依赖包（`setup.py`）：

| 包 | 版本 | 用途 |
|---|---|---|
| `mujoco` | `==3.3.3` | Sim2Sim 物理仿真 |
| `mujoco-python-viewer` | 最新 | MuJoCo 渲染窗口 |
| `psutil` | 最新 | 系统资源监控 |
| `joblib` | `>=1.2.0` | 动捕数据加载 |
| `pynput` | 最新 | 键盘监听 |

### 4. 安装 RSL-RL 子模块

```bash
pip install -e rsl_rl
```

验证版本（需 ≥ 3.0.1）：

```bash
python -c "import importlib.metadata; print(importlib.metadata.version('rsl-rl-lib'))"
```

### 5. 安装 Foxglove 可视化依赖（可选）

Sim2Sim 中使用 `--foxshow` 参数时需要：

```bash
pip install foxglove-sdk yourdfpy scipy
```

---

## 整体工作流

```
┌─────────────┐    ┌──────────────┐    ┌──────────────────────────┐
│  train.py   │───▶│   play*.py   │───▶│  data/policies/*/        │
│ Isaac Lab   │    │ 导出 ONNX    │    │  policy.onnx（自动同步） │
│ 训练策略    │    │ 并自动同步   │    └────────────┬─────────────┘
└─────────────┘    └──────────────┘                 │
                                                     ▼
                                         ┌───────────────────────┐
                                         │  scripts/mujoco/      │
                                         │  sim2sim_e1*.py       │
                                         │  MuJoCo 物理部署      │
                                         └───────────────────────┘
```

**play 脚本导出路径映射：**

| play 脚本 | 任务 | 自动同步至 |
|---|---|---|
| `play.py` | `E1-Flat` / `E1-Rough` | `data/policies/direct/policy.onnx` |
| `play.py` | `E1-AttnEnc` | `data/policies/attn_enc/policy.onnx` |
| `play.py` | `E1-Interrupt` | `data/policies/interrupt/policy.onnx` |
| `play_amp.py` | `E1-AMP-Play` | `data/policies/amp/policy.onnx` |
| `play_bm.py` | `E1-BeyondMimic` | `data/policies/bm/policy.onnx` |

---

## 训练 Train

所有训练在 **项目根目录** 执行，通过 Isaac Lab 的 Python 启动：

```bash
cd E1_Whole_Lab
python scripts/rsl_rl/train.py --task <TASK_NAME> [OPTIONS]
```

### 训练命令

```bash
# 平地行走（Direct RL）
python scripts/rsl_rl/train.py --task E1-Flat --num_envs 4096 --headless

# 复杂地形行走（Direct RL）
python scripts/rsl_rl/train.py --task E1-Rough --num_envs 4096 --headless

# 注意力编码器（AttnEnc）
python scripts/rsl_rl/train.py --task E1-AttnEnc --num_envs 4096 --headless

# 中断控制（Interrupt）
python scripts/rsl_rl/train.py --task E1-Interrupt --num_envs 4096 --headless

# 对抗运动先验（AMP）
python scripts/rsl_rl/train.py --task E1-AMP --num_envs 8192 --headless

# BeyondMimic 运动模仿
python scripts/rsl_rl/train.py --task E1-BeyondMimic --num_envs 4096 --headless
```

### 从断点继续训练

```bash
python scripts/rsl_rl/train.py --task E1-Flat \
    --resume --load_run xxx-xxx-xxx
```

### 训练参数说明

| 参数 | 说明 | 示例 |
|---|---|---|
| `--task` | 任务名（必填） | `E1-Flat` |
| `--num_envs` | 并行环境数 | `4096` |
| `--headless` | 无头模式（不显示 GUI） | — |
| `--max_iterations` | 最大训练迭代次数 | `5000` |
| `--seed` | 随机种子（`-1` 随机） | `42` |
| `--resume` | 从断点继续训练 | — |
| `--load_run` | 要恢复的运行目录名 | `2025-01-01_12-00-00` |
| `--checkpoint` | 指定 checkpoint 文件名 | `model_1000.pt` |
| `--distributed` | 多 GPU 分布式训练 | — |
| `--video` | 训练时录制视频 | — |
| `--logger` | 日志后端 | `wandb` / `tensorboard` |
| `--log_project_name` | wandb / neptune 项目名 | `e1-locomotion` |

训练日志保存在：`logs/rsl_rl/<experiment_name>/<timestamp>/`

---

## 推理 Play

Play 脚本同时完成两件事：
1. 在 Isaac Lab 中渲染策略效果
2. **自动导出 ONNX 并同步到 `data/policies/` 对应目录**

### 直接 RL / AttnEnc / Interrupt

```bash
# 使用最新 checkpoint（有 GUI）
python scripts/rsl_rl/play.py --task E1-Flat --num_envs 4

# 指定运行目录和 checkpoint
python scripts/rsl_rl/play.py --task E1-Flat \
    --load_run 2025-01-01_12-00-00 --checkpoint model_1000.pt

# 平坦地形模式
python scripts/rsl_rl/play.py --task E1-Rough --plane

# 键盘控制（单环境）
python scripts/rsl_rl/play.py --task E1-Flat --keyboard

# 实时步进（对齐物理时间）
python scripts/rsl_rl/play.py --task E1-Flat --num_envs 1 --real-time
```

### AMP

```bash
python scripts/rsl_rl/play_amp.py --task E1-AMP-Play --num_envs 4

python scripts/rsl_rl/play_amp.py --task E1-AMP-Play \
    --load_run 2025-01-01_12-00-00 --checkpoint model_1000.pt
```

### BeyondMimic

```bash
python scripts/rsl_rl/play_bm.py --task E1-BeyondMimic --num_envs 4
```

导出文件会保存在：
```
logs/rsl_rl/<experiment_name>/<run>/exported/
├── policy.onnx   ← 同时自动复制到 data/policies/<subdir>/
└── policy.pt
```

---

## Sim2Sim 部署

Sim2Sim 在 **MuJoCo** 中加载 ONNX 策略运行，无需 Isaac Sim / Isaac Lab。

**控制参数：**
- 仿真频率：1000 Hz（`dt = 0.001 s`）
- 策略频率：50 Hz（decimation = 20）
- 观测叠帧：10 帧（输入维度 = 10 × 45 = 450）

### 直接 RL 策略

```bash
# 默认加载 data/policies/direct/policy.onnx
python scripts/mujoco/sim2sim_e1.py

# 指定策略文件
python scripts/mujoco/sim2sim_e1.py --load_model data/policies/direct/policy.onnx

# 无头录制（输出 simulation_e1.mp4）
python scripts/mujoco/sim2sim_e1.py --headless

# 启用 Foxglove 实时可视化
python scripts/mujoco/sim2sim_e1.py --foxshow
```

### AMP 策略

```bash
python scripts/mujoco/sim2sim_e1_amp.py
python scripts/mujoco/sim2sim_e1_amp.py --load_model data/policies/amp/policy.onnx
python scripts/mujoco/sim2sim_e1_amp.py --headless
```

### BeyondMimic 策略

```bash
python scripts/mujoco/sim2sim_e1_bm.py
python scripts/mujoco/sim2sim_e1_bm.py --load_model data/policies/bm/policy.onnx
python scripts/mujoco/sim2sim_e1_bm.py --headless
```

### AttnEnc 策略

```bash
python scripts/mujoco/sim2sim_e1_attn_enc.py
python scripts/mujoco/sim2sim_e1_attn_enc.py --headless
```

### Interrupt 策略

```bash
python scripts/mujoco/sim2sim_e1_interrupt.py
python scripts/mujoco/sim2sim_e1_interrupt.py --headless
```

### 键盘控制（数字键盘风格）

Sim2Sim 启动后通过键盘实时控制速度指令，步进量为 0.1：

| 按键 | 功能 | 范围 |
|---|---|---|
| `8` | 增加前进速度 Vx | −0.8 ~ 2.5 m/s |
| `2` | 减少前进速度 Vx | −0.8 ~ 2.5 m/s |
| `4` | 增加左移速度 Vy | −0.8 ~ 0.8 m/s |
| `6` | 减少左移速度 Vy | −0.8 ~ 0.8 m/s |
| `7` | 增加左转角速度 dYaw | −1.0 ~ 1.0 rad/s |
| `9` | 减少左转角速度 dYaw | −1.0 ~ 1.0 rad/s |
| `0` | 重置所有速度指令并复位机器人 | — |

> 当前速度会实时打印到终端：`vx: 0.30, vy: 0.00, dyaw: 0.00`

### Sim2Sim 运行完成后的输出

仿真结束后自动生成对比图表（保存至当前目录）：

| 文件 | 内容 |
|---|---|
| `e1_joint_positions.png` | 12 个关节的指令位置 vs 实际位置 |
| `e1_base_velocities.png` | Vx / Vy / dYaw 指令 vs 实际速度 |

### PD 控制增益（直接 RL）

| 关节组 | Kp | Kd | 力矩限幅 |
|---|---|---|---|
| 髋俯仰 (Hip Pitch) | 150 | 3 | 60 Nm |
| 髋滚动 (Hip Roll) | 150 | 3 | 60 Nm |
| 髋偏转 (Hip Yaw) | 100 | 3 | 36 Nm |
| 膝关节 (Knee) | 150 | 5 | 60 Nm |
| 踝俯仰 (Ankle Pitch) | 20 | 2 | 60 Nm |
| 踝滚动 (Ankle Roll) | 20 | 2 | 14 Nm |

---

## Foxglove 可视化

[Foxglove](https://foxglove.dev/) 用于实时可视化机器人三维状态，录制数据保存为 MCAP 格式。

### 安装

```bash
pip install foxglove-sdk yourdfpy scipy
```

### 启用方式

在任意 Sim2Sim 脚本中加 `--foxshow`：

```bash
python scripts/mujoco/sim2sim_e1.py --foxshow
```

启动后自动：
1. 开启 Foxglove WebSocket 服务（`ws://localhost:8765`）
2. 加载 E1 URDF 并发布坐标树 `/tf`
3. 实时发布 `/joint_states`（关节位置/速度）和 `/joint_target`（目标位置）
4. 数据录制至 `scripts/mujoco/foxshow_data/e1_YYMMDD_HHMMSS.mcap`

### 连接 Foxglove Studio

1. 下载 [Foxglove Studio](https://foxglove.dev/download) 或打开 [app.foxglove.dev](https://app.foxglove.dev/)
2. **Open connection** → **Foxglove WebSocket** → 填入 `ws://localhost:8765`
3. 添加 **3D 面板** 即可看到机器人实时三维状态

### 回放 MCAP 录制文件

1. Foxglove Studio → **Open local file**
2. 选择 `scripts/mujoco/foxshow_data/*.mcap`

---

## 动作重定向工具

将外部 GMR 格式动捕数据转换为 Isaac Lab 可用的 `.pkl` 格式，用于 AMP / BeyondMimic 训练。

### 单文件转换

```bash
python scripts/tools/retarget/single_retarget.py \
    --robot e1 \
    --input_file  data/motions/e1_gmr/walk.pkl \
    --output_file data/motions/e1_lab/walk.pkl \
    --config_file scripts/tools/retarget/config/e1_12dof.yaml \
    --headless

# 指定帧范围并设置循环模式
python scripts/tools/retarget/single_retarget.py \
    --robot e1 \
    --input_file  data/motions/e1_gmr/walk.pkl \
    --output_file data/motions/e1_lab/walk_clip.pkl \
    --config_file scripts/tools/retarget/config/e1_12dof.yaml \
    --frame_range 10 100 --loop wrap --headless
```

### 批量转换

```bash
python scripts/tools/retarget/dataset_retarget.py \
    --robot e1 \
    --input_dir  data/motions/e1_gmr/ \
    --output_dir data/motions/e1_lab/ \
    --config_file scripts/tools/retarget/config/e1_12dof.yaml \
    --loop clamp
```

### 修复动捕数据 numpy 兼容性

若 `.pkl` 文件因 numpy 版本差异加载失败，运行：

```bash
python scripts/tools/fix_pkl_numpy.py
# 自动处理 data/motions/e1_lab/ 下所有 .pkl 文件
```

---

## 可用任务一览

```bash
# 列出所有已注册环境
python scripts/tools/list_envs.py
```

| 任务名称 | 类型 | play 脚本 | 说明 | 推荐并行环境数 |
|---|---|---|---|---|
| `E1-Flat` | Direct RL | `play.py` | 平地行走 | 4096 |
| `E1-Rough` | Direct RL | `play.py` | 复杂地形行走 | 4096 |
| `E1-AttnEnc` | Direct RL | `play.py` | 注意力编码器 | 4096 |
| `E1-Interrupt` | Direct RL | `play.py` | 中断控制 | 4096 |
| `E1-AMP` | Manager-Based | `play_amp.py` | 对抗运动先验 | 8192 |
| `E1-AMP-Play` | Manager-Based | `play_amp.py` | AMP 推理专用 | — |
| `E1-BeyondMimic` | Manager-Based | `play_bm.py` | BeyondMimic 运动模仿 | 4096 |

---

## 机器人规格

**DroidRobot E1**

| 参数 | 值 |
|---|---|
| 自由度 | 12 DOF（腿部），另有 21 DOF 含手臂版本 |
| 质量 | 约 26 kg |
| 关节分布 | 左/右各：髋俯仰、髋滚动、髋偏转、膝关节、踝俯仰、踝滚动 |
| 默认站立姿态 | `[−0.1, 0, 0, 0.2, −0.1, 0]`（左/右腿，弧度） |

**观测空间（45 维，叠帧 10 次，最终输入 450 维）：**

| 维度 | 内容 |
|---|---|
| 0–2 | 角速度 ω（机体系） |
| 3–5 | 重力向量投影 |
| 6–8 | 速度指令（Vx, Vy, dYaw） |
| 9–20 | 关节位置（Isaac Lab 顺序，12 维） |
| 21–32 | 关节速度（12 维） |
| 33–44 | 上一步动作（12 维） |

**关节顺序对比：**

| Isaac Lab 顺序 | MuJoCo 顺序 |
|---|---|
| L_pitch, R_pitch, L_roll, R_roll, ... | L_pitch, L_roll, L_yaw, L_knee, ... |
| 同侧交替排列 | 同腿连续排列 |

---

## 许可证

Copyright (c) 2022–2025, The Isaac Lab Project Developers.
Copyright (c) 2025–2026, The RoboLab Project Developers.
Licensed under [BSD-3-Clause](LICENSE).
