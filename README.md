# Biped Motion Imitation with Deep Reinforcement Learning

Training a 12-DOF biped robot (MiniPi) to imitate reference walking motions using deep RL in Isaac Gym, with sim-to-sim transfer to MuJoCo.

<p align="center">
  <img src="animRL/resources/images/walk.gif" width="350" alt="Reference walking motion">
</p>

## Highlights

- **4,096 parallel Isaac Gym envs**, ~98k transitions per PPO update, ~393M total environment steps
- **6 multiplicative Gaussian-kernel reward terms**, 0.87 mean per-step reward at convergence
- **215-dim observation** (43 features × 5-frame history), onboard-only signals — no privileged state
- **Adaptive KL-based LR scheduling**, auto-adjusts between 1e-4 and 1e-2
- **Multi-modal domain randomization**: friction, mass, pushes, observation noise, action delay
- **Sim-to-sim transfer**: batched 6-robot MuJoCo inference with explicit PD control

## Overview

This project implements a **DeepMimic**-style motion imitation pipeline for a small bipedal robot. A control policy is trained via **Proximal Policy Optimization (PPO)** to track a reference walking clip, then hardened with domain randomization so the learned behavior transfers across simulators.

<p align="center">
  <img src="animRL/resources/images/minipi.png" width="280" alt="MiniPi robot">
</p>

### Pipeline

| Stage | Description |
|-------|-------------|
| **1. Motion Imitation** | Multiplicative reward shaping (joint tracking, base height/orientation/velocity, end-effector position, action smoothness) drives the policy to replicate a reference walk cycle. |
| **2. Observation Redesign** | The policy is retrained using only onboard-available signals (projected gravity, angular velocity, joint states, action history) plus a 5-step observation history — removing privileged state like ground-truth yaw and linear velocity. |
| **3. Domain Randomization & Sim2Sim** | Friction, base mass, and random external pushes are varied during training. The final policy is evaluated in MuJoCo under perturbed physical parameters. |

## Training Monitoring

Training is monitored in real time using [Weights & Biases](https://wandb.ai). The W&B dashboard provides four panels:

| Panel | What it shows |
|-------|---------------|
| **Learn** | Policy/value loss, learning rate schedule, action standard deviation. A healthy run shows decreasing loss and std over time. |
| **Train** | Mean episode length and cumulative episode reward. Episode length should plateau as the policy learns to survive full cycles. |
| **Episode** | Per-step average of each reward term and the total reward. Useful for diagnosing which reward component is lagging. |
| **Media** | Periodically recorded matplotlib animations of the robot during training, allowing visual inspection of policy progress. |

<p align="center">
  <img src="animRL/resources/images/learning_curve.png" width="600" alt="W&B learning curves">
</p>

## Results

Each stage is evaluated by running a full episode in Isaac Gym. The evaluation script (`eval.py`) produces:
- **animation.mp4** — a matplotlib animation of the robot replaying the learned policy
- **eval_rewards.png** — per-step reward breakdown across all reward terms
- **eval_buf.json** — raw observations, actions, rewards, and done flags for further analysis

The GIFs below are converted from these evaluation animations.

### Stage 1 — Motion Imitation (Isaac Gym, full state)

<p align="center">
  <img src="assets/stage1_imitation.gif" width="400" alt="Stage 1: motion imitation">
</p>

With full simulator state available (base velocity, orientation quaternion, height), all reward terms converge close to 1.0 — accurate tracking of joint angles, base height, orientation, velocity, and end-effector positions. **Mean per-step reward: 0.846.**

<p align="center">
  <img src="assets/rewards_stage1.png" width="500" alt="Stage 1 reward curves">
</p>

### Stage 2 — Onboard Observation Only

<p align="center">
  <img src="assets/stage2_onboard_obs.gif" width="400" alt="Stage 2: onboard observation">
</p>

Privileged signals (ground-truth yaw, linear velocity) are removed. The policy relies on projected gravity, angular velocity, joint states, and a 5-step observation history to infer the missing state. Despite the reduced information, imitation quality remains high. **Mean per-step reward: 0.862.**

<p align="center">
  <img src="assets/rewards_stage2.png" width="500" alt="Stage 2 reward curves">
</p>

### Stage 3 — Domain Randomization + Sim-to-Sim Transfer

<p align="center">
  <img src="assets/stage3_domain_rand.gif" width="400" alt="Stage 3: domain randomization">
</p>

Domain randomization (friction, mass, random pushes) introduces noisier per-step rewards during evaluation, but the policy learns a more conservative and robust gait. DR acts as a regularizer, improving the final reward despite added training noise. **Mean per-step reward: 0.873.** This is the key stage enabling successful transfer — the same checkpoint is deployed in MuJoCo with multiple robot instances under perturbed physical parameters, and the robot maintains stable walking.

<p align="center">
  <img src="assets/rewards_stage3.png" width="500" alt="Stage 3 reward curves">
</p>

## Method Details

### Reward Design

Each reward term uses an exponential kernel on the L2 error, combined multiplicatively:

$$r = \prod_i \exp\!\Bigl(-\frac{\max(0,\,\|e_i\| - \tau_i)^2}{\sigma_i}\Bigr)$$

| Reward Term | Tracks | σ |
|---|---|---|
| `track_base_height` | Reference CoM height | 0.1 |
| `track_joint_pos` | Reference joint angles | 1.8 |
| `track_base_orientation` | Reference base quaternion | 0.5 |
| `track_base_vel` | Reference base linear velocity | 1.0 |
| `track_ee_pos` | Reference end-effector (foot) positions | 0.25 |
| `joint_targets_rate` | Action smoothness (penalizes large changes) | 5.0 |

### Phase Variable

A `[0, 1]` phase signal indexes into the motion clip, providing the target reference frame at each timestep. The phase increments each step and wraps cyclically for continuous walking.

### Observation Space

**Deployment-ready observation** (Stage 2 & 3):

| Signal | Dim |
|---|---|
| Projected gravity | 3 |
| Base angular velocity | 3 |
| Joint angle offsets | 12 |
| Joint velocities | 12 |
| Previous action | 12 |
| Phase variable | 1 |
| Observation history (5 steps) | 5 x 43 |

### Network Architecture

- **Actor**: `215 → 512 → ELU → 256 → ELU → 12` + learned diagonal Gaussian log-std (init −1.0)
- **Critic**: `215 → 512 → ELU → 256 → ELU → 1`
- Running observation normalization
- **PPO**: γ = 0.99, λ = 0.95, clip = 0.2, entropy coeff = 0.01, 5 epochs, 4 mini-batches, grad clip 1.0

### Domain Randomization (Stage 3)

| Parameter | Randomization |
|---|---|
| Friction coefficient | Uniform [0.6, 1.2] per env |
| Base mass | Additive offset ∈ [−0.5, +0.5] kg |
| External pushes | ±0.3 m/s linear, ±0.3 rad/s angular, every 5 s |
| Observation noise | σ = 0.01 additive Gaussian |
| Action delay | Random interpolation with previous action |

## Sim-to-Sim Transfer

The trained policy is deployed zero-shot from Isaac Gym to MuJoCo via `sim2sim.py`:

- **Batched inference**: 6 robots in a single MuJoCo scene, each starting at a different walk-cycle phase (reference state initialization)
- **PD control**: per-joint Kp/Kd gains (e.g., hip_pitch: 50/0.8, ankle_roll: 25/0.5), torque clipped to ±20 Nm
- **Quaternion convention**: XYZW (Isaac Gym) ↔ WXYZ (MuJoCo) conversion handled at the interface
- **Action mapping**: `target_q = default_q + clip(action × 0.2, −1, 1)`, policy dt = 0.02 s (4× decimation at 0.005 s sim step)

## Extensibility

The framework supports arbitrary reference motions — a `Jump.txt` clip is included alongside the walk clip. Adding new motions requires only a motion file and a config entry; the reward functions and training loop are motion-agnostic by design.

## Project Structure

```
animRL/
├── cfg/mimic/           # Training configs (reward weights, DR params)
├── dataloader/          # Motion clip loader & phase utilities
├── envs/mimic/          # Isaac Gym environments
│   ├── mimic_task.py    # Full-state observation (Stage 1)
│   └── mimic_hw_task.py # Onboard observation (Stage 2 & 3)
├── rewards/             # Reward function implementations
├── scripts/
│   ├── train.py         # PPO training entry point
│   ├── eval.py          # Policy evaluation & video export
│   └── sim2sim.py       # MuJoCo sim-to-sim transfer
├── resources/
│   └── datasets/pi/     # Reference motion data (Walk.txt)
└── results/             # Trained models & evaluation artifacts
    ├── task-1/          # Stage 1 checkpoint + eval
    ├── task-2/          # Stage 2 checkpoint + eval
    └── task-3/          # Stage 3 checkpoint + eval
```

## Getting Started

### Prerequisites

- Python 3.8
- [Isaac Gym](https://developer.nvidia.com/isaac-gym) (requires NVIDIA GPU)
- [Poetry](https://python-poetry.org/) for dependency management

### Installation

```bash
poetry env use python3.8
poetry install
```

For GPU training inside Docker (with Isaac Gym pre-installed):

```bash
bash install.sh
```

### Training

```bash
# Stage 1 – motion imitation (full state)
python animRL/scripts/train.py --task=walk --dv

# Stage 2 – onboard observation
python animRL/scripts/train.py --task=walk-hw --dv

# Stage 3 – domain randomization (configure in walk_hw_config.py)
python animRL/scripts/train.py --task=walk-hw --dv
```

Add `--wb` to enable [Weights & Biases](https://wandb.ai) logging.

### Evaluation

```bash
# Evaluate in Isaac Gym
python animRL/scripts/eval.py --task=walk-hw --load_run=<run_id> --checkpoint=<iter>

# Sim-to-sim transfer (MuJoCo, runs locally without GPU)
python animRL/scripts/sim2sim.py --load_run=<run_name>
```

### Pre-trained Models

Pre-trained checkpoints for all three stages are available in `animRL/results/task-{1,2,3}/model.pt`.

## References

- Peng, Xue Bin, et al. *"DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills."* ACM Transactions on Graphics (TOG) 37.4 (2018): 1-14.
- Schulman, John, et al. *"Proximal Policy Optimization Algorithms."* arXiv preprint arXiv:1707.06347 (2017).

## Acknowledgments

Developed as part of the Computational Models of Motion course at ETH Zurich. Robot model and environment scaffolding by Fatemeh Zargarbashi.

## License

This project is for educational and research purposes.
