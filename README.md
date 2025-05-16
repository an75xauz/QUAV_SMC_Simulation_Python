# Quadrotor UAV Reinforcement Learning Control Project

This project implements a simulation environment for a Quadrotor UAV and uses the TD3 (Twin Delayed Deep Deterministic Policy Gradient) reinforcement learning algorithm to optimize parameters of a Sliding Mode Controller (SMC). Through reinforcement learning, the system can automatically find the optimal control parameters, enabling the quadrotor to reach target positions stably and efficiently.

## Prerequisites

To run this project, your system should meet the following requirements:

- Python 3.6 or higher
- CUDA support (optional but recommended for accelerated training)

## Installation Environment

1. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

The project consists of the following main components:

- `simulation/`: Physical model and simulation environment for the quadrotor
  - `plant.py`: Quadrotor physical model
  - `controller.py`: Sliding Mode Controller (SMC)
  - `sim.py`: Simulation and visualization tools

- `rl/`: Reinforcement learning modules
  - `agent.py`: TD3 algorithm implementation
  - `env_UAV.py`: Reinforcement learning environment
  - `config.py`: Training parameter configuration
  - `register_env.py`: Environment registration utility

- `utils/`: Utility functions
  - `plot_utils.py`: Plotting tools
  - `log_utils.py`: Logging utilities

- Main execution files:
  - `train.py`: Train the reinforcement learning model
  - `test_model_17dim.py` & `test_model_16dim.py`: Test trained models
  - `main.py`: Run simulation with standard controller

## Usage Instructions

### 1. Training the Model

Use the following command to start training the reinforcement learning model:

```bash
python train.py --initial 0 0 0 --target 1 1 2 --episodes 5000 --save_dir checkpoints
```

Parameter descriptions:
- `--initial`: Initial position [x y z]
- `--target`: Target position [x y z]
- `--episodes`: Number of training episodes
- `--save_dir`: Directory to save models and logs

During training, models will be saved to the specified directory periodically, and training curve plots will be generated.

### 2. Testing the Model

Use the following command to test the trained model:

```bash
python test_model_17dim.py --initial 0 0 0 --target 1 1 2 --model_dir model_17dim --actor_file best_actor.pth
```

Parameter descriptions:
- `--initial`: Initial position [x y z]
- `--target`: Target position [x y z]
- `--time`: Simulation duration (seconds)
- `--dt`: Time step (seconds)
- `--model_dir`: Model directory
- `--actor_file`: Actor model filename
- `--plot`: If added, only generates static plots instead of showing animation

### 3. Running Standard SMC Controller Simulation

Use the following command to run a simulation with the standard controller:

```bash
python main.py --initial 0 0 0 --target 1 1 2 --time 10 --dt 0.05
```

Parameter descriptions are the same as for testing the model.

## Training Parameters Explanation

You can adjust the following training parameters in `rl/config.py`:

- `MAX_EPISODES`: Maximum number of training episodes
- `MAX_STEPS`: Maximum steps per episode
- `BUFFER_SIZE`: Experience replay buffer size
- `BATCH_SIZE`: Batch size
- `GAMMA`: Discount factor
- `TAU`: Target network soft update rate
- `POLICY_NOISE`: Policy noise magnitude
- `NOISE_CLIP`: Noise clipping range
- `POLICY_DELAY`: Policy delay update steps
- `LR`: Learning rate

## Model Architecture

- Actor network: Hidden layers of size 256-256, using ReLU activation functions and LayerNorm normalization
- Critic network: Dual Q-network structure, each Q-network having hidden layers of size 256-256

## Results Visualization

When testing the model, the system generates an animation or static plots including:

1. 3D trajectory of the quadrotor
2. Attitude angles (roll, pitch, yaw) over time
3. Position (x, y, z) over time
4. Control torques and total thrust over time
