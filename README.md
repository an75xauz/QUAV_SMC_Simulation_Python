# Quadrotor UAV Control Simulation

A comprehensive quadrotor UAV simulation and control system featuring Sliding Mode Control (SMC) for robust flight performance with real-time visualization.

## Overview

This project implements a complete simulation environment for a quadrotor UAV (Unmanned Aerial Vehicle), including rigid body dynamics, 3D visualization, and a robust Sliding Mode Controller for position and attitude control. The system allows users to simulate quadrotor flight trajectories, analyze control performance, and visualize results in real-time.

## System Architecture

The system is designed with a modular architecture consisting of four main components:

1. **Plant Model** (`plant.py`): Implements the dynamic model of the quadrotor UAV, including rigid body dynamics and aerodynamic effects
2. **Controller** (`controller.py`): Implements the Sliding Mode Control algorithm for robust position and attitude control
3. **Visualization** (`sim.py`): Handles simulation visualization, including real-time animation and static result plotting
4. **Main Program** (`main.py`): Serves as the entry point, handling parameter configuration and initializing the simulation environment

## Installation

### Prerequisites

- Python 3.8 or higher
- NumPy
- SciPy
- Matplotlib

### Setup

```bash
# Clone the repository
git clone https://github.com/an75xauz/QUAV_SMC_Simulation_Python.git
cd quadrotor-smc-simulation

# Install required packages
pip install numpy scipy matplotlib
```

## Usage

Run the simulation with default parameters:

```bash
python main.py
```

Specify custom simulation parameters:

```bash
python main.py --initial 0 0 1 --target 3 2 4 --time 15 --dt 0.02 --initial_attitude 0.1 0.1 0
```

Generate static plots instead of animation:

```bash
python main.py --plot
```

## Parameter Adjustment

### Command Line Parameters

The following parameters can be adjusted via command line arguments in `main.py`:

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `--initial` | Initial position [x y z] in meters | [0, 0, 0] |
| `--target` | Target position [x y z] in meters | [1, 1, 2] |
| `--time` | Simulation duration in seconds | 10.0 |
| `--dt` | Simulation time step in seconds | 0.05 |
| `--initial_attitude` | Initial attitude [roll pitch yaw] in radians | [0, 0.1, 0.1] |
| `--plot` | Generate static plots only (no animation) | False |

### Controller Parameters

Controller parameters can be modified in `controller.py` within the `QuadrotorSMCController` class:

```python
# Height control parameters
self.lambda_alt = 2.8    # Altitude sliding surface slope
self.eta_alt = 20.0      # Altitude control gain

# Attitude control parameters
self.lambda_att = 30.0   # Attitude sliding surface slope
self.eta_att = 9.0       # Attitude control gain

# Position control parameters
self.lambda_pos = 0.5    # Position sliding surface slope
self.eta_pos = 0.5       # Position control gain

# Smoothing factors
self.k_smooth = 0.5      # Smoothing factor for tanh function
self.k_smooth_pos = 0.5  # Position control smoothing factor

# Angle limitation (rad)
self.max_angle = 30 * np.pi/180  # 30 degrees in radians
```

### Plant Parameters

Physical parameters of the quadrotor can be adjusted in `plant.py` within the `QuadrotorPlant` class:

```python
# Physical parameters
self.m = 1.0         # Mass (kg)
self.g = 9.8         # Gravitational acceleration (m/s²)
self.l = 0.2         # Arm length (m)

# Moments of inertia (kg·m²)
self.Ix = 0.0211     # Around x-axis
self.Iy = 0.0219     # Around y-axis
self.Iz = 0.0366     # Around z-axis

# Aerodynamic coefficients
self.kf = 2e-6       # Propeller lift coefficient
self.km = 2.1e-7     # Propeller torque coefficient
self.kd = 0.13       # Linear drag coefficient
self.kd_ang = 0.15   # Angular drag coefficient
```

### Visualization Parameters

Visualization settings can be modified in `sim.py` within the `QuadrotorSimulator` class:

```python
# To modify the figure size:
self.fig = plt.figure(figsize=(13, 8))

# To adjust line styles and colors:
# Trajectory path
self.trajectory_plot = self.ax.plot(
    [], [], [], '--', color='orange', lw=1, alpha=0.8
)[0]

# To change the state text annotation position:
self.ax_attitude_text_annotation = self.ax_attitude.text(
    0.4, 0.85, '',
    ha='center', va='center',
    fontsize=12, color='black',
    transform=self.fig.transFigure
)
```

## Visualization Features

The simulation provides two visualization modes:

1. **Real-time Animation**: Shows the quadrotor movement, trajectory, and control states in real-time
2. **Static Plots**: Generates comprehensive plots of trajectory, position, attitude, and control inputs

The 3D visualization represents the quadrotor as a central body with four arms extending to motors, with the trajectory shown as an orange dashed line.
