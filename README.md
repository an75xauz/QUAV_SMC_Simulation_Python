# Quadrotor Sliding Mode Control Simulation

This repository contains a Python implementation of a quadrotor drone simulation using Sliding Mode Control (SMC). The simulation allows for testing and visualization of a quadrotor's position and attitude control in a 3D environment.

## Overview

The project is structured into three main components:

1. **Plant Model (`plant.py`)**: Implements the dynamic model of a quadrotor, including its physical properties and equations of motion.
2. **Controller (`controller.py`)**: Implements a Sliding Mode Controller for robust position and attitude control.
3. **Simulator (`sim.py`)**: Provides simulation execution, visualization, and data collection functionalities.

## Features

- Full 6-DOF (degrees of freedom) quadrotor dynamics simulation
- Sliding Mode Control implementation for robust control against uncertainties
- 3D visualization of quadrotor movement and trajectory
- Real-time plotting of position, attitude, and control inputs
- Customizable simulation parameters via command-line arguments

## Prerequisites

- Python 3.6+
- NumPy
- SciPy
- Matplotlib

## Installation

Clone the repository and install the required dependencies:

```bash
pip install numpy scipy matplotlib
```

## Usage

Run the simulation using the `sim.py` script:

```bash
python sim.py
```

### Command-line Arguments

The simulation accepts several command-line arguments to customize the run:

- `--initial x y z`: Set the initial position of the quadrotor (default: `[0, 0, 0]`)
- `--target x y z`: Set the target position (default: `[1, 1, 2]`)
- `--time t`: Set the simulation duration in seconds (default: `10.0`)
- `--dt step`: Set the simulation time step in seconds (default: `0.05`)
- `--plot`: Show only the final results without animation
- `--initial_attitude roll pitch yaw`: Set the initial attitude in radians (default: `[0, 0.1, 0.1]`)

Example:

```bash
python sim.py --initial 0 0 0 --target 2 2 3 --time 15 --initial_attitude 0 0 0 --plot
```

## Simulation Details

### Quadrotor Plant Model

The quadrotor model includes:
- Physical parameters (mass, moment of inertia, arm length)
- Dynamic equations for translational and rotational motion
- Aerodynamic drag coefficients
- State representation with 12 variables:
  - Position (x, y, z)
  - Velocity (vx, vy, vz)
  - Attitude (roll/phi, pitch/theta, yaw/psi)
  - Angular velocity (p, q, r)

### Sliding Mode Controller

The controller implements:
- Separate sliding surfaces for altitude, position, and attitude control
- Adaptive control parameters for different control aspects
- Decoupled control design with cascaded structure
- Robust against model uncertainties and disturbances
- Control input vector: [τx, τy, τz, Ftotal] (torques and total thrust)

### Visualization

The simulator provides:
- 3D visualization of the quadrotor and its trajectory
- Time series plots of position, attitude, and control inputs
- Animated visualization mode or static result plots

## Understanding the Control Structure

The control architecture follows a cascaded approach:
1. **Outer Loop**: Position control generates desired attitude references
2. **Inner Loop**: Attitude control tracks these references using angular rate feedback
3. **Sliding Mode**: Each controller utilizes sliding surfaces with equivalent control and switching terms

## Parameter Tuning

Key controller parameters that can be adjusted in `controller.py`:
- `lambda_alt`, `eta_alt`: Altitude control slope and gain
- `lambda_att`, `eta_att`: Attitude control slope and gain
- `lambda_pos`, `eta_pos`: Position control slope and gain
- `k_smooth`, `k_smooth_pos`: Smoothing factors for control signal
- `max_angle`: Safety limit for maximum attitude angles

## License

[Include your license information here]

## Acknowledgments

[Include any acknowledgments or references here]