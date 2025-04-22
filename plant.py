"""
Quadrotor Plant Model Module.

This module implements the dynamic model of a quadrotor UAV, including
rigid body dynamics, aerodynamic effects, and coordinate transformations.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import  Optional, Any


class QuadrotorPlant:
    """Quadrotor UAV physical plant model.
    
    This class implements the physical dynamics of a quadrotor UAV, including
    rigid body dynamics, aerodynamic effects, and coordinate transformations.
    
    Attributes:
        m: Mass of the quadrotor (kg)
        g: Gravitational acceleration (m/s²)
        l: Arm length of the quadrotor (m)
        Ix: Moment of inertia around x-axis (kg·m²)
        Iy: Moment of inertia around y-axis (kg·m²)
        Iz: Moment of inertia around z-axis (kg·m²)
        kf: Propeller lift coefficient
        km: Propeller torque coefficient
        kd: Linear drag coefficient
        kd_ang: Angular drag coefficient
        state: Current state vector [12×1]:
            [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
        control_input: Control input vector [4×1]:
            [tau_x, tau_y, tau_z, F_total]
    """
    
    def __init__(self) -> None:
        """Initialize the quadrotor plant with default parameters."""
        # Physical parameters
        self.m = 1.0         # Mass (kg)
        self.g = 9.8         # Gravitational acceleration (m/s²)
        self.l = 0.2         # Arm length (m)
        
        # Moments of inertia (kg·m²)
        self.Ix = 0.0211     # Around x-axis
        self.Iy = 0.0219     # Around y-axis
        self.Iz = 0.0366     # Around z-axis

        # Aerodynamic coefficients
        # Note: F_total must be ≥ m·g for hovering
        self.kf = 2e-6       # Propeller lift coefficient
        self.km = 2.1e-7     # Propeller torque coefficient
        self.kd = 0.13       # Linear drag coefficient
        self.kd_ang = 0.15   # Angular drag coefficient

        # State vector initialization
        # state = [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
        # Where:
        #   - x, y, z: Position in inertial frame (m)
        #   - vx, vy, vz: Velocity in inertial frame (m/s)
        #   - phi, theta, psi: Euler angles (roll, pitch, yaw) in radians
        #   - p, q, r: Angular velocities in body frame (rad/s)
        self.state = np.zeros(12)
        
        # Control input initialization
        # control_input = [tau_x, tau_y, tau_z, F_total]
        # Where:
        #   - tau_x, tau_y, tau_z: Torques around body axes (N·m)
        #   - F_total: Total thrust force (N)
        self.control_input = np.zeros(4)
        
    def reset(self, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """Reset the quadrotor state.
        
        Args:
            initial_state: Initial state vector [12×1]. If None, zeros are used.
            
        Returns:
            The new state after reset
        """
        if initial_state is None:
            self.state = np.zeros(12)
        else:
            self.state = initial_state
        return self.state
        
    def set_control_input(self, control_input: np.ndarray) -> None:
        """Set the control input for the quadrotor.
        
        Args:
            control_input: Control input vector [4×1]:
                [tau_x, tau_y, tau_z, F_total]
        """
        self.control_input = control_input
        
    def dynamics(self, t: float, state: np.ndarray, control_input: np.ndarray) -> np.ndarray:
        """Calculate the state derivatives based on current state and control inputs.
        
        This function implements the nonlinear dynamics of the quadrotor.
        
        Args:
            t: Current time (s)
            state: Current state vector [12×1]
            control_input: Control input vector [4×1]
            
        Returns:
            State derivative vector [12×1]
        """
        # Unpack state variables
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state
        
        # Unpack control inputs
        # Control input conventions:
        # tau_x: Positive torque causes positive roll rate (clockwise around x-axis)
        # tau_y: Positive torque causes positive pitch rate (clockwise around y-axis)
        # tau_z: Positive torque causes positive yaw rate (clockwise around z-axis)
        # F_total: Total thrust force along body z-axis (upward)
        tau_x, tau_y, tau_z, F_total = control_input
        
        # Calculate trigonometric functions for coordinate transformations
        c_phi = np.cos(phi)      # cos(roll)
        s_phi = np.sin(phi)      # sin(roll)
        c_theta = np.cos(theta)  # cos(pitch)
        s_theta = np.sin(theta)  # sin(pitch)
        
        # Calculate accelerations in inertial frame
        # Forces include thrust, gravity, and drag
        ax = -F_total * s_theta / self.m - self.kd * vx
        ay = F_total * c_theta * s_phi / self.m - self.kd * vy
        az = F_total * c_theta * c_phi / self.m - self.g - self.kd * vz
        
        # Prevent division by zero in angle rate calculations
        # Add small epsilon to cos(theta) to avoid singularity
        epsilon = 1e-6
        tan_theta = s_theta / (c_theta + epsilon)
        sec_theta = 1 / (c_theta + epsilon)
        
        # Calculate angular rates (Euler rates from body rates)
        phi_dot = p + s_phi * tan_theta * q + c_phi * tan_theta * r
        theta_dot = c_phi * q - s_phi * r
        psi_dot = s_phi * sec_theta * q + c_phi * sec_theta * r
        
        # Calculate angular accelerations in body frame
        # Includes gyroscopic effects, control torques, and drag
        p_dot = ((self.Iy - self.Iz) * q * r + tau_x) / self.Ix - self.kd_ang * p
        q_dot = ((self.Iz - self.Ix) * p * r + tau_y) / self.Iy - self.kd_ang * q
        r_dot = ((self.Ix - self.Iy) * p * q + tau_z) / self.Iz - self.kd_ang * r
        
        # Compile the state derivative vector
        state_dot = np.array([
            vx, vy, vz,                    # Position derivatives
            ax, ay, az,                    # Velocity derivatives
            phi_dot, theta_dot, psi_dot,   # Attitude derivatives
            p_dot, q_dot, r_dot            # Angular velocity derivatives
        ])
        
        return state_dot
        
    def step(self, dt: float, control_input: Optional[np.ndarray] = None) -> np.ndarray:
        """Advance the quadrotor state by one time step.
        
        Uses numerical integration (RK45) to update the state based on dynamics.
        
        Args:
            dt: Time step (s)
            control_input: Optional new control input. If None, uses current input.
            
        Returns:
            New state after the time step
        """
        if control_input is not None:
            self.set_control_input(control_input)
            
        # Use RK45 numerical integration to solve the ODEs
        sol = solve_ivp(
            fun=lambda t, y: self.dynamics(t, y, self.control_input),
            t_span=[0, dt],
            y0=self.state,
            method='RK45',
            t_eval=[dt]
        )
        
        # Update the state with the integration result
        self.state = sol.y[:, -1]
        return self.state
        
    def get_state(self) -> np.ndarray:
        """Get the current state vector.
        
        Returns:
            Current state vector [12×1]
        """
        return self.state.copy()
        
    def get_position(self) -> np.ndarray:
        """Get the current position in inertial frame.
        
        Returns:
            Position vector [3×1] in meters
        """
        return self.state[:3].copy()
        
    def get_velocity(self) -> np.ndarray:
        """Get the current velocity in inertial frame.
        
        Returns:
            Velocity vector [3×1] in m/s
        """
        return self.state[3:6].copy()
        
    def get_attitude(self) -> np.ndarray:
        """Get the current attitude (Euler angles).
        
        Returns:
            Attitude vector [3×1] (roll, pitch, yaw) in radians
        """
        return self.state[6:9].copy()
        
    def get_angular_velocity(self) -> np.ndarray:
        """Get the current angular velocity in body frame.
        
        Returns:
            Angular velocity vector [3×1] in rad/s
        """
        return self.state[9:12].copy()

    def get_motor_positions(self) -> np.ndarray:
        """Calculate the positions of the four motors in inertial frame.
        
        Returns:
            Array of motor positions [4×3] in meters
        """
        x, y, z = self.state[:3]
        phi, theta, psi = self.state[6:9]
        
        # Calculate rotation matrix (body to inertial frame)
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_psi = np.cos(psi)
        s_psi = np.sin(psi)
        
        # Rotation matrix (DCM) from body to inertial frame
        R = np.array([
            [c_theta*c_psi, s_phi*s_theta*c_psi - c_phi*s_psi, c_phi*s_theta*c_psi + s_phi*s_psi],
            [c_theta*s_psi, s_phi*s_theta*s_psi + c_phi*c_psi, c_phi*s_theta*s_psi - s_phi*c_psi],
            [-s_theta, s_phi*c_theta, c_phi*c_theta]
        ])
        
        # Motor positions in body frame
        # Arranged in clockwise order: front, right, back, left
        motor_pos_body = np.array([
            [self.l, 0, 0],   # Front motor
            [0, self.l, 0],   # Right motor
            [-self.l, 0, 0],  # Back motor
            [0, -self.l, 0]   # Left motor
        ])
        
        # Transform motor positions from body frame to inertial frame
        motor_pos_world = np.array([x, y, z]) + np.dot(motor_pos_body, R.T)
        return motor_pos_world