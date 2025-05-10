"""
Sliding Mode Controller (SMC) for Quadrotor UAV.

This module implements a Sliding Mode Controller for quadrotor position and attitude control.
The controller uses sliding surfaces and control laws to ensure robust tracking performance.
"""

import numpy as np
from typing import List, Any


class QuadrotorSMCController:
    """Sliding Mode Controller for quadrotor UAV position and attitude control.
    
    This controller implements sliding mode control techniques to achieve robust
    position and attitude tracking for a quadrotor UAV in the presence of 
    uncertainties and disturbances.
    
    Attributes:
        plant: The quadrotor plant model
        m: Mass of the quadrotor (kg)
        g: Gravitational acceleration (m/s²)
        l: Arm length of the quadrotor (m)
        Ixx: Moment of inertia around x-axis (kg·m²)
        Iyy: Moment of inertia around y-axis (kg·m²)
        Izz: Moment of inertia around z-axis (kg·m²)
        lambda_alt: Altitude control slope parameter
        eta_alt: Altitude control gain parameter
        lambda_att: Attitude control slope parameter
        eta_att: Attitude control gain parameter
        lambda_pos: Position control slope parameter
        eta_pos: Position control gain parameter
        k_smooth: Smoothing factor for control inputs
        k_smooth_pos: Position control smoothing factor
        max_angle: Maximum allowed angle for roll and pitch (rad)
        target_position: Target position [x, y, z] (m)
        target_attitude: Target attitude [roll, pitch, yaw] (rad)
        prev_error_pos: Previous position error
        prev_error_att: Previous attitude error
    """
    
    def __init__(self, plant: Any) -> None:
        """Initialize the SMC controller with plant parameters.
        
        Args:
            plant: The quadrotor plant model containing mass, inertia, etc.
        """
        # Plant parameters
        self.plant = plant
        self.m = plant.m
        self.g = plant.g
        self.l = plant.l
        # Use consistent naming with plant.py
        self.Ix = plant.Ix
        self.Iy = plant.Iy
        self.Iz = plant.Iz
        
        # Control parameters
        # Altitude control parameters
        self.lambda_alt = 2.3    # Altitude sliding surface slope
        self.eta_alt = 25.0      # Altitude control gain
        
        # Attitude control parameters
        self.lambda_att = 30  # roll pitch sliding surface slope
        self.eta_att = 25     # roll pitch control gain
        # yall control parameters
        self.lambda_att_yall = 30   # yall sliding surface slope
        self.eta_att_yall = 9.0       # yall control gain
        
        # Position control parameters
        self.lambda_pos = 0.5    # Position sliding surface slope
        self.eta_pos = 0.5       # Position control gain
        
        # Smoothing factors
        self.k_smooth = 0.5      # Smoothing factor for tanh function
        self.k_smooth_pos = 0.5  # Position control smoothing factor
        
        # Angle limitation (rad)
        self.max_angle = 30 * np.pi/180  # 30 degrees in radians

        # Maximum torque
        self.max_roll_pitch_torque = 0.65  # Roll/Pitch  (Nm)
        self.max_yaw_torque = 0.6         # Yaw  (Nm)
        
        # Target states
        self.target_position = np.zeros(3)
        self.target_attitude = np.zeros(3)
        
        # Previous errors for monitoring
        self.prev_error_pos = np.zeros(3)
        self.prev_error_att = np.zeros(3)
        
    def reset(self) -> None:
        """Reset controller internal states."""
        self.prev_error_pos = np.zeros(3)
        self.prev_error_att = np.zeros(3)
        
    def set_target_position(self, position: List[float]) -> None:
        """Set the target position for the controller.
        
        Args:
            position: Target position [x, y, z] in meters
        """
        self.target_position = np.array(position)
        
    def set_target_attitude(self, attitude: List[float]) -> None:
        """Set the target attitude for the controller.
        
        Args:
            attitude: Target attitude [roll, pitch, yaw] in radians
        """
        self.target_attitude = np.array(attitude)
        
    def update(self, dt: float) -> np.ndarray:
        """Update controller and compute control inputs.
        
        This method implements the sliding mode control algorithm to compute
        the control inputs based on the current state of the quadrotor.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            np.ndarray: Control inputs [U2, U3, U4, U1] where:
                U1: Total thrust
                U2: Roll torque
                U3: Pitch torque
                U4: Yaw torque
        """
        # Get current state from plant
        state = self.plant.get_state()
        position = state[:3]               # x, y, z
        velocity = state[3:6]              # vx, vy, vz
        attitude = state[6:9]              # φ, θ, ψ (roll, pitch, yaw)
        angular_velocity = state[9:12]     # p, q, r
        
        # Extract individual state components
        x, y, z = position
        vx, vy, vz = velocity
        phi, theta, psi = attitude  # roll, pitch, yaw
        p, q, r = angular_velocity
        
        # Compute trigonometric values for coordinate transformations
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Prevent division by zero
        denom = cos_theta * cos_phi
        denom = np.sign(denom) * max(abs(denom), 1e-6)
            
        # -------------------- Altitude Control (Z-axis) --------------------
        z_d = self.target_position[2]
        e_z = z_d - z
        e_z_dot = 0 - vz  # Assuming target velocity is zero
        
        # Altitude sliding surface
        sliding_surface_alt = e_z_dot + self.lambda_alt * e_z
        
        # Equivalent control term for altitude
        thrust_eq = self.g + self.lambda_alt * vz
        
        # Switching control term for altitude with smoothing
        thrust_sw = self.eta_alt * np.tanh(self.k_smooth * sliding_surface_alt)
        
        # Total thrust force
        total_thrust = self.m * (thrust_eq + thrust_sw) / denom
        total_thrust = max(total_thrust, 0.1)  # Ensure positive non-zero thrust
        
        # Safety factor to prevent division by small values
        safe_thrust = max(total_thrust, 0.1)
        
        # -------------------- Position Control (X-axis) --------------------
        # Maps to pitch angle (theta)
        x_d = self.target_position[0]
        e_x = x_d - x
        e_x_dot = 0 - vx  # Assuming target velocity is zero
        
        # X-direction sliding surface
        sliding_surface_x = e_x_dot + self.lambda_pos * e_x
        
        # Equivalent control and switching control terms
        # In plant dynamics: ax = -F_total * sin(theta) / m
        # Negative x_virtual_control creates positive theta, which creates positive ax
        x_control_eq = -(self.m/safe_thrust) * self.lambda_pos * (-vx)
        x_control_sw = self.eta_pos * np.tanh(self.k_smooth_pos * sliding_surface_x)
        
        # Virtual control input for X direction
        x_virtual_control = x_control_eq + x_control_sw
        
        # Prevent arcsin domain error
        x_virtual_control = np.clip(x_virtual_control, -0.99, 0.99)
        theta_d = -np.arcsin(x_virtual_control)
        
        # -------------------- Position Control (Y-axis) --------------------
        # Maps to roll angle (phi)
        y_d = self.target_position[1]
        e_y = y_d - y
        e_y_dot = 0 - vy  # Assuming target velocity is zero
        
        # Y-direction sliding surface
        sliding_surface_y = e_y_dot + self.lambda_pos * e_y
        
        # Safety factor for cos(theta)
        safe_cos_theta = np.sign(cos_theta) * max(abs(cos_theta), 1e-3)
        
        # Equivalent control and switching control terms
        # In plant dynamics: ay = F_total * cos(theta) * sin(phi) / m
        # Positive y_virtual_control creates positive phi, which creates positive ay
        y_control_eq = (self.m/safe_thrust) * self.lambda_pos * (-vy) / safe_cos_theta
        y_control_sw = self.eta_pos * np.tanh(self.k_smooth_pos * sliding_surface_y)
        
        # Virtual control input for Y direction
        y_virtual_control = y_control_eq + y_control_sw
        
        # Prevent arcsin domain error
        y_virtual_control = np.clip(y_virtual_control, -0.99, 0.99)
        phi_d = np.arcsin(y_virtual_control)
        
        # Limit desired angles for safety
        phi_d = np.clip(phi_d, -self.max_angle, self.max_angle)
        theta_d = np.clip(theta_d, -self.max_angle, self.max_angle)
        
        # Target yaw angle
        psi_d = self.target_attitude[2]
        
        # -------------------- Attitude Control --------------------
        # Roll control
        e_phi = (phi_d - phi)
        sliding_surface_roll = p + self.lambda_att * e_phi
        
        roll_torque_eq = -((self.Iy - self.Iz) * q * r + self.Ix * self.lambda_att * p)
        roll_torque_sw = self.Ix * self.eta_att * np.tanh(self.k_smooth * sliding_surface_roll)
        roll_torque = roll_torque_eq + roll_torque_sw
        
        # Pitch control
        e_theta = theta_d - theta
        sliding_surface_pitch = q + self.lambda_att * e_theta
        
        pitch_torque_eq = -((self.Iz - self.Ix) * p * r + self.Iy * self.lambda_att * q)
        pitch_torque_sw = self.Iy * self.eta_att * np.tanh(self.k_smooth * sliding_surface_pitch)
        pitch_torque = pitch_torque_eq + pitch_torque_sw
        
        # Yaw control
        e_psi = psi_d - psi
        sliding_surface_yaw = r + self.lambda_att_yall * e_psi
        
        yaw_torque_eq = -((self.Ix - self.Iy) * p * q + self.Iz * self.lambda_att_yall * r)
        yaw_torque_sw = self.Iz * self.eta_att_yall * np.tanh(self.k_smooth * sliding_surface_yaw)
        yaw_torque = yaw_torque_eq + yaw_torque_sw
        
        # Ensure control inputs do not contain NaN or Inf values
        control_out = np.array([roll_torque, pitch_torque, yaw_torque, total_thrust])
        control_out = np.nan_to_num(control_out, nan=0.0, posinf=50.0, neginf=-50.0)
        
        # Update internal state
        self.prev_error_pos = np.array([e_x, e_y, e_z])
        self.prev_error_att = np.array([e_phi, e_theta, e_psi])
        
        # Return control outputs:
        # control_out[0]: Roll torque (around x-axis)
        # control_out[1]: Pitch torque (around y-axis)
        # control_out[2]: Yaw torque (around z-axis)
        # control_out[3]: Total thrust force
        return control_out