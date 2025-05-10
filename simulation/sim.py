"""
Quadrotor Simulation and Visualization Module.

This module provides visualization capabilities for a quadrotor UAV simulation,
including real-time animation and static result plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from typing import List


class QuadrotorSimulator:
    """Simulator for quadrotor UAV dynamics and control.
    
    This class implements a complete simulation environment for a quadrotor UAV,
    including time integration, data recording, and visualization.
    
    Attributes:
        plant: The quadrotor plant model
        controller: The controller for the quadrotor
        dt: Simulation time step (s)
        t: Current simulation time (s)
        max_time: Maximum simulation time (s)
        trajectory: History of quadrotor positions
        time_history: History of simulation times
        attitude_history: History of quadrotor attitudes
        control_inputs_history: History of control inputs
        error_history: History of position errors
    """
    
    def __init__(self, 
                 plant,
                 controller,
                 initial_position=[0, 0, 0],
                 target_position=[1, 1, 1],
                 initial_attitude=[0, 0, 0]) -> None:
        """Initialize the quadrotor simulator.
        
        Args:
            plant: The quadrotor plant model
            controller: The SMC controller for the quadrotor
            initial_position: Initial position [x, y, z] in meters
            target_position: Target position [x, y, z] in meters
            initial_attitude: Initial attitude [roll, pitch, yaw] in radians
        """
        # Store plant and controller references
        self.plant = plant
        self.controller = controller
        
        # Set up initial state
        initial_state = np.zeros(12)
        initial_state[:3] = initial_position
        initial_state[6:9] = initial_attitude
        self.plant.reset(initial_state)
        
        # Set target states
        self.controller.set_target_position(target_position)
        self.controller.set_target_attitude([0, 0, 0])
        
        # Simulation parameters
        self.dt = 0.01        # Time step (s)
        self.t = 0.0          # Current time (s)
        self.max_time = 10.0  # Maximum simulation time (s)
        
        # Initialize data storage
        self.trajectory = [self.plant.get_position()]
        self.time_history = [self.t]
        self.attitude_history = [self.plant.get_attitude()]
        self.control_inputs_history = []
        self.error_history = []
        
        # Visualization objects
        self.fig = None
        self.ax = None
        self.quadrotor_plot = None
        self.trajectory_plot = None
        self.target_plot = None
        
    def step(self) -> bool:
        """Advance the simulation by one time step.
        
        Returns:
            bool: True if simulation is complete, False otherwise
        """
        # Compute control inputs
        control_input = self.controller.update(self.dt)
        self.control_inputs_history.append(control_input)

        # Update quadrotor state
        self.plant.step(self.dt, control_input)
        
        # Advance simulation time
        self.t += self.dt
        
        # Record data
        self.trajectory.append(self.plant.get_position())
        self.time_history.append(self.t)
        self.attitude_history.append(self.plant.get_attitude())
        
        # Calculate and record position error
        current_position = self.plant.get_position()
        error = np.linalg.norm(current_position - self.controller.target_position)
        self.error_history.append(error)

        # Check if simulation is complete
        return self.t >= self.max_time
        
    def run(self) -> None:
        """Run the complete simulation until max_time is reached."""
        done = False
        while not done:
            done = self.step()
            
        # Convert recorded data to numpy arrays for easier analysis
        self.trajectory = np.array(self.trajectory)
        self.time_history = np.array(self.time_history)
        self.attitude_history = np.array(self.attitude_history)
        self.control_inputs_history = np.array(self.control_inputs_history)
        self.error_history = np.array(self.error_history)
        
    def setup_animation(self) -> plt.Figure:
        """Set up the visualization figure and axes.
        
        Returns:
            The configured matplotlib figure
        """
        # Create figure with grid layout - now with 4 rows for the separate thrust plot
        self.fig = plt.figure(figsize=(13, 8))
        gs = gridspec.GridSpec(4, 2)
        
        # Set up 3D visualization subplot
        self.ax = self.fig.add_subplot(gs[:, 0], projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Quadrotor Simulation')
        
        # Set up attitude subplot
        self.ax_attitude = self.fig.add_subplot(gs[0, 1])
        self.ax_attitude.set_xlabel('Time (s)')
        self.ax_attitude.set_ylabel('Angle (rad)')
        self.ax_attitude.set_title('Attitude')
        
        # Set up position subplot
        self.ax_position = self.fig.add_subplot(gs[1, 1])
        self.ax_position.set_xlabel('Time (s)')
        self.ax_position.set_ylabel('Position (m)')
        self.ax_position.set_title('Position')
        
        # Set up torque control inputs subplot
        self.ax_torques = self.fig.add_subplot(gs[2, 1])
        self.ax_torques.set_xlabel('Time (s)')
        self.ax_torques.set_ylabel('Torque (N·m)')
        self.ax_torques.set_title('Control Torques')
        
        # NEW: Set up thrust control input subplot
        self.ax_thrust = self.fig.add_subplot(gs[3, 1])
        self.ax_thrust.set_xlabel('Time (s)')
        self.ax_thrust.set_ylabel('Thrust (N)')
        self.ax_thrust.set_title('Total Thrust Force')
        
        # Text annotation for real-time state display with updated position and font size
        self.ax_attitude_text_annotation = self.ax_attitude.text(
            0.4, 0.85, '',  # Initial text is empty
            ha='center', va='center',
            fontsize=12, color='black',
            transform=self.fig.transFigure
        )
        
        # Set up axis limits based on target position
        target_pos = self.controller.target_position
        max_range = max(abs(target_pos).max() * 1.5, 2.0)
        self.ax.set_xlim([-max_range, max_range])
        self.ax.set_ylim([-max_range, max_range])
        self.ax.set_zlim([-max_range, max_range])
        
        # Initialize plots
        # Quadrotor body - using blue with thinner line
        self.quadrotor_plot = [
            self.ax.plot([], [], [], 'o-', color='blue', lw=1)[0]
        ]
        
        # Trajectory - using orange dashed line with thinner width
        self.trajectory_plot = self.ax.plot(
            [], [], [], '--', color='orange', lw=1, alpha=0.8
        )[0]
        
        # Target position - using orange (to match trajectory)
        self.target_plot = self.ax.plot(
            [target_pos[0]], [target_pos[1]], [target_pos[2]],
            'o', color='orange', markersize=10
        )[0]
        
        # Using blue, red, orange solid lines for attitude plots with thinner lines
        self.roll_plot, = self.ax_attitude.plot([], [], '-', color='blue', linewidth=1, label='Roll (ϕ)')
        self.pitch_plot, = self.ax_attitude.plot([], [], '-', color='red', linewidth=1, label='Pitch (θ)')
        self.yaw_plot, = self.ax_attitude.plot([], [], '-', color='orange', linewidth=1, label='Yaw (ψ)')
        self.ax_attitude.legend()
        
        # Position plots with blue, red, orange solid lines with thinner lines
        self.x_plot, = self.ax_position.plot([], [], '-', color='blue', linewidth=1, label='X')
        self.y_plot, = self.ax_position.plot([], [], '-', color='red', linewidth=1, label='Y')
        self.z_plot, = self.ax_position.plot([], [], '-', color='orange', linewidth=1, label='Z')
        self.ax_position.legend()

        # Torque control input plots with blue, red, orange solid lines with thinner lines
        self.roll_torque_plot, = self.ax_torques.plot([], [], '-', color='blue', linewidth=1, label='Roll Torque')
        self.pitch_torque_plot, = self.ax_torques.plot([], [], '-', color='red', linewidth=1, label='Pitch Torque')
        self.yaw_torque_plot, = self.ax_torques.plot([], [], '-', color='orange', linewidth=1, label='Yaw Torque')
        self.ax_torques.legend()
        
        # Total thrust plot - using black solid line with thinner width
        self.total_thrust_plot, = self.ax_thrust.plot([], [], '-', color='black', linewidth=1.5, label='Total Thrust')
        self.ax_thrust.legend()
        
        return self.fig
        
    def update_animation(self, frame: int) -> List:
        """Update the animation for a given frame.
        
        Args:
            frame: The current frame number
            
        Returns:
            List of artists that were updated
        """
        # Limit frame index to available data
        t_idx = min(frame, len(self.time_history) - 1)
        
        # Reset plant state for visualization
        self.plant.reset(np.zeros(12))
        
        # Update plant state with historical data
        state = np.zeros(12)
        state[:3] = self.trajectory[t_idx]
        state[6:9] = self.attitude_history[t_idx]
        self.plant.state = state
        
        # Get current motor positions for quadrotor visualization
        motor_positions = self.plant.get_motor_positions()
        center = self.plant.get_position()
        
        # Create data for quadrotor visualization with motors only connected to center
        # Instead of drawing a perimeter, we'll only show the arms connecting to the center
        
        # Create separate plots for the arms connecting motors to center
        if len(self.quadrotor_plot) == 1:
            # Initialize arm plots the first time
            self.quadrotor_plot = []
            for i in range(4):
                arm, = self.ax.plot([], [], [], 'o-', color='blue', lw=1)
                self.quadrotor_plot.append(arm)
            
        # Update each arm (connecting each motor to the center)
        for i in range(4):
            x_arm = [motor_positions[i, 0], center[0]]
            y_arm = [motor_positions[i, 1], center[1]]
            z_arm = [motor_positions[i, 2], center[2]]
            self.quadrotor_plot[i].set_data(x_arm, y_arm)
            self.quadrotor_plot[i].set_3d_properties(z_arm)
        
        # Update trajectory plot
        self.trajectory_plot.set_data(self.trajectory[:t_idx+1, 0], self.trajectory[:t_idx+1, 1])
        self.trajectory_plot.set_3d_properties(self.trajectory[:t_idx+1, 2])
        
        # Update attitude plots
        self.roll_plot.set_data(self.time_history[:t_idx+1], self.attitude_history[:t_idx+1, 0])
        self.pitch_plot.set_data(self.time_history[:t_idx+1], self.attitude_history[:t_idx+1, 1])
        self.yaw_plot.set_data(self.time_history[:t_idx+1], self.attitude_history[:t_idx+1, 2])
        
        # Update text annotation with current state values
        current_roll = self.attitude_history[t_idx, 0]
        current_pitch = self.attitude_history[t_idx, 1]
        current_yaw = self.attitude_history[t_idx, 2]
        current_x = self.trajectory[t_idx, 0]
        current_y = self.trajectory[t_idx, 1]
        current_z = self.trajectory[t_idx, 2]
        
        state_text = (
            f'ϕ = {current_roll:.2f} rad, x = {current_x:.2f}\n'
            f'θ = {current_pitch:.2f} rad, y = {current_y:.2f}\n'
            f'ψ = {current_yaw:.2f} rad, z = {current_z:.2f}'
        )
        self.ax_attitude_text_annotation.set_text(state_text)
        
        # Update position plots
        self.x_plot.set_data(self.time_history[:t_idx+1], self.trajectory[:t_idx+1, 0])
        self.y_plot.set_data(self.time_history[:t_idx+1], self.trajectory[:t_idx+1, 1])
        self.z_plot.set_data(self.time_history[:t_idx+1], self.trajectory[:t_idx+1, 2])

        # Update control input plots (if data is available)
        if t_idx < len(self.control_inputs_history):
            # Update torque plots
            # control_inputs_history[:, 0] is roll_torque (formerly U2)
            self.roll_torque_plot.set_data(
                self.time_history[:t_idx+1], 
                self.control_inputs_history[:t_idx+1, 0]
            )
            # control_inputs_history[:, 1] is pitch_torque (formerly U3)
            self.pitch_torque_plot.set_data(
                self.time_history[:t_idx+1], 
                self.control_inputs_history[:t_idx+1, 1]
            )
            # control_inputs_history[:, 2] is yaw_torque (formerly U4)
            self.yaw_torque_plot.set_data(
                self.time_history[:t_idx+1], 
                self.control_inputs_history[:t_idx+1, 2]
            )
            
            # Update total thrust plot
            # control_inputs_history[:, 3] is total_thrust (formerly U1)
            self.total_thrust_plot.set_data(
                self.time_history[:t_idx+1], 
                self.control_inputs_history[:t_idx+1, 3]
            )
              
        # Automatically adjust subplot axes limits
        if t_idx > 0:
            self.ax_attitude.relim()
            self.ax_attitude.autoscale_view()
            self.ax_position.relim()
            self.ax_position.autoscale_view()
            self.ax_torques.relim()
            self.ax_torques.autoscale_view()
            self.ax_thrust.relim()
            self.ax_thrust.autoscale_view()
        
        # Return all quadrotor plot elements plus the trajectory
        return self.quadrotor_plot + [self.trajectory_plot]
        
    def animate(self) -> None:
        """Create and display an animation of the simulation results."""
        self.setup_animation()
        ani = FuncAnimation(
            self.fig,
            self.update_animation,
            frames=len(self.time_history),
            interval=self.dt*10,  # Animation speed
            blit=False
        )
        plt.tight_layout()
        plt.show()
        
    def plot_results(self) -> None:
        """Create and display static plots of the simulation results."""
        fig = plt.figure(figsize=(16, 12))
        
        # 3D trajectory plot with orange dashed line (thinner) for path and orange for target
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot(self.trajectory[:, 0], self.trajectory[:, 1], self.trajectory[:, 2], '--', color='orange', linewidth=1)
        ax1.plot(
            [self.trajectory[0, 0]], [self.trajectory[0, 1]], [self.trajectory[0, 2]],
            'o', color='blue', markersize=8, label='Initial Point'
        )
        ax1.plot(
            [self.controller.target_position[0]], [self.controller.target_position[1]], 
            [self.controller.target_position[2]], 'o', color='orange', markersize=8, label='Target Point'
        )
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        # Simple title without color descriptions
        ax1.set_title('Quadrotor Trajectory')
        ax1.legend()
        
        # Position plot with blue, red, orange solid lines - thinner lines
        ax2 = fig.add_subplot(222)
        ax2.plot(self.time_history, self.trajectory[:, 0], '-', color='blue', linewidth=1, label='X')
        ax2.plot(self.time_history, self.trajectory[:, 1], '-', color='red', linewidth=1, label='Y')
        ax2.plot(self.time_history, self.trajectory[:, 2], '-', color='orange', linewidth=1, label='Z')
        ax2.axhline(y=self.controller.target_position[0], color='blue', linestyle='--', alpha=0.5)
        ax2.axhline(y=self.controller.target_position[1], color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=self.controller.target_position[2], color='orange', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position (m)')
        ax2.set_title('Position vs. Time')
        ax2.legend()
        
        # Attitude plot with blue, red, orange solid lines - thinner lines
        ax3 = fig.add_subplot(223)
        ax3.plot(self.time_history, self.attitude_history[:, 0], '-', color='blue', linewidth=1, label='Roll (ϕ)')
        ax3.plot(self.time_history, self.attitude_history[:, 1], '-', color='red', linewidth=1, label='Pitch (θ)')
        ax3.plot(self.time_history, self.attitude_history[:, 2], '-', color='orange', linewidth=1, label='Yaw (ψ)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Angle (rad)')
        ax3.set_title('Attitude vs. Time')
        ax3.legend()
        
        # Create 2x1 grid for control inputs
        gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=fig.add_subplot(224).get_subplotspec())
        
        # Control torques plot with blue, red, orange solid lines - thinner lines
        ax4 = fig.add_subplot(gs[0, 0])
        ax4.plot(self.time_history[:-1], self.control_inputs_history[:, 0], '-', color='blue', linewidth=1, label='Roll Torque')
        ax4.plot(self.time_history[:-1], self.control_inputs_history[:, 1], '-', color='red', linewidth=1, label='Pitch Torque')
        ax4.plot(self.time_history[:-1], self.control_inputs_history[:, 2], '-', color='orange', linewidth=1, label='Yaw Torque')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Torque (N·m)')
        ax4.set_title('Control Torques vs. Time')
        ax4.legend()
        
        # Total thrust plot with black solid line - thinner line
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.plot(self.time_history[:-1], self.control_inputs_history[:, 3], '-', color='black', linewidth=1.5, label='Total Thrust')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Thrust (N)')
        ax5.set_title('Total Thrust vs. Time')
        ax5.legend()

        plt.tight_layout()
        plt.show()