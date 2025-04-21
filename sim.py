import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import argparse
import time

from plant import QuadrotorPlant
from controller import QuadrotorSMCController

class QuadrotorSimulator:
    def __init__(self, initial_position=[0, 0, 0], target_position=[1, 1, 1],initial_attitude=[0, 0, 0]):
    
        self.plant = QuadrotorPlant()
        self.controller = QuadrotorSMCController(self.plant)
        
        # setup initial state
        initial_state = np.zeros(12)
        initial_state[:3] = initial_position
        initial_state[6:9] = initial_attitude
        self.plant.reset(initial_state)
        
        # set taget
        self.controller.set_target_position(target_position)
        self.controller.set_target_attitude([0, 0, 0]) 
        
        self.dt = 0.01  
        self.t = 0.0  
        self.max_time = 10.0  
        
        # save data 
        self.trajectory = [self.plant.get_position()]
        self.time_history = [self.t]
        self.attitude_history = [self.plant.get_attitude()]
        self.control_inputs_history = []
        
        # Figure used
        self.fig = None
        self.ax = None
        self.quadrotor_plot = None
        self.trajectory_plot = None
        self.target_plot = None
        self.error_history = []
        
    def step(self):

        # Update control input and read control input
        control_input = self.controller.update(self.dt)
        self.control_inputs_history.append(control_input)

        # Update QUAV state
        self.plant.step(self.dt, control_input)
        
 
        self.t += self.dt
        
        # record trajectory
        self.trajectory.append(self.plant.get_position())
        self.time_history.append(self.t)
        self.attitude_history.append(self.plant.get_attitude())
        # record error
        current_position = self.plant.get_position()
        error = np.linalg.norm(current_position - self.controller.target_position)
        self.error_history.append(error)

        return self.t >= self.max_time
        
    def run(self):
        done = False
        while not done:
            done = self.step()
            
        #Convert to numpy data type
        self.trajectory = np.array(self.trajectory)
        self.time_history = np.array(self.time_history)
        self.attitude_history = np.array(self.attitude_history)
        self.control_inputs_history = np.array(self.control_inputs_history)
        
    def setup_animation(self):
        self.fig = plt.figure(figsize=(13, 6))
        gs = gridspec.GridSpec(3, 2)
        
        # 3D Vision figure
        self.ax = self.fig.add_subplot(gs[:,0], projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f'Simulation')
        
        # Attitude figure
        self.ax_attitude = self.fig.add_subplot(gs[0,1])
        self.ax_attitude.set_xlabel('Time (s)')
        self.ax_attitude.set_ylabel('Degree (rad)')
        self.ax_attitude.set_title('Attitude')
        # text
        self.ax_attitude_text_annotation = self.ax_attitude.text(0.1, 0.05 , '', # 初始文字為空
                          ha='center', va='center', # 置中對齊
                          fontsize=10, color='black',
                          transform=self.fig.transFigure) # 使用座標軸相對座標

        # Position
        self.ax_position = self.fig.add_subplot(gs[1,1])
        self.ax_position.set_xlabel('Time (s)')
        self.ax_position.set_ylabel('Position (m)')
        self.ax_position.set_title('Position')
        # Error
        self.ax_control = self.fig.add_subplot(gs[2,1])
        self.ax_control.set_xlabel('Time (s)')
        self.ax_control.set_ylabel('Control Inputs')
        self.ax_control.set_title('Control Inputs')
        
        # Set up limitation of the figure
        target_pos = self.controller.target_position
        max_range = max(abs(target_pos).max() * 1.5, 2.0)
        self.ax.set_xlim([-max_range, max_range])
        self.ax.set_ylim([-max_range, max_range])
        self.ax.set_zlim([-max_range, max_range])
        
        # initialize figure
        self.quadrotor_plot = [
            self.ax.plot([], [], [], 'o-', color='blue', lw=2)[0]  # QUAV body
        ]
        self.trajectory_plot = self.ax.plot([], [], [], '--', color='green', alpha=0.5)[0]  
        self.target_plot = self.ax.plot([target_pos[0]], [target_pos[1]], [target_pos[2]], 'ro', markersize=10)[0]  # target
        
        self.roll_plot, = self.ax_attitude.plot([], [], label='Roll (ϕ)')
        self.pitch_plot, = self.ax_attitude.plot([], [], label='Pitch (θ)')
        self.yaw_plot, = self.ax_attitude.plot([], [], label='Yaw (ψ)')
        self.ax_attitude.legend()
        
        self.x_plot, = self.ax_position.plot([], [], label='X')
        self.y_plot, = self.ax_position.plot([], [], label='Y')
        self.z_plot, = self.ax_position.plot([], [], label='Z')
        self.ax_position.legend()


        self.control_tau_x_plot, = self.ax_control.plot([], [], 'b-', label='tau_x')
        self.control_tau_y_plot, = self.ax_control.plot([], [], 'y-', label='tau_y')
        self.control_tau_z_plot, = self.ax_control.plot([], [], 'm-', label='tau_z')
        self.control_fz_plot, = self.ax_control.plot([], [], 'c:',label='Fz')
        self.ax_control.legend()
        
        return self.fig
        
    def update_animation(self, frame):
        #compute the frame about current time
        t_idx = min(frame, len(self.time_history) - 1)
        
        # to update the plant state
        self.plant.reset(np.zeros(12))
        
        # state
        state = np.zeros(12)
        state[:3] = self.trajectory[t_idx]
        state[6:9] = self.attitude_history[t_idx]
        self.plant.state = state
        
        # update the body position
        motor_positions = self.plant.get_motor_positions()
        center = self.plant.get_position()
        
        # connect to four motors
        x_data = np.append(motor_positions[:, 0], [center[0], motor_positions[0, 0]])
        y_data = np.append(motor_positions[:, 1], [center[1], motor_positions[0, 1]])
        z_data = np.append(motor_positions[:, 2], [center[2], motor_positions[0, 2]])
        
        self.quadrotor_plot[0].set_data(x_data, y_data)
        self.quadrotor_plot[0].set_3d_properties(z_data)
        
        # update trajectory
        self.trajectory_plot.set_data(self.trajectory[:t_idx+1, 0], self.trajectory[:t_idx+1, 1])
        self.trajectory_plot.set_3d_properties(self.trajectory[:t_idx+1, 2])
        
        # attitude figure
        self.roll_plot.set_data(self.time_history[:t_idx+1], self.attitude_history[:t_idx+1, 0])
        self.pitch_plot.set_data(self.time_history[:t_idx+1], self.attitude_history[:t_idx+1, 1])
        self.yaw_plot.set_data(self.time_history[:t_idx+1], self.attitude_history[:t_idx+1, 2])
        # plot text
        current_roll = self.attitude_history[t_idx, 0]
        current_pitch = self.attitude_history[t_idx, 1]
        current_yaw = self.attitude_history[t_idx, 2]
        current_x = self.trajectory[t_idx, 0]
        current_y = self.trajectory[t_idx, 1]
        current_z = self.trajectory[t_idx, 2]
        self.ax_attitude_text_annotation.set_text(f'ϕ = {current_roll:.2f} rad, x = {current_x:.2f}\n'
        f'θ = {current_pitch:.2f} rad, y = {current_y:.2f}\n'
        f'ψ = {current_yaw:.2f} rad, z = {current_z:.2f}' ) 
        
        # position figure
        self.x_plot.set_data(self.time_history[:t_idx+1], self.trajectory[:t_idx+1, 0])
        self.y_plot.set_data(self.time_history[:t_idx+1], self.trajectory[:t_idx+1, 1])
        self.z_plot.set_data(self.time_history[:t_idx+1], self.trajectory[:t_idx+1, 2])

        # control input figure
        if t_idx < len(self.control_inputs_history):
            self.control_tau_x_plot.set_data(self.time_history[:t_idx+1], self.control_inputs_history[:t_idx+1, 0])
            self.control_tau_y_plot.set_data(self.time_history[:t_idx+1], self.control_inputs_history[:t_idx+1, 1])
            self.control_tau_z_plot.set_data(self.time_history[:t_idx+1], self.control_inputs_history[:t_idx+1, 2])
            self.control_fz_plot.set_data(self.time_history[:t_idx+1], self.control_inputs_history[:t_idx+1, 3])
              
        #set up the range of the figure
        if t_idx > 0:
            self.ax_attitude.relim()
            self.ax_attitude.autoscale_view()
            self.ax_position.relim()
            self.ax_position.autoscale_view()
            self.ax_control.relim()
            self.ax_control.autoscale_view()
        
        return self.quadrotor_plot + [self.trajectory_plot]
        
    def animate(self):
        self.setup_animation()
        ani = FuncAnimation(self.fig, self.update_animation, frames=len(self.time_history),
                            interval=self.dt*10, blit=False)
        plt.tight_layout()
        plt.show()
        
    def plot_results(self):

        fig = plt.figure(figsize=(16, 10))
        # 3D
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot(self.trajectory[:, 0], self.trajectory[:, 1], self.trajectory[:, 2], 'b-')
        ax1.plot([self.trajectory[0, 0]], [self.trajectory[0, 1]], [self.trajectory[0, 2]], 'go', label='Initial Point')
        ax1.plot([self.controller.target_position[0]], [self.controller.target_position[1]], 
                [self.controller.target_position[2]], 'ro', label='Target Point')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Simulation')
        ax1.legend()
        
        # position
        ax2 = fig.add_subplot(222)
        ax2.plot(self.time_history, self.trajectory[:, 0], 'r-', label='X')
        ax2.plot(self.time_history, self.trajectory[:, 1], 'g-', label='Y')
        ax2.plot(self.time_history, self.trajectory[:, 2], 'b-', label='Z')
        ax2.axhline(y=self.controller.target_position[0], color='r', linestyle='--')
        ax2.axhline(y=self.controller.target_position[1], color='g', linestyle='--')
        ax2.axhline(y=self.controller.target_position[2], color='b', linestyle='--')

        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position (m)')
        ax2.set_title('Position Change')
        ax2.legend()
        
        # attitude
        ax3 = fig.add_subplot(223)
        ax3.plot(self.time_history, self.attitude_history[:, 0], 'r-', label='Roll (ϕ)')
        ax3.plot(self.time_history, self.attitude_history[:, 1], 'g-', label='Pitch (θ)')
        ax3.plot(self.time_history, self.attitude_history[:, 2], 'b-', label='Yaw (ψ)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Degrees (rad)')
        ax3.set_title('Attitude Change')
        ax3.legend()
        # control input
        ax4 = fig.add_subplot(224)
        ax4.plot(self.time_history[:-1], self.control_inputs_history[:, 0], 'y-', label='τ_x')
        ax4.plot(self.time_history[:-1], self.control_inputs_history[:, 1], 'm-', label='τ_y')
        ax4.plot(self.time_history[:-1], self.control_inputs_history[:, 2], 'k-', label='τ_z')
        ax4.plot(self.time_history[:-1], self.control_inputs_history[:, 3], 'b-', label='Fz')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Control Inputs')
        ax4.set_title('Control Inputs')
        ax4.legend()

        plt.tight_layout()
        plt.show()



def main():
    # You can change in this function for simulation parameters
    parser = argparse.ArgumentParser(description='Control simulation')
    parser.add_argument('--initial', type=float, nargs=3, default=[0, 0, 0],
                        help='Initial Position [x y z] (Default: [0 0 0])')
    parser.add_argument('--target', type=float, nargs=3, default=[1, 5, 5],
                        help='Target Position [x y z] (Default: [1 1 1])')
    parser.add_argument('--time', type=float, default=10,
                        help='Simulation times (s) (Default: 10.0)')
    parser.add_argument('--dt', type=float, default=0.05,
                        help='Times step (s) (Default: 0.01)')
    parser.add_argument('--plot', action='store_true',
                        help='Only Result')
    parser.add_argument('--initial_attitude', type=float, nargs=3, default=[0, 0.2, 0],
                        help='Initial Attitude [roll pitch yaw] (Default: [0 0 0])')
    
    
    args = parser.parse_args()
    
    simulator = QuadrotorSimulator(args.initial, args.target, args.initial_attitude)
    simulator.dt = args.dt
    simulator.max_time = args.time
    
    print(f"Run initial{args.initial} to {args.target} \n intitail attitude {args.initial_attitude}" )
    start_time = time.time()
    simulator.run()
    end_time = time.time()

    if args.plot:
        simulator.plot_results()
    else:
        simulator.animate()

if __name__ == "__main__":
    main()