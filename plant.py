import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class QuadrotorPlant:
    def __init__(self):
        self.m = 1 #  (kg)
        self.g = 9.8  # (m/s^2)
        self.l = 0.2  # (m)
        
        # (kg*m^2)
        self.Ix = 0.0211
        self.Iy = 0.0219
        self.Iz = 0.0366
        self.J = np.diag([self.Ix,self.Iy,self.Iz])

        # !!!! F_total ≥ m·g
        self.kf = 2e-6  # lift coefficient
        self.km = 2.1e-7  # torque coefficient

        self.kd = 0.13  # 線性阻力係數 drag coefficient
        self.kd_ang = 0.15  # angular drag coefficient

        self.max_rpm = 12000 
        self.min_rpm = 0  
        
        # state: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]

        # inertial frame
        # x, y, z
        # vx, vy, vz
        # phi, theta, psi: (roll, pitch, yaw) 

        # body frame
        # p, q, r: angular velocity
        self.state = np.zeros(12)
        
        # control input: [f1, f2, f3, f4]
        self.control_input = np.zeros(4)
        
    def reset(self, initial_state=None):
        if initial_state is None:
            self.state = np.zeros(12)
        else:
            self.state = initial_state
        return self.state
        
    def set_control_input(self, control_input):
        self.control_input = np.clip(control_input, 0, self.max_rpm)
        
    def dynamics(self, t, state, control_input):
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state
        tau_x, tau_y, tau_z, Fz = control_input
        
        foce_matrix = np.array([
                            [1,-1,-1,-1],
                            [1,1,-1,1],
                            [1,1,1,-1],
                            [1,-1,1,1]
        ])
        # print(foce_matrix)
        f1 ,f2 ,f3 ,f4= 0.25*np.dot(foce_matrix,np.array([Fz, tau_x/self.l, tau_y/self.l, tau_z*self.kf/self.km]))
        F_total = f1+f2+f3+f4
        # print(f'f1 = {f1},f2 = {f2},f3 = {f3},f4 = {f4}')
        # Torque
        # tau_phi = self.l * (f1 - f2 - f3 + f4) # Roll Torque from force differences
        # tau_theta = self.l * (f1 + f2 - f3 - f4) 
        # tau_psi = self.km / self.kf * (f1 - f2 + f3 - f4)
        # attitude cos matrix
        # Body frame to Inertial frame
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_psi = np.cos(psi)
        s_psi = np.sin(psi)
        
        # inertial frame
        ax = (s_theta * F_total) / self.m - self.kd * vx
        ay = (s_phi * c_theta * F_total) / self.m - self.kd * vy
        az = (c_phi * c_theta * F_total) / self.m - self.g - self.kd * vz
        
        # 歐拉角速度到機體角速度的轉換矩陣 transition matrix for Euler_a to Body_a
        phi_dot = p + s_phi * s_theta / c_theta * q + c_phi * s_theta / c_theta * r
        theta_dot = c_phi * q - s_phi * r
        psi_dot = s_phi / c_theta * q + c_phi / c_theta * r
        
        # acceleration (Body frame)
        p_dot = (tau_x + (self.Iy - self.Iz) * q * r) / self.Ix - self.kd_ang * p
        q_dot = (tau_y + (self.Iz - self.Ix) * p * r) / self.Iy - self.kd_ang * q
        r_dot = (tau_z + (self.Ix - self.Iy) * p * q) / self.Iz - self.kd_ang * r
        
        # return state
        state_dot = np.array([vx, vy, vz, ax, ay, az, phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot])
        return state_dot
        
    def step(self, dt, control_input=None):
        if control_input is not None:
            self.set_control_input(control_input)
            
        # to compute the next state by solved ODE
        sol = solve_ivp(
            fun=lambda t, y: self.dynamics(t, y, self.control_input),
            t_span=[0, dt],
            y0=self.state,
            method='RK45',
            t_eval=[dt]
        )
        
        self.state = sol.y[:, -1]
        return self.state
        
    def get_state(self):
        return self.state.copy()
        
    def get_position(self):
        return self.state[:3].copy()
        
    def get_velocity(self):
        return self.state[3:6].copy()
        
    def get_attitude(self):
        return self.state[6:9].copy()
        
    def get_angular_velocity(self):
        return self.state[9:12].copy()

    def get_motor_positions(self):
        x, y, z = self.state[:3]
        phi, theta, psi = self.state[6:9]
        
        # rotate matrix
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_psi = np.cos(psi)
        s_psi = np.sin(psi)
        
        # body frame to inertial frame 
        R = np.array([
            [c_theta*c_psi, s_phi*s_theta*c_psi - c_phi*s_psi, c_phi*s_theta*c_psi + s_phi*s_psi],
            [c_theta*s_psi, s_phi*s_theta*s_psi + c_phi*c_psi, c_phi*s_theta*s_psi - s_phi*c_psi],
            [-s_theta, s_phi*c_theta, c_phi*c_theta]
        ])
        
        # motor position in body frame
        motor_pos_body = np.array([
            [self.l, 0, 0],
            [0, self.l, 0],
            [-self.l, 0, 0],
            [0, -self.l, 0]
        ])
        
        # convert to inertial frame
        motor_pos_world = np.array([x, y, z]) + np.dot(motor_pos_body, R.T)
        # print(f'world : {motor_pos_world}')
        return motor_pos_world