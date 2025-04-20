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
        
        # control input: [tau_x, tau_y, tau_z, F_total]
        self.control_input = np.zeros(4)
        
    def reset(self, initial_state=None):
        if initial_state is None:
            self.state = np.zeros(12)
        else:
            self.state = initial_state
        return self.state
        
    def set_control_input(self, control_input):
        self.control_input = control_input
        
    def dynamics(self, t, state, control_input):
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state
        tau_x, tau_y, tau_z, F_total = control_input
        
        # 確保力矩和推力的符號定義正確
        # tau_x: 正值使機體繞X軸正向旋轉(增加roll角phi)
        # tau_y: 正值使機體繞Y軸正向旋轉(增加pitch角theta)
        # tau_z: 正值使機體繞Z軸正向旋轉(增加yaw角psi)
        # F_total: 總推力，沿機體Z軸向上
        
        # Body frame to Inertial frame三角函數
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_psi = np.cos(psi)
        s_psi = np.sin(psi)
        
        # 加速度計算 (inertial frame)
        # 這些方程與MATLAB完全一致
        ax = -F_total * s_theta / self.m - self.kd * vx
        ay = F_total * c_theta * s_phi / self.m - self.kd * vy
        az = F_total * c_theta * c_phi / self.m - self.g - self.kd * vz
        
        # 保護計算，防止數值問題
        tan_theta = s_theta / (c_theta + 1e-6)  # 加小值防止除零
        sec_theta = 1 / (c_theta + 1e-6)  # 加小值防止除零
        
        # 歐拉角運動學方程，與MATLAB一致
        phi_dot = p + s_phi * tan_theta * q + c_phi * tan_theta * r
        theta_dot = c_phi * q - s_phi * r
        psi_dot = s_phi * sec_theta * q + c_phi * sec_theta * r
        
        # 角加速度計算 (Body frame)
        # 確保力矩定義與MATLAB一致
        p_dot = ((self.Iy - self.Iz) * q * r + tau_x) / self.Ix - self.kd_ang * p
        q_dot = ((self.Iz - self.Ix) * p * r + tau_y) / self.Iy - self.kd_ang * q
        r_dot = ((self.Ix - self.Iy) * p * q + tau_z) / self.Iz - self.kd_ang * r
        
        # 返回狀態導數
        state_dot = np.array([vx, vy, vz, ax, ay, az, phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot])
        return state_dot
        
    def step(self, dt, control_input=None):
        if control_input is not None:
            self.set_control_input(control_input)
            
        # 使用RK45解微分方程
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
        
        # 旋轉矩陣 - Body frame to Inertial frame
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_psi = np.cos(psi)
        s_psi = np.sin(psi)
        
        # 計算旋轉矩陣
        R = np.array([
            [c_theta*c_psi, s_phi*s_theta*c_psi - c_phi*s_psi, c_phi*s_theta*c_psi + s_phi*s_psi],
            [c_theta*s_psi, s_phi*s_theta*s_psi + c_phi*c_psi, c_phi*s_theta*s_psi - s_phi*c_psi],
            [-s_theta, s_phi*c_theta, c_phi*c_theta]
        ])
        
        # 十字形配置的馬達位置 (Body frame)
        # 按順時針順序：前、右、後、左
        motor_pos_body = np.array([
            [self.l, 0, 0],   # 前
            [0, self.l, 0],   # 右
            [-self.l, 0, 0],  # 後
            [0, -self.l, 0]   # 左
        ])
        
        # 將馬達位置從Body frame轉換到Inertial frame
        motor_pos_world = np.array([x, y, z]) + np.dot(motor_pos_body, R.T)
        return motor_pos_world