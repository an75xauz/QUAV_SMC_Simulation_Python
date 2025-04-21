import numpy as np

class QuadrotorSMCController:
    def __init__(self, plant):
        self.plant = plant
        self.m = plant.m
        self.g = plant.g
        self.l = plant.l
        self.Ixx = plant.Ix
        self.Iyy = plant.Iy
        self.Izz = plant.Iz
        self.max_U1 = 30
        # Parameters setting
        # 高度控制參數 height
        self.lambda_alt = 2.8    # z  slope = 2.8
        self.eta_alt = 20.0      # z gain 20
        
        # 姿態控制參數 attitude
        self.lambda_att = 30.0   # attitude slope 30
        self.eta_att = 9.0       # attitude gain 9
        
        # 位置控制參數 position
        self.lambda_pos = 0.5  # pos slope 0.25
        self.eta_pos = 0.5      # pose gain 0.05
        
        # 平滑因子 smaller smoothly
        self.k_smooth = 50     # tanh 平滑因子 50.0
        self.k_smooth_pos = 50 # 位置控制平滑因子 50.0
        
        # angel limitation(rad)
        self.max_angle = 30 * np.pi/180
        
        self.target_position = np.zeros(3)
        self.target_attitude = np.zeros(3)
        
        self.prev_error_pos = np.zeros(3)
        self.prev_error_att = np.zeros(3)
        
    def reset(self):
        self.prev_error_pos = np.zeros(3)
        self.prev_error_att = np.zeros(3)
        
    def set_target_position(self, position):
        self.target_position = np.array(position)
        
    def set_target_attitude(self, attitude):
        self.target_attitude = np.array(attitude)
        
    def update(self, dt):
        # current state
        state = self.plant.get_state()
        position = state[:3]               # x, y, z
        velocity = state[3:6]              # vx, vy, vz
        attitude = state[6:9]              # φ, θ, ψ
        angular_velocity = state[9:12]     # p, q, r
        
        # 提取各個狀態
        x, y, z = position
        vx, vy, vz = velocity
        phi, theta, psi = attitude
        p, q, r = angular_velocity
        
        # 計算三角函數值
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # 防止分母為零的情況
        denom = cos_theta * cos_phi
        if abs(denom) < 1e-6:
            denom = np.sign(denom) * 1e-6
            
        z_d = self.target_position[2]
        e_z = z_d - z
        e_z_dot = 0 - vz  # 假設目標高度不變，其導數為0
        
        # z 
        s_alt = e_z_dot + self.lambda_alt * e_z
        
        # 等效控制項
        U1_eq = self.g - self.lambda_alt * vz
        
        # 切換控制項
        U1_sw = self.eta_alt * np.tanh(self.k_smooth * s_alt)
        
        # 總推力
        U1 = self.m * (U1_eq + U1_sw) / denom
        U1 = np.clip(U1, 0, self.max_U1)
        U1 = max(U1, 0.1)  # 確保推力為正且非零，避免除以零的錯誤
        
        # 保護因子，防止除以非常小的值
        safe_U1 = max(U1, 0.1)
        
        # X 方向控制 - 對應 theta 俯仰角
        x_d = self.target_position[0]
        e_x = x_d - x
        e_x_dot = 0 - vx  
        
        # X 方向滑動面
        s_x = e_x_dot + self.lambda_pos * e_x
        
        # 等效控制項和切換控制項
        # 注意：確保與plant.py中的動力學方程一致
        # ax = -F_total * s_theta / self.m，所以我們需要負的Ux產生正的theta，從而產生正的ax
        Ux_eq = -(self.m/safe_U1) * self.lambda_pos * (-vx)  
        Ux_sw = self.eta_pos * np.tanh(self.k_smooth_pos * s_x)
        
        # 虛擬控制輸入
        Ux = Ux_eq + Ux_sw
        
        # prevent arcsin has error trace back
        Ux = np.clip(Ux, -0.99, 0.99)
        theta_d = -np.arcsin(Ux)
        
        # Y 方向控制 - 對應 phi 滾轉角
        y_d = self.target_position[1]
        e_y = y_d - y
        e_y_dot = 0 - vy
        
        # Y 方向滑動面
        s_y = e_y_dot + self.lambda_pos * e_y
        
        # 保護因子，防止除以非常小的值
        safe_cos_theta = max(abs(cos_theta), 1e-3) * np.sign(cos_theta)
        
        # 等效控制項和切換控制項
        # 注意：確保與plant.py中的動力學方程一致
        # ay = F_total * c_theta * s_phi / self.m，所以我們需要正的Uy產生正的phi，從而產生正的ay
        Uy_eq = (self.m/safe_U1) * self.lambda_pos * (-vy) / safe_cos_theta 
        Uy_sw = self.eta_pos * np.tanh(self.k_smooth_pos * s_y)
        
        # 虛擬控制輸入
        Uy = Uy_eq + Uy_sw
        
        # prevent arcsin has error trace back
        Uy = np.clip(Uy, -0.99, 0.99)
        phi_d = np.arcsin(Uy)
        
        # 限制期望角度在安全範圍內
        # prevent too large angle
        phi_d = max(min(phi_d, self.max_angle), -self.max_angle)
        theta_d = max(min(theta_d, self.max_angle), -self.max_angle)
        
        # 期望偏航角
        psi_d = self.target_attitude[2]
        
      
        # 滾轉角控制
        e_phi = phi_d - phi
        s_phi = p + self.lambda_att * e_phi
        
        U2_eq = -((self.Iyy - self.Izz) * q * r + self.Ixx * self.lambda_att * p)
        U2_sw = self.Ixx * self.eta_att * np.tanh(self.k_smooth * s_phi)
        U2 = U2_eq + U2_sw
        
        # 俯仰角控制
        e_theta = theta_d - theta
        s_theta = q + self.lambda_att * e_theta
        
        U3_eq = -((self.Izz - self.Ixx) * p * r + self.Iyy * self.lambda_att * q)
        U3_sw = self.Iyy * self.eta_att * np.tanh(self.k_smooth * s_theta)
        U3 = U3_eq + U3_sw
        
        # 偏航角控制
        e_psi = psi_d - psi
        s_psi = r + self.lambda_att * e_psi
        
        U4_eq = -((self.Ixx - self.Iyy) * p * q + self.Izz * self.lambda_att * r)
        U4_sw = self.Izz * self.eta_att * np.tanh(self.k_smooth * s_psi)
        U4 = U4_eq + U4_sw
        
        # 確保控制輸入不包含 NaN 或 Inf
        # Make sure input will not contain NaN or Inf
        control_out = np.array([U2, U3, U4, U1])
        control_out = np.nan_to_num(control_out, nan=0.0, posinf=50.0, neginf=-50.0)
        
        # 更新內部狀態
        self.prev_error_pos = np.array([e_x, e_y, e_z])
        self.prev_error_att = np.array([e_phi, e_theta, e_psi])
        # control_out: 
        # U1: F_total 
        # U2 U3 U4: torqe
        return control_out
