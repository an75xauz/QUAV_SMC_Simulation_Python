import numpy as np

class SlidingModeController:
    def __init__(self, c, k, output_limits=None):
        """
        c: 控制滑動面趨近的速度
        k: 控制切換強度（越大系統收斂越快）
        """
        self.c = c
        self.k = k
        self.output_limits = output_limits
        self.prev_error = 0

    def control(self, target, current, dt):
        """
        target: 目標值（期望姿態或高度）
        current: 目前實際值
        current_rate: 目前值的導數（速度或角速度）
        """
        # 誤差
        error = target - current
        d_error = (error - self.prev_error)/dt
        # 建立滑動面
        s = self.c * error + d_error 

        # SMC 控制律（等價控制項 + 切換控制項）
        u_eq = self.c * d_error
        u_sw = self.k * np.tanh(s/1) 

        u = u_eq + u_sw
        if self.output_limits is not None:
            u = np.clip(u, self.output_limits[0], self.output_limits[1])
        self.prev_error = error

        return u
    def reset(self):
        self.prev_error = 0


class QuadrotorSMCController:
    def __init__(self, plant):
        self.plant = plant

        # 建立滑模控制器（調整參數以改善性能）
        # 降低水平控制的增益，並增加更嚴格的輸出限制
        self.x_smc = SlidingModeController(c = 2.0, k=4.0, output_limits=[-20, 20]) # 降低增益
        self.y_smc = SlidingModeController(c = 2.0, k=4.0, output_limits=[-20, 20]) # 降低增益
        self.z_smc = SlidingModeController(c = 2.0, k=4.0, output_limits=[-500, 500])

        self.phi_smc   = SlidingModeController(c = 2.0, k=5.0, output_limits=[-50, 50])   # φ：x軸旋轉
        self.theta_smc = SlidingModeController(c = 2.0, k=5.0, output_limits=[-50, 50])   # θ：y軸旋轉
        self.psi_smc   = SlidingModeController(c = 2.0, k=5.0, output_limits=[-100, 100])   # ψ：z軸旋轉

        self.p_smc = SlidingModeController(c = 2.0, k=4.0, output_limits=[-100, 100])
        self.q_smc = SlidingModeController(c = 2.0, k=4.0, output_limits=[-100, 100])
        self.r_smc = SlidingModeController(c = 2.0, k=4.0, output_limits=[-100, 100])

        # 目標值
        self.target_attitude = np.zeros(3)   # φ, θ, ψ
        self.target_position = np.zeros(3)
        self.target_z = 0
        self.prev_trans_matrix = np.eye(3)  # 初始化為單位矩陣，防止第一次計算的數值問題
        
        # 安全限制
        self.max_phi_theta = np.radians(30)  # 限制最大傾角為30度
        
    def reset(self):
        self.x_smc.reset()
        self.y_smc.reset()
        self.z_smc.reset()
        self.phi_smc.reset()
        self.theta_smc.reset()
        self.psi_smc.reset()
        self.p_smc.reset()
        self.q_smc.reset()
        self.r_smc.reset()
        self.prev_trans_matrix = np.eye(3)

    def set_target_attitude(self, attitude):
        self.target_attitude = np.array(attitude)

    def set_target_position(self, position):
        self.target_position = np.array(position)

    def update(self, dt):
        # 讀取狀態
        state = self.plant.get_state()
        position = state[:3]               # x, y, z
        velocity = state[3:6]              # vx, vy, vz
        attitude = state[6:9]              # φ, θ, ψ
        angular_velocity = state[9:12]     # p, q, r
        phi     = attitude[0]
        theta   = attitude[1]
        psi     = attitude[2]
        
        # 防止角度過大導致數值問題
        safe_phi = np.clip(phi, -np.pi/3, np.pi/3)
        safe_theta = np.clip(theta, -np.pi/3, np.pi/3)
        
        # 計算三角函數值，使用安全值
        c_phi = np.cos(safe_phi)
        s_phi = np.sin(safe_phi)
        c_theta = np.cos(safe_theta)
        s_theta = np.sin(safe_theta)
        
        # 使用平滑的方法計算 tan 和 sec，防止數值不穩定
        # 避免除以接近零的值
        tan_theta = s_theta / (c_theta + 1e-6)  # 添加小值防止除零
        sec_theta = 1 / (c_theta + 1e-6)        # 添加小值防止除零

        # 轉換矩陣
        transition_matrix = np.array([
                                [1, s_phi*tan_theta, c_phi*tan_theta],
                                [0, c_phi, -s_phi],
                                [0, s_phi*sec_theta, c_phi*sec_theta]
        ])
        
        # 確保轉換矩陣的數值穩定性
        if np.any(np.isnan(transition_matrix)) or np.any(np.isinf(transition_matrix)):
            # 如果有不穩定值，使用前一個矩陣
            transition_matrix = self.prev_trans_matrix
            
        # 計算矩陣導數，使用更穩定的方法
        if dt > 1e-6:  # 避免除以極小值
            trans_dot = (transition_matrix - self.prev_trans_matrix)/dt
        else:
            trans_dot = np.zeros((3,3))  # 如果dt極小，假設變化為零

        # 控制項計算（SMC）
        # 增加數值穩定性，防止除零
        cos_product = np.cos(safe_phi) * np.cos(safe_theta)
        hover_thrust = self.plant.m * self.plant.g/max(cos_product, 0.7)  # 設定下限，防止除以小值
        
        # 高度控制
        z_thrust = self.z_smc.control(self.target_position[2], position[2], dt)

        # 水平位置控制生成姿態目標
        # 對於水平移動，需降低控制增益以改善穩定性
        phi_target = self.y_smc.control(self.target_position[1], position[1], dt)
        theta_target = self.x_smc.control(self.target_position[0], position[0], dt)
        
        # 限制姿態角度在安全範圍內
        phi_target = np.clip(phi_target, -self.max_phi_theta, self.max_phi_theta)
        theta_target = np.clip(theta_target, -self.max_phi_theta, self.max_phi_theta)
        
        psi_target = self.target_attitude[2]

        # 姿態控制（輸出三軸力矩）
        phi_error = self.phi_smc.control(phi_target, attitude[0], dt)
        theta_error = self.theta_smc.control(theta_target, attitude[1], dt)
        psi_error = self.psi_smc.control(psi_target, attitude[2], dt)

        p_target = phi_error
        q_target = theta_error
        r_target = psi_error

        tau_x = self.p_smc.control(p_target, angular_velocity[0], dt)
        tau_y = self.q_smc.control(q_target, angular_velocity[1], dt)
        tau_z = self.r_smc.control(r_target, angular_velocity[2], dt)

        # 嘗試計算滑模控制力矩
        try:
            # 添加更多的數值穩定性檢查
            tau_SMC = np.array([tau_x, tau_y, tau_z]).T - np.dot(trans_dot, angular_velocity)
            
            # 檢查是否能安全地求逆矩陣
            # 計算條件數來檢查矩陣是否接近奇異
            if np.linalg.cond(transition_matrix) < 1e6:  # 條件數較小，矩陣較穩定
                inv_trans_matrix = np.linalg.inv(transition_matrix)
                tau = np.cross(angular_velocity, np.dot(self.plant.J, angular_velocity).T) + (
                    np.dot(np.dot(self.plant.J, inv_trans_matrix), tau_SMC)
                )
            else:
                # 矩陣接近奇異，使用簡化計算
                tau = np.array([tau_x, tau_y, tau_z])
        except:
            # 如果計算出錯，使用直接控制輸出
            tau = np.array([tau_x, tau_y, tau_z])
        
        # 確保力矩不包含 NaN 或 Inf
        tau = np.nan_to_num(tau, nan=0.0, posinf=50.0, neginf=-50.0)
        
        tau_phi, tau_theta, tau_psi = tau
        Fz = hover_thrust + z_thrust
        
        # 確保總推力不為負
        Fz = max(Fz, 0.1)
        
        self.prev_trans_matrix = transition_matrix
        
        return np.array([tau_phi, tau_theta, tau_psi, Fz])