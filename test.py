import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import art3d  # 添加此導入

class QuadrotorModel:
    def __init__(self):
        # 四旋翼參數
        self.m = 1.0  # 質量 (kg)
        self.g = 9.81  # 重力加速度 (m/s^2)
        self.Ixx = 0.01  # x軸轉動慣量
        self.Iyy = 0.01  # y軸轉動慣量
        self.Izz = 0.02  # z軸轉動慣量
        self.l = 0.2  # 旋翼到重心的距離 (m)
        self.kf = 1e-5  # 推力係數
        self.km = 1e-6  # 力矩係數
        
        # 狀態變數 - 確保所有數組都是浮點類型
        self.pos = np.zeros(3, dtype=np.float64)       # 位置 [x, y, z]
        self.vel = np.zeros(3, dtype=np.float64)       # 速度 [vx, vy, vz]
        self.angles = np.zeros(3, dtype=np.float64)    # 姿態角 [roll, pitch, yaw]
        self.omega = np.zeros(3, dtype=np.float64)     # 角速度 [wx, wy, wz]
        
        # 歷史記錄（用於繪圖）
        self.pos_history = []
        self.angles_history = []
        self.time_history = []
        
    def reset(self, pos=None, vel=None, angles=None, omega=None):
        """重置四旋翼狀態"""
        if pos is not None:
            self.pos = np.array(pos, dtype=np.float64)
        else:
            self.pos = np.zeros(3, dtype=np.float64)
            
        if vel is not None:
            self.vel = np.array(vel, dtype=np.float64)
        else:
            self.vel = np.zeros(3, dtype=np.float64)
            
        if angles is not None:
            self.angles = np.array(angles, dtype=np.float64)
        else:
            self.angles = np.zeros(3, dtype=np.float64)
            
        if omega is not None:
            self.omega = np.array(omega, dtype=np.float64)
        else:
            self.omega = np.zeros(3, dtype=np.float64)
            
        # 清空歷史記錄
        self.pos_history = []
        self.angles_history = []
        self.time_history = []
        
    def update(self, forces, dt):
        """更新四旋翼狀態"""
        # 解構推力和力矩
        F_total, M = self._calculate_forces_moments(forces)
        
        # 更新位置和速度
        # 計算加速度（包括重力）
        R = self._rotation_matrix()
        thrust_body = np.array([0, 0, F_total])
        thrust_world = R @ thrust_body
        
        acc = thrust_world / self.m
        acc[2] -= self.g  # 減去重力加速度
        
        # 歐拉積分更新速度和位置
        self.vel += acc * dt
        self.pos += self.vel * dt
        
        # 更新姿態角和角速度
        # 計算角加速度
        omega_dot = np.zeros(3)
        omega_dot[0] = (M[0] - (self.Izz - self.Iyy) * self.omega[1] * self.omega[2]) / self.Ixx
        omega_dot[1] = (M[1] - (self.Ixx - self.Izz) * self.omega[0] * self.omega[2]) / self.Iyy
        omega_dot[2] = (M[2] - (self.Iyy - self.Ixx) * self.omega[0] * self.omega[1]) / self.Izz
        
        # 歐拉積分更新角速度
        self.omega += omega_dot * dt
        
        # 根據角速度更新姿態角（歐拉角）
        # 注意：這是簡化版，實際應使用四元數以避免萬向鎖問題
        phi, theta, psi = self.angles
        p, q, r = self.omega
        
        phi_dot = p + np.sin(phi) * np.tan(theta) * q + np.cos(phi) * np.tan(theta) * r
        theta_dot = np.cos(phi) * q - np.sin(phi) * r
        psi_dot = np.sin(phi) / np.cos(theta) * q + np.cos(phi) / np.cos(theta) * r
        
        self.angles[0] += phi_dot * dt
        self.angles[1] += theta_dot * dt
        self.angles[2] += psi_dot * dt
        
        # 記錄歷史數據
        self.pos_history.append(self.pos.copy())
        self.angles_history.append(self.angles.copy())
        
    def _rotation_matrix(self):
        """計算從機體坐標系到世界坐標系的旋轉矩陣"""
        phi, theta, psi = self.angles
        
        # 旋轉矩陣
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])
        
        R_y = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        
        R_z = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])
        
        R = R_z @ R_y @ R_x
        return R
    
    def _calculate_forces_moments(self, motor_forces):
        """根據四個馬達的推力計算總推力和力矩"""
        f1, f2, f3, f4 = motor_forces
        
        # 總推力
        F_total = f1 + f2 + f3 + f4
        
        # 力矩
        tau_phi = self.l * (f2 - f4)                            # roll力矩
        tau_theta = self.l * (f3 - f1)                          # pitch力矩
        tau_psi = self.km/self.kf * (f1 - f2 + f3 - f4)         # yaw力矩
        
        return F_total, np.array([tau_phi, tau_theta, tau_psi])


class SMController:
    """滑模控制器"""
    def __init__(self, quad):
        self.quad = quad
        
        # 外環控制參數（位置控制）
        self.lambda_pos = np.array([1.0, 1.0, 1.0])   # 滑動面參數
        self.eta_pos = np.array([3.0, 3.0, 3.0])      # 接近律增益 - 增加以加速收斂
        self.phi_pos = np.array([0.1, 0.1, 0.1])      # 邊界層厚度
        
        # 內環控制參數（姿態控制）
        self.lambda_att = np.array([5.0, 5.0, 5.0])   # 滑動面參數
        self.eta_att = np.array([5.0, 5.0, 5.0])      # 接近律增益 - 增加以改善姿態控制
        self.phi_att = np.array([0.05, 0.05, 0.1])    # 邊界層厚度
        
    def sat(self, s, phi):
        """飽和函數（用於減少抖振）"""
        # 處理標量輸入
        if np.isscalar(s):
            if abs(s) <= phi:
                return s / phi
            else:
                return np.sign(s)
        
        # 處理向量輸入
        result = np.zeros_like(s)
        for i in range(len(s)):
            if abs(s[i]) <= phi[i]:
                result[i] = s[i] / phi[i]
            else:
                result[i] = np.sign(s[i])
        return result
        
    def control(self, desired_pos, desired_yaw, dt):
        """實現嵌套控制結構：外環（位置控制）產生期望姿態，內環（姿態控制）產生馬達指令"""
        # 當前狀態
        pos = self.quad.pos
        vel = self.quad.vel
        angles = self.quad.angles
        omega = self.quad.omega
        
        # Step 1: 外環控制 - 位置控制
        # 計算位置誤差和速度誤差
        pos_error = pos - desired_pos
        vel_error = vel - np.zeros(3)  # 假設期望速度為零
        
        # 構建位置滑動面
        s_pos = vel_error + self.lambda_pos * pos_error
        
        # 外環控制律（計算期望加速度）
        desired_acc = np.zeros(3)
        for i in range(3):
            desired_acc[i] = -self.lambda_pos[i] * vel_error[i] - self.eta_pos[i] * self.sat(s_pos[i], self.phi_pos[i])
        
        # 加上重力補償
        desired_acc[2] += self.quad.g
        
        # Step 2: 從期望加速度計算期望姿態（roll和pitch）
        # 總推力大小
        thrust_magnitude = self.quad.m * np.sqrt(desired_acc[0]**2 + desired_acc[1]**2 + desired_acc[2]**2)
        
        # 計算期望的roll和pitch角（簡化版）
        desired_angles = np.zeros(3)
        if thrust_magnitude > 0:
            desired_angles[0] = np.arcsin(-self.quad.m * desired_acc[1] / thrust_magnitude)  # 期望roll角
            desired_angles[1] = np.arcsin(self.quad.m * desired_acc[0] / (thrust_magnitude * np.cos(desired_angles[0])))  # 期望pitch角
        desired_angles[2] = desired_yaw  # 期望yaw角
        
        # Step 3: 內環控制 - 姿態控制
        # 角度誤差和角速度誤差
        angle_error = angles - desired_angles
        omega_error = omega - np.zeros(3)  # 假設期望角速度為零
        
        # 構建姿態滑動面
        s_att = omega_error + self.lambda_att * angle_error
        
        # 內環控制律（計算期望力矩）
        desired_moments = np.zeros(3)
        for i in range(3):
            desired_moments[i] = -self.lambda_att[i] * omega_error[i] - self.eta_att[i] * self.sat(s_att[i], self.phi_att[i])
        
        # 轉換為摺合的慣性項（簡化版）
        tau_phi = desired_moments[0] * self.quad.Ixx
        tau_theta = desired_moments[1] * self.quad.Iyy
        tau_psi = desired_moments[2] * self.quad.Izz
        
        # Step 4: 計算各馬達的推力
        # 總推力
        f_total = thrust_magnitude
        
        # 解四個馬達的推力方程
        l = self.quad.l
        k_ratio = self.quad.km / self.quad.kf
        
        # 馬達推力分配矩陣
        A = np.array([
            [1, 1, 1, 1],               # 總推力
            [0, l, 0, -l],              # roll力矩
            [-l, 0, l, 0],              # pitch力矩
            [k_ratio, -k_ratio, k_ratio, -k_ratio]  # yaw力矩
        ])
        
        b = np.array([f_total, tau_phi, tau_theta, tau_psi])
        
        # 求解馬達推力
        motor_forces = np.linalg.solve(A, b)
        
        # 確保馬達推力為非負
        motor_forces = np.maximum(motor_forces, 0)
        
        return motor_forces


# 期望軌跡函數 - 全局定義，以便在plot_results中使用
def desired_position(t):
    if t < 2.0:
        return np.array([0, 0, 2])  # 首先上升到2米高度
    elif t < 5.0:
        # 從(0,0,2)線性移動到(1,1,2)
        progress = (t - 2.0) / 3.0
        return np.array([progress, progress, 2])
    else:
        return np.array([1, 1, 2])  # 保持在(1,1,2)

def simulate_quadrotor():
    # 創建四旋翼模型
    quad = QuadrotorModel()
    
    # 初始位置
    quad.reset(pos=[0, 0, 0])
    
    # 創建控制器
    controller = SMController(quad)
    
    # 模擬參數
    t_final = 10.0  # 模擬總時間
    dt = 0.01       # 時間步長
    
    # 期望偏航角 - 保持為零
    desired_yaw = 0.0
    
    # 添加一些干擾
    def disturbance(t):
        # 在t=7秒時添加一個脈衝干擾
        if 7.0 <= t < 7.2:
            return np.array([0.5, 0.5, 0])
        else:
            return np.zeros(3)
    
    # 模擬循環
    t = 0.0
    while t < t_final:
        # 獲取當前期望位置
        des_pos = desired_position(t)
        
        # 計算控制輸入
        motor_forces = controller.control(des_pos, desired_yaw, dt)
        
        # 添加干擾
        quad.vel += disturbance(t) * dt
        
        # 更新四旋翼狀態
        quad.update(motor_forces, dt)
        
        # 記錄時間
        quad.time_history.append(t)
        
        # 更新時間
        t += dt
        
        # 每100步打印一次進度
        if len(quad.time_history) % 100 == 0:
            print(f"模擬進度: {t:.1f}/{t_final} 秒 ({t/t_final*100:.0f}%)")
    
    print(f"模擬完成，記錄了 {len(quad.pos_history)} 個數據點")
    return quad


def plot_results(quad):
    # 轉換歷史記錄為numpy數組
    pos_history = np.array(quad.pos_history)
    angles_history = np.array(quad.angles_history)
    time_history = np.array(quad.time_history)
    
    # 打印軌跡信息進行調試
    print(f"軌跡起點: ({pos_history[0][0]:.2f}, {pos_history[0][1]:.2f}, {pos_history[0][2]:.2f})")
    print(f"軌跡終點: ({pos_history[-1][0]:.2f}, {pos_history[-1][1]:.2f}, {pos_history[-1][2]:.2f})")
    print(f"X方向最大值: {np.max(pos_history[:, 0]):.2f}, 最小值: {np.min(pos_history[:, 0]):.2f}")
    print(f"Y方向最大值: {np.max(pos_history[:, 1]):.2f}, 最小值: {np.min(pos_history[:, 1]):.2f}")
    
    # 創建圖形
    fig = plt.figure(figsize=(15, 10))
    
    # 3D軌跡
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(pos_history[:, 0], pos_history[:, 1], pos_history[:, 2], 'b')
    # 添加起點和終點標記
    ax1.scatter(pos_history[0, 0], pos_history[0, 1], pos_history[0, 2], c='g', s=100, marker='o', label='Start')
    ax1.scatter(pos_history[-1, 0], pos_history[-1, 1], pos_history[-1, 2], c='r', s=100, marker='x', label='End')
    # 添加中間點標記
    mid_index = len(pos_history) // 2
    ax1.scatter(pos_history[mid_index, 0], pos_history[mid_index, 1], pos_history[mid_index, 2], 
                c='m', s=100, marker='*', label=f'Middle (t={time_history[mid_index]:.1f}s)')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True)
    
    # 位置隨時間變化
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(time_history, pos_history[:, 0], 'r', label='X')
    ax2.plot(time_history, pos_history[:, 1], 'g', label='Y')
    ax2.plot(time_history, pos_history[:, 2], 'b', label='Z')
    
    # 添加期望位置
    desired_pos = np.array([desired_position(t) for t in time_history])
    ax2.plot(time_history, desired_pos[:, 0], 'r--', label='X desired')
    ax2.plot(time_history, desired_pos[:, 1], 'g--', label='Y desired')
    ax2.plot(time_history, desired_pos[:, 2], 'b--', label='Z desired')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position vs Time')
    ax2.legend()
    ax2.grid(True)
    
    # 姿態角隨時間變化
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(time_history, angles_history[:, 0], 'r', label='Roll')
    ax3.plot(time_history, angles_history[:, 1], 'g', label='Pitch')
    ax3.plot(time_history, angles_history[:, 2], 'b', label='Yaw')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Angle (rad)')
    ax3.set_title('Attitude vs Time')
    ax3.legend()
    ax3.grid(True)
    
    # XY平面軌跡
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(pos_history[:, 0], pos_history[:, 1], 'k')
    # 添加起點和終點標記
    ax4.scatter(pos_history[0, 0], pos_history[0, 1], c='g', s=100, marker='o', label='Start')
    ax4.scatter(pos_history[-1, 0], pos_history[-1, 1], c='r', s=100, marker='x', label='End')
    # 添加中間點標記
    ax4.scatter(pos_history[mid_index, 0], pos_history[mid_index, 1], 
               c='m', s=100, marker='*', label=f'Middle (t={time_history[mid_index]:.1f}s)')
    
    # 添加一些時間點標記
    for t in [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0]:
        idx = np.abs(time_history - t).argmin()
        ax4.annotate(f"{t}s", 
                   (pos_history[idx, 0], pos_history[idx, 1]),
                   textcoords="offset points",
                   xytext=(0,10),
                   ha='center')
    
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_title('XY Trajectory')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()


def animate_quadrotor(quad):
    # 轉換歷史記錄為numpy數組
    pos_history = np.array(quad.pos_history)
    angles_history = np.array(quad.angles_history)
    
    # 檢查是否有足夠的數據
    if len(pos_history) == 0:
        print("沒有足夠的數據用於動畫顯示")
        return
    
    # 打印軌跡信息進行調試
    print(f"軌跡起點: ({pos_history[0][0]:.2f}, {pos_history[0][1]:.2f}, {pos_history[0][2]:.2f})")
    print(f"軌跡終點: ({pos_history[-1][0]:.2f}, {pos_history[-1][1]:.2f}, {pos_history[-1][2]:.2f})")
    
    # 檢查x和y方向是否有變化
    x_change = np.max(pos_history[:, 0]) - np.min(pos_history[:, 0])
    y_change = np.max(pos_history[:, 1]) - np.min(pos_history[:, 1])
    print(f"X方向變化: {x_change:.2f} m")
    print(f"Y方向變化: {y_change:.2f} m")
    
    # 創建圖形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 設置軸範圍 - 確保能顯示完整軌跡
    x_min, x_max = np.min(pos_history[:, 0]) - 0.5, np.max(pos_history[:, 0]) + 0.5
    y_min, y_max = np.min(pos_history[:, 1]) - 0.5, np.max(pos_history[:, 1]) + 0.5
    z_min, z_max = np.min(pos_history[:, 2]) - 0.5, np.max(pos_history[:, 2]) + 0.5
    
    # 確保軸的範圍至少是1m
    x_range = max(1.0, x_max - x_min)
    y_range = max(1.0, y_max - y_min)
    z_range = max(1.0, z_max - z_min)
    
    # 設置對稱的軸範圍
    max_range = max(x_range, y_range, z_range) / 2
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2
    
    ax.set_xlim([x_mid - max_range, x_mid + max_range])
    ax.set_ylim([y_mid - max_range, y_mid + max_range])
    ax.set_zlim([z_mid - max_range, z_mid + max_range])
    
    # 四旋翼的旋臂長度
    arm_length = quad.l
    
    # 畫出完整軌跡（這是靜態的，不是動畫的一部分）
    ax.plot(pos_history[:, 0], pos_history[:, 1], pos_history[:, 2], 'b-', alpha=0.7, label='Complete Trajectory')
    
    # 添加起點和終點標記
    ax.scatter(pos_history[0, 0], pos_history[0, 1], pos_history[0, 2], c='g', s=100, label='Start')
    ax.scatter(pos_history[-1, 0], pos_history[-1, 1], pos_history[-1, 2], c='r', s=100, label='End')
    
    # 初始化四旋翼的可視化元素
    # 當前軌跡點
    current_point, = ax.plot([], [], [], 'mo', markersize=8)
    
    # 四個旋臂
    line_arms = [ax.plot([], [], [], 'k-', linewidth=2)[0] for _ in range(4)]
    
    # 四個旋翼
    rotor_points = ax.scatter([], [], [], c='r', s=50)
    
    # 四旋翼中心
    center_point = ax.scatter([], [], [], c='b', s=100)
    
    # 設置標籤
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Quadrotor Animation')
    ax.legend()
    
    # 添加軸網格
    ax.grid(True)
    
    # 設置視角以便更好地查看XY平面運動
    ax.view_init(elev=30, azim=45)
    
    def update(frame):
        # 確保frame在有效範圍內
        frame = min(frame, len(pos_history) - 1)
        
        # 獲取當前位置和姿態
        pos = pos_history[frame]
        angles = angles_history[frame]
        
        # 更新當前點
        current_point.set_data([pos[0]], [pos[1]])
        current_point.set_3d_properties([pos[2]])
        
        # 更新中心點
        center_point._offsets3d = ([pos[0]], [pos[1]], [pos[2]])
        
        # 計算旋轉矩陣
        phi, theta, psi = angles
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])
        R_y = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        R_z = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])
        R = R_z @ R_y @ R_x
        
        # 定義四旋翼的旋臂向量（機體坐標系）
        arms_body = np.array([
            [arm_length, 0, 0],  # 前
            [0, arm_length, 0],  # 右
            [-arm_length, 0, 0], # 後
            [0, -arm_length, 0]  # 左
        ])
        
        # 轉換到世界坐標系並加上當前位置
        arms_world = np.array([R @ arm + pos for arm in arms_body])
        
        # 更新四旋翼的旋臂
        for i, line in enumerate(line_arms):
            x_data = [pos[0], arms_world[i, 0]]
            y_data = [pos[1], arms_world[i, 1]]
            z_data = [pos[2], arms_world[i, 2]]
            line.set_data(x_data, y_data)
            line.set_3d_properties(z_data)
        
        # 更新旋翼點
        rotor_points._offsets3d = (arms_world[:, 0], arms_world[:, 1], arms_world[:, 2])
        
        return [current_point, center_point, rotor_points] + line_arms
    
    # 創建動畫
    frames = len(pos_history)
    step = max(1, frames // 200)  # 確保不超過200幀
    frame_indices = range(0, frames, step)
    
    print(f"創建動畫: 總幀數={frames}, 使用步長={step}, 實際顯示幀數={len(frame_indices)}")
    
    ani = FuncAnimation(fig, update, frames=frame_indices, interval=50, blit=False)
    
    plt.tight_layout()
    plt.show()


def main():
    # 運行模擬
    print("開始四旋翼滑模控制模擬...")
    quad = simulate_quadrotor()
    
    # 驗證是否有數據
    if len(quad.pos_history) == 0:
        print("警告：模擬沒有產生任何數據點!")
        return
    
    print(f"模擬完成，共產生 {len(quad.pos_history)} 個數據點")
    
    # 繪製結果
    plot_results(quad)
    
    try:
        # 動畫演示
        print("正在生成動畫...")
        from mpl_toolkits.mplot3d import art3d  # 導入3D補丁所需模塊
        animate_quadrotor(quad)
    except Exception as e:
        print(f"動畫生成失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()