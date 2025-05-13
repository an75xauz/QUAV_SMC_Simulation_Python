"""環境模組，處理模擬環境的創建與管理"""
import numpy as np
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.Logger import Logger

class DroneQuadrotorPlant:
    """為滑模控制器提供四旋翼植物模型接口"""
    def __init__(self, drone_model):
        # 根據不同的drone_model可以設定不同的參數
        # 預設使用Crazyflie 2.x參數
        self.m = 0.027  # kg, Crazyflie 2.x 重量
        self.g = 9.81   # m/s^2
        # 對角慣性矩陣 [Ixx, Iyy, Izz]
        self.J = np.array([1.4e-5, 1.4e-5, 2.17e-5])  # 慣性矩陣
        self.state = np.zeros(12)  # [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
        
    def get_state(self):
        """返回無人機狀態"""
        return self.state
        
    def update_state(self, obs):
        """從PyBullet觀測更新狀態
        
        參數:
            obs: PyBullet的觀測值，包含位置、速度、姿態和角速度
            
        返回:
            更新後的狀態
        """
        pos = obs[0:3]         # 位置 [x, y, z]
        vel = obs[3:6]         # 速度 [vx, vy, vz]
        rpy = obs[6:9]         # 姿態角 [phi, theta, psi]
        ang_vel = obs[10:13]   # 角速度 [p, q, r]
        
        # 更新狀態向量
        self.state = np.concatenate([pos, vel, rpy, ang_vel])
        return self.state

def create_env(drone_model, num_drones, initial_xyzs, initial_rpys, physics, 
              simulation_freq_hz, control_freq_hz, gui, record_video, 
              obstacles, user_debug_gui):
    """創建模擬環境
    
    參數:
        drone_model: 無人機型號
        num_drones: 無人機數量
        initial_xyzs: 初始位置
        initial_rpys: 初始姿態
        physics: 物理引擎類型
        simulation_freq_hz: 模擬頻率
        control_freq_hz: 控制頻率
        gui: 是否啟用GUI
        record_video: 是否記錄視頻
        obstacles: 是否添加障礙物
        user_debug_gui: 是否啟用用戶調試GUI
        
    返回:
        創建的環境實例
    """
    env = CtrlAviary(
        drone_model=drone_model,
        num_drones=num_drones,
        initial_xyzs=initial_xyzs,
        initial_rpys=initial_rpys,
        physics=physics,
        neighbourhood_radius=10,
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        record=record_video,
        obstacles=obstacles,
        user_debug_gui=user_debug_gui
    )
    return env

def create_logger(control_freq_hz, num_drones, output_folder, colab):
    """創建日誌記錄器
    
    參數:
        control_freq_hz: 控制頻率
        num_drones: 無人機數量
        output_folder: 輸出資料夾
        colab: 是否在Colab環境中運行
        
    返回:
        創建的日誌記錄器實例
    """
    logger = Logger(
        logging_freq_hz=control_freq_hz,
        num_drones=num_drones,
        output_folder=output_folder,
        colab=colab
    )
    return logger