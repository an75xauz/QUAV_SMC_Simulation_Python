import gym
import numpy as np
from gym import spaces

from simulation.plant import QuadrotorPlant
from simulation.controller import QuadrotorSMCController
from rl.config import MAX_STEPS

class QuadrotorEnv(gym.Env):
    """四旋翼機強化學習環境，使用SMC控制器參數作為動作"""
    
    def __init__(self, 
                 initial_position=[0, 0, 0],
                 target_position=[1, 1, 2],
                 random_target=True,
                 target_range=[3.0, 3.0, 3.0],
                 max_steps=MAX_STEPS,
                 dt=0.05):
        """初始化環境"""
        super(QuadrotorEnv, self).__init__()
        
        # 設置四旋翼機物理模型和控制器
        self.plant = QuadrotorPlant()
        self.controller = QuadrotorSMCController(self.plant)

        self.random_target = random_target
        self.target_range = target_range
        
        # 儲存環境參數
        self.initial_position = np.array(initial_position, dtype=np.float32)
        self.target_position = np.array(target_position, dtype=np.float32)
        self.max_steps = max_steps
        self.dt = dt
        
        # 設置控制器目標
        self.controller.set_target_position(target_position)
        
        # 定義動作空間 (SMC控制器參數)
        # 各參數的範圍需要根據實際情況調整
        # [lambda_att, eta_att, lambda_pos, eta_pos]
        self.action_space = spaces.Box(
            low=np.array([ 1.0, 1.0, 0.1, 0.1]),  # 最小值
            high=np.array([ 40.0, 40.0, 1, 1]),  # 最大值
            dtype=np.float32
        )
        
        # 定義觀測空間 (狀態: x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, 
    #             控制器參數: lambda_att, eta_att, lambda_pos, eta_pos,
    #             normalized_distance)
        max_pos = 10.0
        max_vel = 5.0
        max_angle = np.pi
        max_ang_vel = 5.0
        max_lambda = 40.0
        max_eta = 40.0
        
        self.observation_space = spaces.Box(
            low=np.array([
                -max_pos, -max_pos, -max_pos,
                -max_vel, -max_vel, -max_vel,
                -max_angle, -max_angle, -max_angle,
                -max_ang_vel, -max_ang_vel, -max_ang_vel,
                1.0, 1.0, 0.1, 0.1,  # 控制器參數的最小值
                0.0                  # 歸一化距離的最小值
            ]),
            high=np.array([
                max_pos, max_pos, max_pos,
                max_vel, max_vel, max_vel,
                max_angle, max_angle, max_angle,
                max_ang_vel, max_ang_vel, max_ang_vel,
                max_lambda, max_eta, 1.0, 1.0,  # 控制器參數的最大值
                10.0                             # 歸一化距離的最大值 (理論上最大為1，但設大一點更安全)
            ]),
            dtype=np.float32
        )
        
        # 用於追蹤任務完成情況
        self.reached_target = False
        self.step_sucess = 0
        
        # 重置環境
        self.reset()
        
    def reset(self):
        """重置環境到初始狀態"""
        # 重置步數計數器和狀態
        self.step_count = 0
        self.reached_target = False
        self.step_sucess = 0
        # 初始狀態
        initial_state = np.zeros(12)
        initial_state[:3] = self.initial_position
        initial_state[6:9] = (np.random.rand(3) - 0.5) * 0.2  # 小隨機角度
        
        # 重置四旋翼機物理模型
        self.plant.reset(initial_state)

        if self.random_target:
            # 範例：在3x3x3的空間內隨機生成目標
            self.target_position = np.array([
                np.random.uniform(-3.0, 3.0),
                np.random.uniform(-3.0, 3.0),
                np.random.uniform(0.0, 4.0)  # 確保高度為正
            ], dtype=np.float32)

        self.initial_distance = np.linalg.norm(self.initial_position - self.target_position)

        print("----------")
        print(f"initial position:{self.initial_position}")
        print(f"target position:{self.target_position}")
        # 設置控制器目標
        self.controller.set_target_position(self.target_position)

        self.controller_params = np.array([
            self.controller.lambda_att,
            self.controller.eta_att,
            self.controller.lambda_pos,
            self.controller.eta_pos
        ], dtype=np.float32)
        # 返回初始觀測值
        return self._get_obs()
    
    def step(self, action):
        """執行一步模擬，使用SMC控制器參數作為動作
        
        Args:
            action: SMC控制器參數 [lambda_alt, eta_alt, lambda_att, eta_att, lambda_pos, eta_pos]
            
        Returns:
            observation, reward, done, info 的元組
        """
        # 將動作應用到控制器參數
        # self.controller.lambda_alt = action[0]
        # self.controller.eta_alt = action[1]
        self.controller.lambda_att = action[0]
        self.controller.eta_att = action[1]
        self.controller.lambda_pos = action[2]
        self.controller.eta_pos = action[3]

        self.controller_params = np.array([action[0], action[1], action[2], action[3]], dtype=np.float32)
        # 使用控制器計算控制輸入
        control_input = self.controller.update(self.dt)
        
        # 將控制輸入應用到四旋翼機
        self.plant.step(self.dt, control_input)
        
        # 增加步數計數
        self.step_count += 1
        
        # 獲取當前狀態
        obs = self._get_obs()
        
        # 計算獎勵
        reward = self._compute_reward(obs, action)
        
        # 檢查回合是否結束
        done = self._is_done(obs)
        
        # 額外信息
        position = obs[:3]
        distance = np.linalg.norm(position - self.target_position)
        info = {
            'distance_to_target': distance,
            'step': self.step_count,
            'reached_target': self.reached_target,
            'control_input': control_input
        }
        
        return obs, reward, done, info
    
    def _get_obs(self):
        """獲取當前觀測值（狀態）"""
        state = self.plant.get_state().astype(np.float32)
        position = state[:3]
        distance = np.linalg.norm(position - self.target_position)
        normalized_distance = distance / self.initial_distance

        self.controller_params = np.array([
            self.controller.lambda_att,
            self.controller.eta_att,
            self.controller.lambda_pos,
            self.controller.eta_pos
        ], dtype=np.float32)
        return np.concatenate([state, self.controller_params, np.array([normalized_distance])])
    
    def _compute_reward(self, state, action):
        """計算當前狀態和動作的獎勵值"""
        # 提取位置和角度
        position = state[:3]
        velocity = state[3:6]
        angles = state[6:9]
        ang_vel = state[9:12]
        
        # 到目標位置的距離
        distance = np.linalg.norm(position - self.target_position)
        prev_distance = getattr(self, 'prev_distance', distance)
        distance_improvement = prev_distance - distance
        self.prev_distance = distance
        
        # 基於距離的基本獎勵
        normalized_distance = distance / np.linalg.norm(self.initial_position - self.target_position)
        reward = -0.1*normalized_distance**2  # 離目標越遠，負獎勵越大
        
        # 接近目標的額外獎勵
        # prev_normalized_distance = prev_distance / np.linalg.norm(self.initial_position - self.target_position)
        # improvement = prev_normalized_distance - normalized_distance
        # reward += improvement * 0.5

        # if distance < 0.2:
        #     reward += 0.1  # 正獎勵
        
        # 姿態角偏差懲罰
        # orientation_penalty = np.tanh(np.sum(np.square(angles[:2]))) * 0.05
        # reward -= orientation_penalty 

        # # 速度懲罰（希望平穩運動）
        velocity_penalty = np.tanh(np.sum(np.square(velocity))) * 0.01
        reward -= velocity_penalty
        
        # # 角速度懲罰（希望穩定姿態）
        # ang_vel_penalty = np.tanh(np.sum(np.square(ang_vel))) * 0.005
        # reward -= ang_vel_penalty

        # 控制參數變化太大的懲罰（鼓勵平穩變化）
        # param_change_penalty = 0.0
        # if hasattr(self, 'prev_action'):
        #     param_change_penalty = np.tanh(np.sum(np.square(action - self.prev_action))) * 0.05
        #     reward -= param_change_penalty
        # self.prev_action = action.copy()
        
        # 扣除懲罰
        # reward -= (orientation_penalty + velocity_penalty + ang_vel_penalty)*0.1
        
        # 達到目標的大獎勵
        if distance < 0.2 and not self.reached_target:
            reward += 3.0
            print("\t\treach!")         
            self.reached_target = True
        elif distance < 0.2 and self.reached_target:
            self.step_sucess += 1
            reward += 0.1

        return np.clip(reward, -2.0, 2.0)
    
    def _is_done(self, state):
        """檢查回合是否結束"""
        # 提取位置和角度
        position = state[:3]
        angles = state[6:9]
        
        # 到目標的距離
        distance = np.linalg.norm(position - self.target_position)
        
        # 檢查是否達到最大步數
        if self.step_count >= self.max_steps:
            print("\t \033[34m dead :(\033[0m")
            return True
        
        
        # 成功條件：達到目標並在那裡停留一段時間
        if distance < 0.2 and self.reached_target:
            self.step_sucess += 1
            if self.step_sucess > 30:
                print("\033[33m 成功!!! \033[0m")
                return True
        
        return False