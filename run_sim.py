"""模擬運行模組，處理模擬的主循環邏輯"""
import time
import numpy as np
from gym_pybullet_drones.utils.utils import sync

def run_simulation(env, plant, controller, logger, initial_pos, initial_rpy, target_pos, duration_sec, gui):
    """運行無人機模擬
    
    參數:
        env: 模擬環境
        plant: 無人機植物模型
        controller: 滑模控制器
        logger: 日誌記錄器
        initial_pos: 初始位置
        initial_rpy: 初始姿態
        target_pos: 目標位置
        duration_sec: 模擬時長
        gui: 是否啟用GUI
    """
    # 設置目標位置和姿態
    controller.set_target_position(target_pos)
    controller.set_target_attitude([0, 0, 0])  # 保持水平航向
    
    # 初始化動作數組
    action = np.zeros((1, 4))
    
    # 記錄開始時間
    START = time.time()
    
    # 打印初始位置和目標位置
    print(f"初始位置: ({initial_pos[0]}, {initial_pos[1]}, {initial_pos[2]})")
    print(f"目標位置: ({target_pos[0]}, {target_pos[1]}, {target_pos[2]})")
    
    # 重置控制器狀態
    controller.reset()
    
    # 主模擬循環
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):
        # 步進模擬 - 調用PyBullet進行物理模擬
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 更新植物模型狀態
        plant.update_state(obs[0])
        
        # 計算控制輸入 - 使用滑模控制器
        dt = 1.0/env.CTRL_FREQ
        control_output = controller.update(dt)
        
        # 將控制器輸出轉換為PyBullet期望的格式
        action[0, :] = control_output
        
        # 每50步顯示當前位置
        if i % 50 == 0:
            current_pos = obs[0][0:3]
            print(f"當前位置: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f})")
            current_rpy = obs[0][6:9]
            print(f"當前姿態: ({np.degrees(current_rpy[0]):.2f}°, {np.degrees(current_rpy[1]):.2f}°, {np.degrees(current_rpy[2]):.2f}°)")
        
        # 記錄模擬數據
        logger.log(
            drone=0,
            timestamp=i/env.CTRL_FREQ,
            state=obs[0],
            control=np.hstack([target_pos, initial_rpy[0], np.zeros(6)])
        )
        
        # 渲染環境 - 使用PyBullet的渲染功能
        env.render()
        
        # 同步模擬 - 確保模擬速度與實時時間同步
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)
    
    # 關閉環境 - 釋放PyBullet資源
    env.close()
    
    # 保存模擬結果
    logger.save()
    logger.save_as_csv("single_drone_smc")
    
    return logger