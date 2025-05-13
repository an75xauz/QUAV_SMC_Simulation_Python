"""模擬運行模組，處理模擬的主循環邏輯"""
import time
import numpy as np
from gym_pybullet_drones.utils.utils import sync
from controller import compute_control_input

def run_simulation(env, ctrl, logger, initial_pos, initial_rpy, target_pos, duration_sec, gui):
    """運行無人機模擬"""
    # 初始化動作數組
    action = np.zeros((1, 4))
    
    # 記錄開始時間
    START = time.time()
    
    # 打印初始位置和目標位置
    print(f"初始位置: ({initial_pos[0]}, {initial_pos[1]}, {initial_pos[2]})")
    print(f"目標位置: ({target_pos[0]}, {target_pos[1]}, {target_pos[2]})")
    
    # 主模擬循環
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):
        # 步進模擬
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 計算控制輸入
        action[0, :] = compute_control_input(
            controller=ctrl,
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[0],
            target_pos=target_pos,
            target_rpy=initial_rpy[0]
        )
        
        # 每50步顯示當前位置
        if i % 50 == 0:
            current_pos = obs[0][0:3]
            print(f"當前位置: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f})")
        
        # 記錄模擬數據
        logger.log(
            drone=0,
            timestamp=i/env.CTRL_FREQ,
            state=obs[0],
            control=np.hstack([target_pos, initial_rpy[0], np.zeros(6)])
        )
        
        # 渲染環境
        env.render()
        
        # 同步模擬
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)
    
    # 關閉環境
    env.close()
    
    # 保存模擬結果
    logger.save()
    logger.save_as_csv("single_drone")
    
    return logger