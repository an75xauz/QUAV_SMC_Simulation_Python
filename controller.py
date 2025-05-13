"""控制器模組，處理無人機的控制邏輯"""
import numpy as np
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

def create_controller(drone_model):
    """創建並返回無人機控制器
    
    參數:
        drone_model: 無人機型號
        
    返回:
        控制器實例
    """
    return DSLPIDControl(drone_model=drone_model)

def compute_control_input(controller, control_timestep, state, target_pos, target_rpy):
    """根據目前狀態計算控制輸入
    
    參數:
        controller: 控制器實例
        control_timestep: 控制時間步長
        state: 當前狀態
        target_pos: 目標位置
        target_rpy: 目標姿態
        
    返回:
        控制輸入
    """
    action, _, _ = controller.computeControlFromState(
        control_timestep=control_timestep,
        state=state,
        target_pos=target_pos,
        target_rpy=target_rpy
    )
    return action