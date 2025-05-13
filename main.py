"""單一無人機滑模控制模擬程式的主入口。

範例
-------
在終端中運行：

    $ python main.py --initial_x 0 --initial_y 0 --initial_z 0.1 --target_x 1 --target_y 1 --target_z 0.5
"""
import os
import argparse
import numpy as np
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import str2bool

# 導入自定義模組
import config
from controller import QuadrotorSMCController, set_smc_parameters
from env import create_env, create_logger, DroneQuadrotorPlant
from run_sim import run_simulation

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='單一無人機滑模控制模擬程式')
    
    # 基本設置參數
    parser.add_argument('--drone', default=config.DEFAULT_DRONE, type=DroneModel,
                       help='無人機型號 (默認: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--physics', default=config.DEFAULT_PHYSICS, type=Physics,
                       help='物理更新 (默認: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui', default=config.DEFAULT_GUI, type=str2bool,
                       help='是否使用PyBullet GUI (默認: True)', metavar='')
    parser.add_argument('--record_video', default=config.DEFAULT_RECORD_VISION, type=str2bool,
                       help='是否記錄視頻 (默認: False)', metavar='')
    parser.add_argument('--plot', default=config.DEFAULT_PLOT, type=str2bool,
                       help='是否繪製模擬結果 (默認: True)', metavar='')
    parser.add_argument('--user_debug_gui', default=config.DEFAULT_USER_DEBUG_GUI, type=str2bool,
                       help='是否添加調試線和參數到GUI (默認: False)', metavar='')
    parser.add_argument('--obstacles', default=config.DEFAULT_OBSTACLES, type=str2bool,
                       help='是否在環境中添加障礙物 (默認: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=config.DEFAULT_SIMULATION_FREQ_HZ, type=int,
                       help='模擬頻率(Hz) (默認: 240)', metavar='')
    parser.add_argument('--control_freq_hz', default=config.DEFAULT_CONTROL_FREQ_HZ, type=int,
                       help='控制頻率(Hz) (默認: 48)', metavar='')
    parser.add_argument('--duration_sec', default=config.DEFAULT_DURATION_SEC, type=int,
                       help='模擬時長(秒) (默認: 5)', metavar='')
    parser.add_argument('--output_folder', default=config.DEFAULT_OUTPUT_FOLDER, type=str,
                       help='保存日誌的資料夾 (默認: "results")', metavar='')
    parser.add_argument('--colab', default=config.DEFAULT_COLAB, type=bool,
                       help='是否由筆記本運行示例 (默認: "False")', metavar='')
    
    # 位置參數
    parser.add_argument('--initial_x', default=config.DEFAULT_INITIAL_X, type=float,
                       help='無人機初始X座標 (默認: 0.0)', metavar='')
    parser.add_argument('--initial_y', default=config.DEFAULT_INITIAL_Y, type=float,
                       help='無人機初始Y座標 (默認: 0.0)', metavar='')
    parser.add_argument('--initial_z', default=config.DEFAULT_INITIAL_Z, type=float,
                       help='無人機初始Z座標 (默認: 0.1)', metavar='')
    parser.add_argument('--target_x', default=config.DEFAULT_TARGET_X, type=float,
                       help='無人機目標X座標 (默認: 1.0)', metavar='')
    parser.add_argument('--target_y', default=config.DEFAULT_TARGET_Y, type=float,
                       help='無人機目標Y座標 (默認: 1.0)', metavar='')
    parser.add_argument('--target_z', default=config.DEFAULT_TARGET_Z, type=float,
                       help='無人機目標Z座標 (默認: 0.5)', metavar='')
    
    # # 滑模控制器參數
    # parser.add_argument('--lambda_pos', default=0.5, type=float,
    #                    help='位置控制滑動面斜率 (默認: 0.5)', metavar='')
    # parser.add_argument('--eta_pos', default=0.5, type=float,
    #                    help='位置控制增益 (默認: 0.5)', metavar='')
    # parser.add_argument('--lambda_alt', default=2.3, type=float,
    #                    help='高度控制滑動面斜率 (默認: 2.3)', metavar='')
    # parser.add_argument('--eta_alt', default=25.0, type=float,
    #                    help='高度控制增益 (默認: 25.0)', metavar='')
    # parser.add_argument('--lambda_att', default=30, type=float,
    #                    help='姿態控制滑動面斜率 (默認: 30)', metavar='')
    # parser.add_argument('--eta_att', default=25, type=float,
    #                    help='姿態控制增益 (默認: 25)', metavar='')
    # parser.add_argument('--lambda_att_yaw', default=30, type=float,
    #                    help='偏航控制滑動面斜率 (默認: 30)', metavar='')
    # parser.add_argument('--eta_att_yaw', default=9.0, type=float,
    #                    help='偏航控制增益 (默認: 9.0)', metavar='')
    # parser.add_argument('--k_smooth', default=0.5, type=float,
    #                    help='控制平滑因子 (默認: 0.5)', metavar='')
    # parser.add_argument('--k_smooth_pos', default=0.5, type=float,
    #                    help='位置控制平滑因子 (默認: 0.5)', metavar='')
    # parser.add_argument('--max_angle', default=None, type=float,
    #                    help='最大傾角(度), 若未指定則使用默認值', metavar='')
    
    return parser.parse_args()

def main():
    """主程式入口"""
    # 解析命令行參數
    args = parse_args()
    
    # 設置初始位置和目標位置
    initial_pos = np.array([args.initial_x, args.initial_y, args.initial_z])
    target_pos = np.array([args.target_x, args.target_y, args.target_z])
    
    # 設置初始姿態
    initial_xyzs = np.array([initial_pos])
    initial_rpys = np.array([[0, 0, 0]])
    
    # 創建環境
    env = create_env(
        drone_model=args.drone,
        num_drones=1,
        initial_xyzs=initial_xyzs,
        initial_rpys=initial_rpys,
        physics=args.physics,
        simulation_freq_hz=args.simulation_freq_hz,
        control_freq_hz=args.control_freq_hz,
        gui=args.gui,
        record_video=args.record_video,
        obstacles=args.obstacles,
        user_debug_gui=args.user_debug_gui
    )
    
    # 創建模型
    plant = DroneQuadrotorPlant(args.drone)
    
    # 創建滑模控制器
    controller = QuadrotorSMCController(plant)
    

    # 創建日誌記錄器
    logger = create_logger(
        control_freq_hz=args.control_freq_hz,
        num_drones=1,
        output_folder=args.output_folder,
        colab=args.colab
    )
    
    # 運行模擬
    logger = run_simulation(
        env=env,
        plant=plant,
        controller=controller,
        logger=logger,
        initial_pos=initial_pos,
        initial_rpy=initial_rpys,
        target_pos=target_pos,
        duration_sec=args.duration_sec,
        gui=args.gui
    )
    
    # 繪製模擬結果
    if args.plot:
        logger.plot()

if __name__ == "__main__":
    main()