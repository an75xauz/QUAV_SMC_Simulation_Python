import argparse
import numpy as np
import torch

from simulation.plant import QuadrotorPlant
from simulation.controller import QuadrotorSMCController
from simulation.sim import QuadrotorSimulator
from rl.agent import TD3Agent, Actor

def load_model(actor_path, device, state_dim=17, action_dim=4, hidden_sizes=[256, 256]):
    """
    載入訓練好的 Actor 模型
    
    Args:
        actor_path: 模型檔案路徑
        device: 計算裝置 (CPU/GPU)
        state_dim: 狀態維度
        action_dim: 動作維度 (SMC 參數數量)
        hidden_sizes: 隱藏層大小
    
    Returns:
        已載入權重的 Actor 模型
    """
    # 定義動作空間上下限 (與 env_UAV.py 中相同)
    min_action = np.array([1.0, 1.0, 0.1, 0.1])  # 最小值
    max_action = np.array([40.0, 40.0, 1, 1])  # 最大值
    
    max_action_tensor = torch.tensor(max_action, dtype=torch.float32).to(device)
    
    # 建立模型
    actor = Actor(state_dim, action_dim, max_action_tensor, hidden_sizes).to(device)
    
    # 載入模型權重
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    
    # 設定為評估模式
    actor.eval()
    
    return actor, min_action, max_action


def main():
    """主函數，用於測試強化學習模型控制四旋翼機"""
    # 解析命令列參數
    parser = argparse.ArgumentParser(description='四旋翼機強化學習模型測試程式')
    parser.add_argument(
        '--initial', type=float, nargs=3, default=[0, 0, 0],
        help='初始位置 [x y z] (預設: [0 0 0])'
    )
    parser.add_argument(
        '--target', type=float, nargs=3, default=[1, 1, 2],
        help='目標位置 [x y z] (預設: [1 1 2])'
    )
    parser.add_argument(
        '--time', type=float, default=10,
        help='模擬時間長度 (秒) (預設: 10.0)'
    )
    parser.add_argument(
        '--dt', type=float, default=0.05,
        help='時間步長 (秒) (預設: 0.05)'
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='只產生靜態圖表 (不顯示動畫)'
    )
    parser.add_argument(
        '--initial_attitude', type=float, nargs=3, default=[0, 0, 0],
        help='初始姿態角 [roll pitch yaw] (預設: [0 0.2 0.1]rad)'
    )
    parser.add_argument(
        '--model_dir', type=str, default='checkpoints',
        help='模型存放目錄 (預設: logs)'
    )
    parser.add_argument(
        '--actor_file', type=str, default='best_actor.pth',
        help='Actor 模型檔案名稱 (預設: best_actor.pth)'
    )
    
    args = parser.parse_args()
    
    # 設定計算裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")
    
    # 載入模型
    model_path = f"{args.model_dir}/{args.actor_file}"
    actor_model, min_action, max_action = load_model(model_path, device)
    print(f"已載入模型: {model_path}")
    
    # 建立四旋翼機物理模型和控制器
    plant = QuadrotorPlant()
    controller = QuadrotorSMCController(plant)
    
    # 建立並設定模擬器
    simulator = QuadrotorSimulator(
        plant=plant,
        controller=controller,
        initial_position=args.initial,
        target_position=args.target,
        initial_attitude=args.initial_attitude
    )
    simulator.dt = args.dt
    simulator.max_time = args.time
    
    # 顯示模擬參數
    print(f"執行模擬:")
    print(f"  初始位置: {args.initial}")
    print(f"  目標位置: {args.target}")
    print(f"  初始姿態角: {args.initial_attitude}")
    print(f"  模擬時間: {args.time} 秒")
    print(f"  時間步長: {args.dt} 秒")
    
    # 初始化控制器參數
    smc_params = np.array([30.0, 9.0, 0.5, 0.5])  # 初始控制器參數
    controller.lambda_att = smc_params[0]
    controller.eta_att = smc_params[1]
    controller.lambda_pos = smc_params[2]
    controller.eta_pos = smc_params[3]
    
    # RL控制模擬主循環
    done = False
    
    while not done:
         # 獲取當前狀態
        state = plant.get_state()
        
        # 計算歸一化距離
        position = state[:3]
        distance = np.linalg.norm(position - np.array(args.target))
        initial_distance = np.linalg.norm(np.array(args.initial) - np.array(args.target))
        normalized_distance = distance / initial_distance
        
        # 將狀態擴充為17維（原始12維狀態 + 4維控制器參數 + 1維歸一化距離）
        augmented_state = np.concatenate([
            state, 
            np.array([
                controller.lambda_att,
                controller.eta_att,
                controller.lambda_pos,
                controller.eta_pos
            ]),
            np.array([normalized_distance])
        ])
        
        # 將狀態轉換為張量
        state_tensor = torch.tensor(augmented_state.reshape(1, -1), dtype=torch.float32).to(device)
        
        # 使用RL模型預測SMC控制器參數
        with torch.no_grad():
            smc_params = actor_model(state_tensor).cpu().numpy().flatten()
        
        # 限制參數範圍
        smc_params = np.clip(smc_params, min_action, max_action)
        
        # 將參數應用到控制器
        controller.lambda_att = smc_params[0]
        controller.eta_att = smc_params[1]
        controller.lambda_pos = smc_params[2]
        controller.eta_pos = smc_params[3]
        
        # 模擬一步
        done = simulator.step()

    
    # 生成視覺化
    if args.plot:
        simulator.plot_results()
    else:
        simulator.animate()


if __name__ == "__main__":
    main()