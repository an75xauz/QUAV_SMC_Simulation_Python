import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 使用非互動式後端，不會顯示視窗
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from utils.log_utils import TrainingLogger
import atexit

from plot_utils import plot_training_metrics, plot_training_results, plot_loss, plot_q_values, plot_dual_q_values
from rl.register_env import make_env
from rl.agent import TD3Agent
from rl.env_UAV import QuadrotorEnv
from rl.config import *

def train(
    initial_position=[0, 0, 0],
    target_position=[1, 1, 2],
    seed=0,
    eval_freq=ECAL_FREQ, #每()eqisode檢查性能
    max_episodes=MAX_EPISODES,
    save_dir="checkpoints",
    save_log_freq=50
):
    """TD3強化學習訓練入口函數"""
    # 設置隨機種子
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 創建保存目錄
    os.makedirs(save_dir, exist_ok=True)
    
    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    
    # 創建環境
    env = make_env(initial_position=initial_position, target_position=target_position)
    
    # 提取環境資訊
    state_dim = env.observation_space.shape[0]  #17
    action_dim = env.action_space.shape[0]      # 4
    max_action = env.action_space.high          # 各參數的最大值
    
    # 初始化智能體
    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        hidden_sizes=HIDDEN_SIZES,
        gamma=GAMMA,
        tau=TAU,
        policy_noise=POLICY_NOISE * np.ones(action_dim),  # 根據動作維度調整噪聲
        noise_clip=NOISE_CLIP * np.ones(action_dim),      # 根據動作維度調整噪聲限制
        policy_delay=POLICY_DELAY,
        lr=LR,
        buffer_size=BUFFER_SIZE
    )
    
    logger = TrainingLogger(save_dir=save_dir, save_frequency=save_log_freq)
    atexit.register(logger.force_save)
    # 設置默認SMC參數，用於初始探索
    default_params = np.array([
        # 2.8,   # lambda_alt
        # 20.0,  # eta_alt
        30.0,  # lambda_att
        9.0,   # eta_att
        0.5,   # lambda_pos
        0.5    # eta_pos
    ])
    
    # 用於記錄訓練進度
    episode_rewards = []
    eval_rewards = []
    avg_rewards = []
    best_reward = -np.inf
    
    # 主訓練循環
    for episode in tqdm(range(1, max_episodes + 1), desc="訓練進度"):

        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        # 記錄當前回合的控制參數
        episode_actions = []

        # 單一回合循環
        for step in range(MAX_STEPS):
            # 選擇帶有探索噪聲的動作
            if episode < 20:  # 前幾回合使用默認參數，但有探索
                action = default_params * (1.0 + EXPL_NOISE * (np.random.rand(action_dim) - 0.5))
            else:
                action = agent.select_action(np.array(state))
                action = action + np.random.normal(0, EXPL_NOISE, size=action_dim)
                
            # 將動作限制在合理範圍內
            action = np.clip(action, env.action_space.low, env.action_space.high)
                
            # 執行動作
            next_state, reward, done, info = env.step(action)

            # 記錄本回合使用的控制參數
            episode_actions.append(action.copy())

            # 將轉換儲存到回放緩衝區
            agent.replay_buffer.add(state, action, next_state, reward, done)
            
            # 更新狀態和總獎勵
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            # 訓練智能體
            if agent.replay_buffer.size > BATCH_SIZE:
                agent.train(BATCH_SIZE)
            
            if done:
                break
        
        # 記錄回合獎勵
        episode_rewards.append(episode_reward)
        
        # 計算平均獎勵 (最近10個回合)
        avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
        avg_rewards.append(avg_reward)
        
        # 顯示訓練進度
        # print(f"回合 {episode}/{max_episodes}: 獎勵 = {episode_reward:.2f}, 平均獎勵 = {avg_reward:.2f}, 步數 = {episode_steps}")
        print(f"\033[1;33m回合 {episode}/{max_episodes}: 獎勵 = {episode_reward:.2f}, 平均獎勵 = {avg_reward:.2f}, 步數 = {episode_steps}\033[0m")
        print("***----------***")

        # Log episode data
        logger.log_episode(episode, episode_reward, avg_reward, episode_steps, info)
        
        # Log last control parameters
        if episode_actions:
            logger.log_action(episode, episode_steps, episode_actions[-1])
        
        # Log agent's Q values and losses
        logger.log_agent_metrics(agent.total_it, agent)

        # 定期評估智能體性能
        if episode % eval_freq == 0 :
            eval_reward = evaluate_agent(env, agent, logger)
            eval_rewards.append(eval_reward)
            print(f"評估獎勵: {eval_reward:.2f}")
            logger.log_evaluation(episode, eval_reward)
            plot_training_metrics(agent, episode_rewards, avg_rewards, eval_rewards, eval_freq, save_dir)

            # 保存最佳模型
            if eval_reward > best_reward:
                best_reward = eval_reward
                torch.save(agent.actor.state_dict(), f"{save_dir}/best_actor.pth")
                torch.save(agent.critic.state_dict(), f"{save_dir}/best_critic.pth")
                print(f"保存最佳模型，獎勵: {best_reward:.2f}")
        
        # 定期保存模型檢查點
        if episode % SAVE_INTERVAL == 0:
            torch.save(agent.actor.state_dict(), f"{save_dir}/actor_{episode}.pth")
            torch.save(agent.critic.state_dict(), f"{save_dir}/critic_{episode}.pth")
            
            # 保存訓練曲線
            plot_training_results(episode_rewards, avg_rewards, eval_rewards, eval_freq, save_dir)
            plot_loss(agent, save_dir)
            plot_q_values(agent, save_dir)
            plot_dual_q_values(agent, save_dir)
    
    # 保存最終模型
    torch.save(agent.actor.state_dict(), f"{save_dir}/actor_final.pth")
    torch.save(agent.critic.state_dict(), f"{save_dir}/critic_final.pth")
    
    # 繪製訓練曲線
    plot_training_results(episode_rewards, avg_rewards, eval_rewards, eval_freq, save_dir)
    plot_loss(agent, save_dir)
    plot_q_values(agent, save_dir)
    plot_dual_q_values(agent, save_dir)
    # 關閉環境
    print("訓練完成。儲存最終日誌...")
    logger.save_to_csv()
    env.close()


def evaluate_agent(env, agent, logger=None, n_episodes=5):
    """評估智能體在環境中的性能"""
    rewards = []
    
    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        print(f"eval_episiode:{_}")
        while not done and step_count<MAX_STEPS:
            # 使用確定性策略（無探索噪聲）
            action = agent.select_action(np.array(state), noise=0.0)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            step_count +=1
            if logger:
                position = next_state[:3] if len(next_state) >= 3 else next_state
                target_position = env.target_position if hasattr(env, 'target_position') else np.zeros(3)
                logger.log_trajectory(_, step_count, position, target_position)
        rewards.append(episode_reward)
    
    return np.mean(rewards)


if __name__ == "__main__":
    # 命令行參數解析
    parser = argparse.ArgumentParser(description='TD3 Quadrotor Training')
    parser.add_argument('--initial', type=float, nargs=3, default=[0, 0, 0], help='Initial position [x, y, z]')
    parser.add_argument('--target', type=float, nargs=3, default=[1, 1, 2], help='Target position [x, y, z]')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--eval_freq', type=int, default=ECAL_FREQ, help='Evaluation frequency')
    parser.add_argument('--episodes', type=int, default=MAX_EPISODES, help='Number of training episodes')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save models and logs')
    parser.add_argument('--save_log_freq', type=int, default=50, help='Frequency to save log files (in episodes)')
    
    args = parser.parse_args()
    
    # 運行訓練
    train(
        initial_position=args.initial,
        target_position=args.target,
        seed=args.seed,
        eval_freq=args.eval_freq,
        max_episodes=args.episodes,
        save_dir=args.save_dir,
        save_log_freq=args.save_log_freq
    )