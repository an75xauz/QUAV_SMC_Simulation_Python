import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from register_env import make_env
from agent import TD3Agent
from config import HIDDEN_SIZES, GAMMA, TAU, POLICY_NOISE, NOISE_CLIP, POLICY_DELAY, LR
from config import BUFFER_SIZE, BATCH_SIZE, MAX_EPISODES, MAX_STEPS, EXPL_NOISE, SAVE_INTERVAL

def train(
    initial_position=[0, 0, 0],
    target_position=[1, 1, 2],
    seed=0,
    eval_freq=10,
    max_episodes=MAX_EPISODES,
    save_dir="checkpoints"
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
    state_dim = env.observation_space.shape[0]  # 12維狀態
    action_dim = env.action_space.shape[0]      # 6維動作 (SMC參數)
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
    
    # 設置默認SMC參數，用於初始探索
    default_params = np.array([
        2.8,   # lambda_alt
        20.0,  # eta_alt
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
        
        # 單一回合循環
        for step in range(MAX_STEPS):
            # 選擇帶有探索噪聲的動作
            if episode < 10:  # 前幾回合使用默認參數，但有探索
                action = default_params * (1.0 + EXPL_NOISE * (np.random.rand(action_dim) - 0.5))
            else:
                action = agent.select_action(np.array(state))
                action = action + np.random.normal(0, EXPL_NOISE, size=action_dim)
                
            # 將動作限制在合理範圍內
            action = np.clip(action, env.action_space.low, env.action_space.high)
                
            # 執行動作
            next_state, reward, done, info = env.step(action)
            
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
        print(f"回合 {episode}/{max_episodes}: 獎勵 = {episode_reward:.2f}, 平均獎勵 = {avg_reward:.2f}, 步數 = {episode_steps}")
        
        # 定期評估智能體性能
        if episode % eval_freq == 0:
            eval_reward = evaluate_agent(env, agent)
            eval_rewards.append(eval_reward)
            print(f"評估獎勵: {eval_reward:.2f}")
            
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
    
    # 保存最終模型
    torch.save(agent.actor.state_dict(), f"{save_dir}/actor_final.pth")
    torch.save(agent.critic.state_dict(), f"{save_dir}/critic_final.pth")
    
    # 繪製訓練曲線
    plot_training_results(episode_rewards, avg_rewards, eval_rewards, eval_freq, save_dir)
    
    # 關閉環境
    env.close()

def evaluate_agent(env, agent, n_episodes=5):
    """評估智能體在環境中的性能"""
    rewards = []
    
    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 使用確定性策略（無探索噪聲）
            action = agent.select_action(np.array(state), noise=0.0)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        
        rewards.append(episode_reward)
    
    return np.mean(rewards)

def plot_training_results(episode_rewards, avg_rewards, eval_rewards, eval_freq, save_dir):
    """繪製訓練結果"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 繪製每個回合的獎勵
    episodes = list(range(1, len(episode_rewards) + 1))
    ax.plot(episodes, episode_rewards, 'b-', alpha=0.3, label='回合獎勵')
    
    # 繪製移動平均獎勵
    ax.plot(episodes, avg_rewards, 'r-', label='平均獎勵(10回合)')
    
    # 繪製評估獎勵
    eval_episodes = list(range(eval_freq, len(episode_rewards) + 1, eval_freq))
    if len(eval_episodes) > len(eval_rewards):
        eval_episodes = eval_episodes[:len(eval_rewards)]
    ax.plot(eval_episodes, eval_rewards, 'g-', label='評估獎勵')
    
    ax.set_xlabel('回合')
    ax.set_ylabel('獎勵')
    ax.set_title('TD3訓練曲線：SMC控制器參數優化')
    ax.legend()
    ax.grid(True)
    
    plt.savefig(f"{save_dir}/training_curve.png")
    plt.close()

if __name__ == "__main__":
    # 命令行參數解析
    parser = argparse.ArgumentParser(description='TD3四旋翼機訓練程式')
    parser.add_argument('--initial', type=float, nargs=3, default=[0, 0, 0], help='初始位置[x, y, z]')
    parser.add_argument('--target', type=float, nargs=3, default=[1, 1, 2], help='目標位置[x, y, z]')
    parser.add_argument('--seed', type=int, default=0, help='隨機種子')
    parser.add_argument('--eval_freq', type=int, default=10, help='評估頻率')
    parser.add_argument('--episodes', type=int, default=MAX_EPISODES, help='訓練回合數')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='模型保存目錄')
    
    args = parser.parse_args()
    
    # 運行訓練
    train(
        initial_position=args.initial,
        target_position=args.target,
        seed=args.seed,
        eval_freq=args.eval_freq,
        max_episodes=args.episodes,
        save_dir=args.save_dir
    )