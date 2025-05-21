import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_training_metrics(agent, episode_rewards, avg_rewards, eval_rewards, eval_freq, save_dir):
    """Plot comprehensive training metrics"""
    # Create canvas with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('TD3 Training Metrics', fontsize=16)
    
    # 1. Rewards plot (top-left)
    episodes = list(range(1, len(episode_rewards) + 1))
    axs[0, 0].plot(episodes, episode_rewards, 'b-', alpha=0.3, label='Episode Rewards')
    axs[0, 0].plot(episodes, avg_rewards, 'r-', label='Avg Rewards (10 episodes)')
    
    # Evaluation rewards
    eval_episodes = list(range(eval_freq, len(episode_rewards) + 1, eval_freq))
    if len(eval_episodes) > len(eval_rewards):
        eval_episodes = eval_episodes[:len(eval_rewards)]
    axs[0, 0].plot(eval_episodes, eval_rewards, 'g-', label='Evaluation Rewards')
    
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Reward Value')
    axs[0, 0].set_title('Reward Curves')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # 2. Loss functions plot (top-right)
    if hasattr(agent, 'critic_losses') and len(agent.critic_losses) > 0:
        iterations = list(range(1, len(agent.critic_losses) + 1))
        axs[0, 1].plot(iterations, agent.critic_losses, 'b-', alpha=0.7, label='Critic Loss')
        
        # Actor losses (may be sparse due to delayed updates)
        if hasattr(agent, 'actor_losses') and len(agent.actor_losses) > 0:
            # 直接使用索引作為 x 軸而不是嘗試計算實際的迭代次數
            actor_x = list(range(1, len(agent.actor_losses) + 1))
            axs[0, 1].plot(actor_x, agent.actor_losses, 'r-', label='Actor Loss')
        
        axs[0, 1].set_xlabel('Training Iterations')
        axs[0, 1].set_ylabel('Loss Value')
        axs[0, 1].set_title('TD3 Loss Functions')
        axs[0, 1].legend()
        axs[0, 1].grid(True)
    else:
        axs[0, 1].text(0.5, 0.5, 'Loss data not yet collected', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=axs[0, 1].transAxes)
    
    # 3. Q-values plot (bottom-left)
    if hasattr(agent, 'q_values') and len(agent.q_values) > 0:
        iterations = list(range(1, len(agent.q_values) + 1))
        axs[1, 0].plot(iterations, agent.q_values, 'g-', label='Average Q-value')
        
        # Calculate moving average
        window = min(100, len(agent.q_values))
        if window > 0:
            q_moving_avg = np.convolve(agent.q_values, np.ones(window)/window, mode='valid')
            axs[1, 0].plot(list(range(window, len(agent.q_values) + 1)), q_moving_avg, 'r-', 
                          label=f'Moving Avg (window={window})')
        
        axs[1, 0].set_xlabel('Training Iterations')
        axs[1, 0].set_ylabel('Q-value')
        axs[1, 0].set_title('Critic Q-value Changes')
        axs[1, 0].legend()
        axs[1, 0].grid(True)
    else:
        axs[1, 0].text(0.5, 0.5, 'Q-value data not yet collected', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=axs[1, 0].transAxes)
    
    # 4. Action parameter values (bottom-right)
    if hasattr(agent, 'clipped_action_values') and len(agent.clipped_action_values) > 0:
        # Convert action values to numpy array
        action_array = np.array(agent.clipped_action_values)
        
        # Sample to reduce plot points if too many
        sample_rate = max(1, len(action_array) // 1000)
        sampled_indices = range(0, len(action_array), sample_rate)
        sampled_actions = action_array[sampled_indices]
        
        # Get action dimensions
        action_dim = sampled_actions.shape[1]
        
        # Plot a line for each SMC parameter
        param_names = ['lambda_att', 'eta_att', 'lambda_pos', 'eta_pos']
        for i in range(action_dim):
            param_name = param_names[i] if i < len(param_names) else f'Param {i+1}'
            axs[1, 1].plot(sampled_indices, sampled_actions[:, i], label=param_name)
        
        axs[1, 1].set_xlabel('Steps')
        axs[1, 1].set_ylabel('Parameter Value')
        axs[1, 1].set_title('SMC Controller Parameters (Clipped)')
        axs[1, 1].legend()
        axs[1, 1].grid(True)
    else:
    # Fall back to raw action values if clipped ones aren't available
        axs[1, 1].text(0.5, 0.5, 'Clipped action parameter data not available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=axs[1, 1].transAxes)
    
    # Adjust subplots spacing
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{save_dir}/training_metrics.png", dpi=200)
    plt.close(fig)

def plot_training_results(episode_rewards, avg_rewards, eval_rewards, eval_freq, save_dir):
    """繪製訓練結果"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 繪製每個回合的獎勵
    episodes = list(range(1, len(episode_rewards) + 1))
    ax.plot(episodes, episode_rewards, 'b-', alpha=0.3, label='Episodes Rewards')
    
    # 繪製移動平均獎勵
    ax.plot(episodes, avg_rewards, 'r-', label='avg rewards(10 episodes)')
    
    # 繪製評估獎勵
    eval_episodes = list(range(eval_freq, len(episode_rewards) + 1, eval_freq))
    if len(eval_episodes) > len(eval_rewards):
        eval_episodes = eval_episodes[:len(eval_rewards)]
    ax.plot(eval_episodes, eval_rewards, 'g-', label='eval rewards')
    
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    ax.set_title('TD3 training curve')
    ax.legend()
    ax.grid(True)
    
    plt.savefig(f"{save_dir}/training_curve.png")
    plt.close()

def moving_average(data, window_size=20):
    if len(data) < window_size:
        return data
    return [sum(data[i:i+window_size]) / window_size for i in range(len(data) - window_size + 1)]

def plot_loss(agent, save_dir, critic_window=20, actor_window=5):
    """
    繪製 TD3 中 actor 與 critic 的 loss 曲線（含平滑處理）
    
    參數：
    - agent: TD3 的 agent 物件，需包含 `critic_losses`, `actor_losses`, `policy_delay`
    - critic_window: critic loss 的移動平均視窗大小
    - actor_window: actor loss 的移動平均視窗大小
    """
    if hasattr(agent, 'critic_losses') and len(agent.critic_losses) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))  # 新圖

        # Critic Loss（平滑處理）
        smooth_critic = moving_average(agent.critic_losses, window_size=critic_window)
        critic_x = list(range(1, len(smooth_critic) + 1))
        ax.plot(critic_x, smooth_critic, 'b-', alpha=0.8, label='Critic Loss (Smoothed)')

        # Actor Loss（根據 policy delay 對齊 X 軸）
        if hasattr(agent, 'actor_losses') and len(agent.actor_losses) > 0:
            policy_delay = int(agent.policy_delay.item()) if torch.is_tensor(agent.policy_delay) else agent.policy_delay
            smooth_actor = moving_average(agent.actor_losses, window_size=actor_window)
            actor_x = list(range(policy_delay, policy_delay * len(smooth_actor) + 1, policy_delay))
            if len(actor_x) > len(smooth_actor):
                actor_x = actor_x[:len(smooth_actor)]
            ax.plot(actor_x, smooth_actor, 'r-', label='Actor Loss (Smoothed)')

        # 圖表設定
        ax.set_xlabel('Training Iterations')
        ax.set_ylabel('Loss Value')
        ax.set_title('TD3 Loss Functions')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/loss_curve.png")
        plt.close()
    else:
        print("❗ 尚未收集到 loss 資料，無法繪圖")
def plot_q_values(agent, save_dir):
    """Plot Q-values with moving average window
    
    Args:
        agent: TD3Agent object containing stored Q-values
        save_dir: Directory to save the plot
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Check if sufficient Q-value data exists
    if len(agent.q_values) < 10:
        print("Insufficient Q-value data for plotting")
        return
    
    # Convert to numpy array
    q1_values = np.array(agent.q_values)
    
    # Calculate moving average (window=100)
    def moving_average(data, window_size=100):
        if len(data) < window_size:
            return data
        
        weights = np.ones(window_size) / window_size
        return np.convolve(data, weights, mode='valid')
    
    # Apply moving average if data length is sufficient
    x_vals = np.arange(len(q1_values))
    window_size = 100
    
    if len(q1_values) >= 100:
        q1_ma = moving_average(q1_values)
        # Adjust x-axis to match moving average length
        x_vals_ma = np.arange(len(q1_ma)) + window_size - 1
    else:
        q1_ma = q1_values
        x_vals_ma = x_vals
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot Q-values with moving average
    plt.plot(x_vals_ma, q1_ma, color='blue', linewidth=2, label='Q-value (Moving Avg)')
    
    # Optional: Plot original Q-values as scatter points
    plt.scatter(x_vals[::100], q1_values[::100], color='gray', alpha=0.4, s=15, label='Original Q-values (every 100 points)')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Q-value')
    plt.title('TD3 Q-values During Training (Moving Average, Window=100)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(f"{save_dir}/q_values.png", dpi=200, bbox_inches='tight')
    plt.close()

def plot_dual_q_values(agent, save_dir):
    """Plot individual Q-values of dual Q-networks with moving average window
    
    Args:
        agent: TD3Agent object containing stored Q-values
        save_dir: Directory to save the plot
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Check if sufficient Q-value data exists
    if not hasattr(agent, 'q1_values') or not hasattr(agent, 'q2_values'):
        print("Need to add q1_values and q2_values attributes to TD3Agent to record dual Q-network values")
        return
    
    if len(agent.q1_values) < 10 or len(agent.q2_values) < 10:
        print("Insufficient Q-value data for plotting")
        return
    
    # Convert to numpy arrays
    q1_values = np.array(agent.q1_values)
    q2_values = np.array(agent.q2_values)
    
    # Ensure consistent lengths
    min_length = min(len(q1_values), len(q2_values))
    q1_values = q1_values[:min_length]
    q2_values = q2_values[:min_length]
    
    # Calculate moving average (window=100)
    def moving_average(data, window_size=100):
        if len(data) < window_size:
            return data
        
        weights = np.ones(window_size) / window_size
        return np.convolve(data, weights, mode='valid')
    
    window_size = 100
    
    # Apply moving average if data length is sufficient
    if min_length >= window_size:
        q1_ma = moving_average(q1_values)
        q2_ma = moving_average(q2_values)
        # Adjust x-axis to match moving average length
        x_vals = np.arange(len(q1_ma)) + window_size - 1
    else:
        q1_ma = q1_values
        q2_ma = q2_values
        x_vals = np.arange(min_length)
    
    # Create plot
    plt.figure(figsize=(12, 7))
    
    # Plot Q1 and Q2 values with moving average
    plt.plot(x_vals, q1_ma, color='blue', linewidth=2, label='Q1 Network (Moving Avg)')
    plt.plot(x_vals, q2_ma, color='red', linewidth=2, label='Q2 Network (Moving Avg)')
    
    # Calculate and plot difference between Q1 and Q2
    q_diff = np.abs(q1_ma - q2_ma)
    plt.plot(x_vals, q_diff, color='green', linestyle='--', linewidth=1.5, label='|Q1-Q2| Difference')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Q Values')
    plt.title('TD3 Dual Q-Network Values (Moving Average, Window=100)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add secondary y-axis to highlight Q1-Q2 difference
    ax2 = plt.gca().twinx()
    ax2.plot(x_vals, q_diff, color='green', linestyle='--', linewidth=1.5, alpha=0)  # Transparent line, only for secondary y-axis
    ax2.set_ylabel('|Q1-Q2| Difference', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Save plot
    plt.savefig(f"{save_dir}/dual_q_values.png", dpi=200, bbox_inches='tight')
    plt.close()