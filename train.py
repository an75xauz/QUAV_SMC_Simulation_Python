import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend, won't display windows
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from utils.log_utils import TrainingLogger
import atexit

from utils.plot_utils import plot_training_metrics, plot_training_results, plot_loss, plot_q_values, plot_dual_q_values
from rl.register_env import make_env
from rl.agent import TD3Agent
from rl.env_UAV import QuadrotorEnv
from rl.config import *
from rl.model_loader import check_and_load_model  

def train(
    initial_position=[0, 0, 0],
    target_position=[1, 1, 2],
    seed=0,
    eval_freq=ECAL_FREQ, # Evaluate performance every () episodes
    max_episodes=MAX_EPISODES,
    save_dir="checkpoints",
    save_log_freq=50
):
    """TD3 reinforcement learning training entry function"""
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = make_env(initial_position=initial_position, target_position=target_position)
    
    # Extract environment information
    state_dim = env.observation_space.shape[0]  # 17
    action_dim = env.action_space.shape[0]      # 4
    max_action = env.action_space.high          # Maximum values for each parameter
    
    # Initialize agent
    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        hidden_sizes=HIDDEN_SIZES,
        gamma=GAMMA,
        tau=TAU,
        policy_noise=POLICY_NOISE * np.ones(action_dim),  # Adjust noise based on action dimension
        noise_clip=NOISE_CLIP * np.ones(action_dim),      # Adjust noise clip based on action dimension
        policy_delay=POLICY_DELAY,
        lr=LR,
        buffer_size=BUFFER_SIZE
    )
    
    # Check if there are available models and try to load
    model_loaded, start_episode, metrics = check_and_load_model(
        agent=agent,
        save_dir=save_dir,
        device=device
    )
    
    # Initialize tracking metrics
    episode_rewards = metrics["episode_rewards"] if model_loaded else []
    avg_rewards = metrics["avg_rewards"] if model_loaded else []
    eval_rewards = metrics["eval_rewards"] if model_loaded else []
    best_reward = max(eval_rewards) if eval_rewards else -np.inf
    
    # Set training start and end episodes
    start_episode = start_episode if model_loaded else 1
    end_episode = start_episode + max_episodes - 1 if model_loaded else max_episodes
    
    # Setup training logger
    logger = TrainingLogger(save_dir=save_dir, save_frequency=save_log_freq)
    atexit.register(logger.force_save)
    
    # Set default SMC parameters for initial exploration
    default_params = np.array([
        30.0,  # lambda_att
        9.0,   # eta_att
        0.5,   # lambda_pos
        0.5    # eta_pos
    ])
    
    clipped_action_values = []
    # Main training loop
    for episode in tqdm(range(start_episode, end_episode + 1), desc="Training Progress"):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        # Record control parameters for current episode
        episode_actions = []

        # Single episode loop
        for step in range(MAX_STEPS):
            # Select action with exploration noise
            if episode < start_episode + 20 and not model_loaded:  # Only use default params for new training
                action = default_params * (1.0 + EXPL_NOISE * (np.random.rand(action_dim) - 0.5))
                # Clip action to valid range
                action = np.clip(action, env.action_space.low, env.action_space.high)
                
                # Record clipped action for plotting when using default params
                if hasattr(agent, 'clipped_action_values'):
                    agent.clipped_action_values.append(action.copy())
            else:
                # The select_action method now handles recording of both raw and clipped actions
                action = agent.select_action(np.array(state))
                action = action + np.random.normal(0, EXPL_NOISE, size=action_dim)
                
                # Clip action to valid range
                action = np.clip(action, env.action_space.low, env.action_space.high)
                
                # Update the clipped action record if it exists
                if hasattr(agent, 'clipped_action_values'):
                    agent.clipped_action_values.append(action.copy())
                
            # Execute action
            next_state, reward, done, info = env.step(action)

            # Record control parameters used this episode
            episode_actions.append(action.copy())

            # Store transition in replay buffer
            agent.replay_buffer.add(state, action, next_state, reward, done)
            
            # Update state and total reward
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            # Train agent
            if agent.replay_buffer.size > BATCH_SIZE:
                agent.train(BATCH_SIZE)
            
            if done:
                break
        
        # Record episode reward
        episode_rewards.append(episode_reward)
        
        # Calculate average reward (last 10 episodes)
        avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
        avg_rewards.append(avg_reward)
        
        # Display training progress
        print(f"\033[1;33mEpisode {episode}/{end_episode}: Reward = {episode_reward:.2f}, Avg Reward = {avg_reward:.2f}, Steps = {episode_steps}\033[0m")
        print("***----------***")

        # Log episode data
        logger.log_episode(episode, episode_reward, avg_reward, episode_steps, info)
        
        # Log last control parameters
        if episode_actions:
            logger.log_action(episode, episode_steps, episode_actions[-1])
        
        # Log agent's Q values and losses
        logger.log_agent_metrics(agent.total_it, agent)

        # Periodically evaluate agent performance
        if episode % eval_freq == 0:
            eval_reward = evaluate_agent(env, agent, logger)
            eval_rewards.append(eval_reward)
            print(f"Evaluation reward: {eval_reward:.2f}")
            logger.log_evaluation(episode, eval_reward)
            plot_training_metrics(agent, episode_rewards, avg_rewards, eval_rewards, eval_freq, save_dir)

            # Save best model
            if eval_reward > best_reward:
                best_reward = eval_reward
                torch.save(agent.actor.state_dict(), f"{save_dir}/best_actor.pth")
                torch.save(agent.critic.state_dict(), f"{save_dir}/best_critic.pth")
                print(f"Saved best model, reward: {best_reward:.2f}")
        
        # Periodically save model checkpoints
        if episode % SAVE_INTERVAL == 0:
            torch.save(agent.actor.state_dict(), f"{save_dir}/actor_{episode}.pth")
            torch.save(agent.critic.state_dict(), f"{save_dir}/critic_{episode}.pth")
            
            # Save training curves
            plot_training_results(episode_rewards, avg_rewards, eval_rewards, eval_freq, save_dir)
            plot_loss(agent, save_dir)
            plot_q_values(agent, save_dir)
            plot_dual_q_values(agent, save_dir)
    
    # Save final model
    torch.save(agent.actor.state_dict(), f"{save_dir}/actor_final.pth")
    torch.save(agent.critic.state_dict(), f"{save_dir}/critic_final.pth")
    
    # Plot training curves
    plot_training_results(episode_rewards, avg_rewards, eval_rewards, eval_freq, save_dir)
    plot_loss(agent, save_dir)
    plot_q_values(agent, save_dir)
    plot_dual_q_values(agent, save_dir)
    # Close environment
    print("Training complete. Saving final logs...")
    logger.save_to_csv()
    env.close()


def evaluate_agent(env, agent, logger=None, n_episodes=5):
    """Evaluate agent performance in the environment"""
    rewards = []
    
    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        print(f"eval_episode:{_}")
        while not done and step_count<MAX_STEPS:
            # Use deterministic policy (no exploration noise)
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
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='TD3 Quadrotor Training')
    parser.add_argument('--initial', type=float, nargs=3, default=[0, 0, 0], help='Initial position [x, y, z]')
    parser.add_argument('--target', type=float, nargs=3, default=[1, 1, 2], help='Target position [x, y, z]')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--eval_freq', type=int, default=ECAL_FREQ, help='Evaluation frequency')
    parser.add_argument('--episodes', type=int, default=MAX_EPISODES, help='Number of training episodes')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save models and logs')
    parser.add_argument('--save_log_freq', type=int, default=50, help='Frequency to save log files (in episodes)')
    
    args = parser.parse_args()
    
    # Run training (will automatically check for available models to continue training)
    train(
        initial_position=args.initial,
        target_position=args.target,
        seed=args.seed,
        eval_freq=args.eval_freq,
        max_episodes=args.episodes,
        save_dir=args.save_dir,
        save_log_freq=args.save_log_freq
    )