import os
import glob
import re
import torch
import numpy as np
import pandas as pd

def check_and_load_model(
    agent,
    save_dir="checkpoints",
    device="cpu"
):
    """
    Check if there are usable model files in the specified directory, load them and restore training metrics if available
    
    Args:
        agent: TD3Agent instance, used to load model weights and metrics
        save_dir: Directory containing models and logs
        device: Device to run on ('cpu' or 'cuda')
        
    Returns:
        loaded: Boolean, indicates whether model was successfully loaded
        start_episode: Integer, next training episode number
        metrics: Dictionary, contains previously recorded training metrics
            - episode_rewards: List of previous episode rewards
            - avg_rewards: List of previous average rewards
            - eval_rewards: List of previous evaluation rewards
    """
    # Check if directory exists
    if not os.path.exists(save_dir):
        print(f"Directory {save_dir} does not exist. Will create this directory and start new training.")
        os.makedirs(save_dir, exist_ok=True)
        return False, 1, {"episode_rewards": [], "avg_rewards": [], "eval_rewards": []}
    
    # Look for the latest model checkpoint
    actor_files = glob.glob(f"{save_dir}/actor_*.pth")
    if not actor_files:
        print(f"No actor models found in {save_dir}. Starting new training.")
        return False, 1, {"episode_rewards": [], "avg_rewards": [], "eval_rewards": []}
    
    # Extract episode numbers from filenames
    episode_numbers = []
    for file in actor_files:
        match = re.search(r'actor_(\d+)\.pth', file)
        if match:
            episode_numbers.append(int(match.group(1)))
    
    if not episode_numbers:
        print(f"No valid actor models found in {save_dir}. Starting new training.")
        return False, 1, {"episode_rewards": [], "avg_rewards": [], "eval_rewards": []}
    
    # Get the latest episode number
    latest_episode = max(episode_numbers)
    print(f"Found model checkpoint at episode {latest_episode}. Will continue training...")
    
    # Confirm corresponding critic file exists
    critic_path = f"{save_dir}/critic_{latest_episode}.pth"
    if not os.path.exists(critic_path):
        print(f"Corresponding critic model {critic_path} not found. Starting new training.")
        return False, 1, {"episode_rewards": [], "avg_rewards": [], "eval_rewards": []}

    # Load metrics from CSV files (if they exist)
    training_summary_path = f"{save_dir}/training_summary.csv"
    q_values_path = f"{save_dir}/q_values_and_losses.csv"
    
    episode_rewards = []
    avg_rewards = []
    eval_rewards = []
    critic_losses = []
    actor_losses = []
    q_values = []
    q1_values = []
    q2_values = []
    
    if os.path.exists(training_summary_path):
        try:
            training_df = pd.read_csv(training_summary_path)
            episode_rewards = training_df['reward'].tolist()
            avg_rewards = training_df['avg_reward'].tolist()
            
            # Check if evaluation metrics exist
            eval_path = f"{save_dir}/evaluation_results.csv"
            if os.path.exists(eval_path):
                eval_df = pd.read_csv(eval_path)
                eval_rewards = eval_df['eval_reward'].tolist()
        except Exception as e:
            print(f"Error loading training metrics: {e}")
    
    if os.path.exists(q_values_path):
        try:
            q_df = pd.read_csv(q_values_path)
            critic_losses = q_df['critic_loss'].tolist()
            actor_losses = q_df['actor_loss'].tolist()
            q_values = q_df['q_value'].tolist()
            if 'q1_value' in q_df.columns and 'q2_value' in q_df.columns:
                q1_values = q_df['q1_value'].tolist()
                q2_values = q_df['q2_value'].tolist()
        except Exception as e:
            print(f"Error loading Q values and losses: {e}")
    
    # Load model weights
    try:
        actor_path = f"{save_dir}/actor_{latest_episode}.pth"
        
        agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
        agent.actor_target.load_state_dict(torch.load(actor_path, map_location=device))
        
        agent.critic.load_state_dict(torch.load(critic_path, map_location=device))
        agent.critic_target.load_state_dict(torch.load(critic_path, map_location=device))
        
        # Restore training metrics
        agent.critic_losses = critic_losses
        agent.actor_losses = actor_losses
        agent.q_values = q_values
        agent.q1_values = q1_values
        agent.q2_values = q2_values
        agent.total_it = len(critic_losses) if critic_losses else 0
        
        print(f"Successfully loaded model weights and training metrics from episode {latest_episode}")
        return True, latest_episode + 1, {
            "episode_rewards": episode_rewards,
            "avg_rewards": avg_rewards,
            "eval_rewards": eval_rewards
        }
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Will start new training.")
        return False, 1, {"episode_rewards": [], "avg_rewards": [], "eval_rewards": []}