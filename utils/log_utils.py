import pandas as pd
import numpy as np
from datetime import datetime
import os
import traceback

class TrainingLogger:
    """用於記錄和儲存強化學習訓練日誌的類（固定文件名 CSV 版本）"""
    
    def __init__(self, save_dir="logs", save_frequency=0):
        """
        初始化訓練日誌記錄器
        
        參數:
            save_dir: 儲存日誌的目錄
            save_frequency: 多久儲存一次 CSV 檔案（以回合為單位）
                            0 表示只在結束時儲存
                            N 表示每 N 個回合儲存一次
        """
        self.save_dir = save_dir
        self.save_frequency = save_frequency
        
        # 確保儲存目錄存在
        try:
            os.makedirs(save_dir, exist_ok=True)
            print(f"Log directory created/verified: {save_dir}")
        except Exception as e:
            print(f"ERROR creating directory {save_dir}: {e}")
            # 如果無法建立目錄，則使用當前目錄
            self.save_dir = "."
            print(f"Using current directory instead")
        
        # 設定固定文件名
        self.file_names = {
            'training_summary': f"{self.save_dir}/training_summary.csv",
            'evaluation_results': f"{self.save_dir}/evaluation_results.csv",
            'control_parameters': f"{self.save_dir}/control_parameters.csv",
            'q_values_and_losses': f"{self.save_dir}/q_values_and_losses.csv",
            'trajectory_record': f"{self.save_dir}/trajectory_record.csv"
        }
        
        # Training summary log
        self.training_log = {
            "episode": [],
            "reward": [],
            "avg_reward": [],
            "step_count": [],
            "reached_target": [],
            "distance_to_target": []
        }
        
        # Control parameters log
        self.action_log = {
            "episode": [],
            "step": [],
            "lambda_att": [],
            "eta_att": [],
            "lambda_pos": [],
            "eta_pos": []
        }
        
        # Evaluation log
        self.eval_log = {
            "episode": [],
            "eval_reward": []
        }
        
        # Q-values and loss functions
        self.q_loss_log = {
            "iteration": [],
            "critic_loss": [],
            "actor_loss": [],
            "q_value": [],
            "q1_value": [],
            "q2_value": []
        }
        
        # Trajectory record
        self.trajectory_log = {
            "episode": [],
            "step": [],
            "x": [],
            "y": [],
            "z": [],
            "target_x": [],
            "target_y": [],
            "target_z": []
        }
        
        # Latest saved episode
        self.last_saved_episode = 0
        
        print("TrainingLogger initialized successfully")
    
    def log_episode(self, episode, reward, avg_reward, step_count, info=None):
        """Record training info for a single episode"""
        try:
            self.training_log["episode"].append(episode)
            self.training_log["reward"].append(reward)
            self.training_log["avg_reward"].append(avg_reward)
            self.training_log["step_count"].append(step_count)
            
            # Record additional info if provided
            if info:
                self.training_log["reached_target"].append(info.get("reached_target", False))
                self.training_log["distance_to_target"].append(info.get("distance_to_target", float("nan")))
            else:
                self.training_log["reached_target"].append(False)
                self.training_log["distance_to_target"].append(float("nan"))
            
            # Auto save based on frequency
            if self.save_frequency > 0 and episode % self.save_frequency == 0:
                print(f"Auto-saving log at episode {episode}")
                self.save_to_csv()
                self.last_saved_episode = episode
        except Exception as e:
            print(f"ERROR in log_episode: {e}")
            traceback.print_exc()
    
    def log_action(self, episode, step, action):
        """Record control parameters for a single step"""
        try:
            self.action_log["episode"].append(episode)
            self.action_log["step"].append(step)
            self.action_log["lambda_att"].append(float(action[0]))
            self.action_log["eta_att"].append(float(action[1]))
            self.action_log["lambda_pos"].append(float(action[2]))
            self.action_log["eta_pos"].append(float(action[3]))
        except Exception as e:
            print(f"ERROR in log_action: {e}")
            traceback.print_exc()
    
    def log_evaluation(self, episode, eval_reward):
        """Record evaluation results"""
        try:
            self.eval_log["episode"].append(episode)
            self.eval_log["eval_reward"].append(eval_reward)
        except Exception as e:
            print(f"ERROR in log_evaluation: {e}")
            traceback.print_exc()
    
    def log_agent_metrics(self, iteration, agent):
        """Record agent's Q-values and loss functions"""
        try:
            # Only record new values
            if hasattr(agent, 'critic_losses') and len(agent.critic_losses) > len(self.q_loss_log["iteration"]):
                # Record latest iteration
                self.q_loss_log["iteration"].append(iteration)
                
                # Record critic loss (if available)
                if agent.critic_losses:
                    self.q_loss_log["critic_loss"].append(agent.critic_losses[-1])
                else:
                    self.q_loss_log["critic_loss"].append(float("nan"))
                
                # Record actor loss (if available)
                if hasattr(agent, 'actor_losses') and agent.actor_losses:
                    self.q_loss_log["actor_loss"].append(agent.actor_losses[-1])
                else:
                    self.q_loss_log["actor_loss"].append(float("nan"))
                    
                # Record Q values (if available)
                if agent.q_values:
                    self.q_loss_log["q_value"].append(agent.q_values[-1])
                else:
                    self.q_loss_log["q_value"].append(float("nan"))
                    
                # Record Q1 and Q2 values (if available)
                if hasattr(agent, 'q1_values') and agent.q1_values:
                    self.q_loss_log["q1_value"].append(agent.q1_values[-1])
                else:
                    self.q_loss_log["q1_value"].append(float("nan"))
                    
                if hasattr(agent, 'q2_values') and agent.q2_values:
                    self.q_loss_log["q2_value"].append(agent.q2_values[-1])
                else:
                    self.q_loss_log["q2_value"].append(float("nan"))
        except Exception as e:
            print(f"ERROR in log_agent_metrics: {e}")
            traceback.print_exc()
    
    def log_trajectory(self, episode, step, position, target_position):
        """Record agent's movement trajectory"""
        try:
            self.trajectory_log["episode"].append(episode)
            self.trajectory_log["step"].append(step)
            self.trajectory_log["x"].append(float(position[0]))
            self.trajectory_log["y"].append(float(position[1]))
            self.trajectory_log["z"].append(float(position[2]))
            self.trajectory_log["target_x"].append(float(target_position[0]))
            self.trajectory_log["target_y"].append(float(target_position[1]))
            self.trajectory_log["target_z"].append(float(target_position[2]))
        except Exception as e:
            print(f"ERROR in log_trajectory: {e}")
            traceback.print_exc()
    
    def save_to_csv(self):
        """Save logs to CSV files with fixed names (overwriting previous files)"""
        try:
            # Check if there's data to save
            has_data = (
                len(self.training_log["episode"]) > 0 or
                len(self.eval_log["episode"]) > 0 or
                len(self.action_log["episode"]) > 0 or
                len(self.q_loss_log["iteration"]) > 0 or
                len(self.trajectory_log["episode"]) > 0
            )
            
            if not has_data:
                print("No data to save!")
                return None
            
            # Ensure directory exists
            os.makedirs(self.save_dir, exist_ok=True)
            
            saved_files = []
            
            # Training summary sheet
            if self.training_log["episode"]:
                print("Saving Training_Summary...")
                train_df = pd.DataFrame(self.training_log)
                train_df.to_csv(self.file_names['training_summary'], index=False)
                saved_files.append(self.file_names['training_summary'])
            
            # Evaluation results sheet
            if self.eval_log["episode"]:
                print("Saving Evaluation_Results...")
                eval_df = pd.DataFrame(self.eval_log)
                eval_df.to_csv(self.file_names['evaluation_results'], index=False)
                saved_files.append(self.file_names['evaluation_results'])
            
            # Action record sheet
            if self.action_log["episode"]:
                print("Saving Control_Parameters...")
                action_df = pd.DataFrame(self.action_log)
                action_df.to_csv(self.file_names['control_parameters'], index=False)
                saved_files.append(self.file_names['control_parameters'])
            
            # Q-values and loss sheet
            if self.q_loss_log["iteration"]:
                print("Saving Q_Values_and_Losses...")
                q_loss_df = pd.DataFrame(self.q_loss_log)
                q_loss_df.to_csv(self.file_names['q_values_and_losses'], index=False)
                saved_files.append(self.file_names['q_values_and_losses'])
            
            # Trajectory record sheet
            if self.trajectory_log["episode"]:
                print("Saving Trajectory_Record...")
                traj_df = pd.DataFrame(self.trajectory_log)
                traj_df.to_csv(self.file_names['trajectory_record'], index=False)
                saved_files.append(self.file_names['trajectory_record'])
            
            print(f"Training logs successfully saved to {self.save_dir}")
            return saved_files
        except Exception as e:
            print(f"ERROR saving CSV files: {e}")
            traceback.print_exc()
            return None

    def force_save(self):
        """Force save the current log data"""
        print("Force saving log data...")
        return self.save_to_csv()