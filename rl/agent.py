import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """Actor network (policy) with hidden layers [256, 256] and tanh output scaled by max_action."""
    def __init__(self, state_dim, action_dim, max_action, hidden_sizes):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_sizes[0])
        self.bn1 = nn.LayerNorm(hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.LayerNorm(hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], action_dim)
        self.max_action = max_action #讓最後的輸出不會是只有[-1, 1]

    def forward(self, state):
        x = F.relu(self.bn1(self.l1(state)))
        x = F.relu(self.bn2(self.l2(x)))
        x = torch.tanh(self.l3(x))
        return x * self.max_action

class Critic(nn.Module):
    """Critic network (value) with two Q-networks for TD3."""
    def __init__(self, state_dim, action_dim, hidden_sizes):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], 1)
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_sizes[0])
        self.l5 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l6 = nn.Linear(hidden_sizes[1], 1)

        self.bn1 = nn.LayerNorm(hidden_sizes[0])
        self.bn2 = nn.LayerNorm(hidden_sizes[1])
        self.bn4 = nn.LayerNorm(hidden_sizes[0])
        self.bn5 = nn.LayerNorm(hidden_sizes[1])

    def forward(self, state, action):
        sa = torch.cat([state, action], 1) # 將兩個張量(tensor) 在第 1 維度上進行串接(concatenate)
        q1 = F.relu(self.bn1(self.l1(sa)))
        q1 = F.relu(self.bn2(self.l2(q1)))
        q1 = self.l3(q1)
        q2 = F.relu(self.bn4(self.l4(sa)))
        q2 = F.relu(self.bn5(self.l5(q2)))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        '''更新Actor只需要計算Q1的梯度，完全不需要Q2網路。'''
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.bn1(self.l1(sa)))
        q1 = F.relu(self.bn2(self.l2(q1)))
        q1 = self.l3(q1)
        return q1

class ReplayBuffer:
    """A simple FIFO experience replay buffer for TD3."""
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        # 建立儲存空間
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        if isinstance(state, torch.Tensor):
            state = state.cpu().detach().numpy()
        if isinstance(action, torch.Tensor):
            action = action.cpu().detach().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().detach().numpy()
        if isinstance(reward, torch.Tensor):
            reward = reward.cpu().detach().numpy()
        # 儲存一筆經驗（狀態、動作、獎勵、下個狀態、是否結束）
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        # 隨機抽一批經驗出來訓練
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            self.state[ind],
            self.action[ind],
            self.next_state[ind],
            self.reward[ind],
            self.done[ind],
        )

class TD3Agent:
    """TD3 agent with methods to select actions and update networks."""
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        hidden_sizes=[256, 256],
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        lr=1e-3,
        buffer_size=int(1e6)
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = torch.tensor(policy_delay, dtype=torch.float32).to(device)
        # 將 NumPy 陣列轉換為 PyTorch 張量

        self.policy_noise = torch.tensor(policy_noise, dtype=torch.float32).to(device)

            
        self.noise_clip = torch.tensor(noise_clip, dtype=torch.float32).to(device)

        if isinstance(max_action, np.ndarray):
            self.max_action = torch.tensor(max_action, dtype=torch.float32).to(device)
        else:
            self.max_action = torch.tensor([max_action], dtype=torch.float32).to(device)

        # 建立 Actor 和 Actor 的 target 網路
        # Initialize actor and critic networks
        self.actor = Actor(state_dim, action_dim, self.max_action, hidden_sizes).to(device)
        self.actor_target = Actor(state_dim, action_dim, self.max_action, hidden_sizes).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # 建立 Critic 和 Critic 的 target 網路
        self.critic = Critic(state_dim, action_dim, hidden_sizes).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_sizes).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=buffer_size)

        self.total_it = 0

        # 增加監控指標追蹤
        self.critic_losses = []
        self.actor_losses = []
        self.q_values = []
        self.action_values = []
        self.q1_values = []
        self.q2_values = []

    def select_action(self, state, noise=0.0):
        """Select an action given a state (with optional exploration noise).根據目前的狀態選擇動作，可以加入噪聲來探索"""
        state_tensor = torch.tensor(state.reshape(1, -1), dtype=torch.float32).to(self.device)
        action = self.actor(state_tensor).cpu().data.numpy().flatten()
        if noise == 0.0:
            self.actor.eval()  # 設為評估模式
            with torch.no_grad():
                action = self.actor(state_tensor).cpu().data.numpy().flatten()
            self.actor.train()  # 恢復訓練模式
        else:
            # 訓練時保持訓練模式
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
        # 追蹤動作值
        if hasattr(self, 'action_values'):
            self.action_values.append(action.copy())
            
        if noise != 0.0:
            action = action + np.random.normal(0, noise, size=action.shape)
        max_action_np = self.max_action.cpu().numpy() if torch.is_tensor(self.max_action) else self.max_action
        return np.clip(action, -max_action_np, max_action_np)

    def train(self, batch_size):
        """Sample a batch and update the networks."""
        if self.replay_buffer.size < batch_size:
            return # 資料不夠就不訓練
        self.total_it += 1

        # 從記憶庫中抽樣
        # Sample replay buffer
        state, action, next_state, reward, done = self.replay_buffer.sample(batch_size)
        # 轉換為張量並移至設備
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)

        # 計算下一步的動作，並加入隨機噪聲（增加探索）
        # Compute target actions with noise
        # 直接在這裡轉換為張量
        policy_noise_tensor = self.policy_noise.clone().detach()
        noise_clip_tensor = self.noise_clip.clone().detach()
        
        # 使用轉換後的張量進行操作
        noise = (torch.randn_like(action) * policy_noise_tensor).clamp(-noise_clip_tensor, noise_clip_tensor)

        # noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

        # Compute target Q-value
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (1 - done) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        # 追蹤Q值
        self.q_values.append(current_Q1.mean().item())
        self.q1_values.append(current_Q1.mean().item())  
        self.q2_values.append(current_Q2.mean().item())  

        # Critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q.detach()) + F.mse_loss(current_Q2, target_Q.detach())
        # 追蹤Critic損失
        self.critic_losses.append(critic_loss.item())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy (actor) updates
        # 延遲更新 Actor 網路（每 policy_delay 次才更新一次）
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean() #Loss function
            self.actor_losses.append(actor_loss.item())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target networks
            # 使用 soft update 更新 target 網路（讓學習更穩定）
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
