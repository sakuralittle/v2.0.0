import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random

# ==================== SAC 模型 ====================

class Actor(nn.Module):
    """SAC Actor 網絡（高斯策略）"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # 重參數化技巧
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    """SAC Critic 網絡（雙 Q 網絡）"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # Q1
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        # Q2
        self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        # Q1
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        # Q2
        q2 = F.relu(self.fc4(x))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2


class ReplayBuffer:
    """經驗回放緩衝區"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class SACAgent:
    """SAC 代理（支援訓練和推理）"""
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, train_mode=False, device='auto', force_cuda=False):
        # 設備選擇邏輯
        if device == 'auto':
            if force_cuda:
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "強制使用 CUDA 但 GPU 不可用！\n"
                        "請確認：\n"
                        "1. 是否安裝了 CUDA 版本的 PyTorch\n"
                        "2. 系統是否有可用的 NVIDIA GPU\n"
                        "3. CUDA 驅動是否正確安裝"
                    )
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError("指定使用 CUDA 但 GPU 不可用！")
            self.device = torch.device("cuda")
        elif device == 'cpu':
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"無效的設備選項: {device}，請使用 'auto', 'cuda' 或 'cpu'")
        
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.train_mode = train_mode
        
        # Actor 網絡
        self.actor = Actor(state_dim, action_dim).to(self.device)
        
        if train_mode:
            # 訓練模式：需要 Critic 網絡和優化器
            self.critic = Critic(state_dim, action_dim).to(self.device)
            self.critic_target = Critic(state_dim, action_dim).to(self.device)
            self.critic_target.load_state_dict(self.critic.state_dict())
            
            # 優化器
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
            
            self.actor.train()
            self.critic.train()
            self.critic_target.eval()
        else:
            # 推理模式
            self.actor.eval()
        
    def select_action(self, state, evaluate=False):
        """選擇動作
        
        參數:
            state: 狀態
            evaluate: 是否為評估模式（True 時使用確定性策略）
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if evaluate:
                # 評估模式：使用確定性策略
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
            else:
                # 訓練模式：使用隨機策略
                action, _ = self.actor.sample(state)
        return action.cpu().numpy()[0]
    
    def update(self, replay_buffer, batch_size):
        """更新網絡參數"""
        if not self.train_mode:
            raise RuntimeError("Agent 未處於訓練模式")
        
        # 從 replay buffer 採樣
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        # ========== 更新 Critic ==========
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            q1_next, q2_next = self.critic_target(next_state, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            q_target = reward + (1 - done) * self.gamma * q_next
        
        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ========== 更新 Actor ==========
        new_action, log_prob = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ========== 軟更新目標網絡 ==========
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
        }
    
    def save(self, path):
        """保存模型"""
        checkpoint = {'actor': self.actor.state_dict()}
        if self.train_mode:
            checkpoint['critic'] = self.critic.state_dict()
            checkpoint['critic_target'] = self.critic_target.state_dict()
            checkpoint['actor_optimizer'] = self.actor_optimizer.state_dict()
            checkpoint['critic_optimizer'] = self.critic_optimizer.state_dict()
        torch.save(checkpoint, path)
    
    def load(self, path):
        """載入模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        
        if self.train_mode and 'critic' in checkpoint:
            self.critic.load_state_dict(checkpoint['critic'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            if 'actor_optimizer' in checkpoint:
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            if 'critic_optimizer' in checkpoint:
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        if not self.train_mode:
            self.actor.eval()


# ==================== 工廠函數 ====================

def create_sac_agent(state_dim, action_dim, learning_rate=3e-4, gamma=0.99, tau=0.005, alpha=0.2, train_mode=False, device='auto', force_cuda=False):
    """創建 SAC 代理
    
    參數:
        state_dim: 狀態維度
        action_dim: 動作維度
        learning_rate: 學習率
        gamma: 折扣因子
        tau: 軟更新係數
        alpha: 熵溫度參數
        train_mode: 是否為訓練模式
        device: 設備選擇 ('auto', 'cuda', 'cpu')
        force_cuda: 是否強制使用 CUDA
    """
    return SACAgent(state_dim, action_dim, learning_rate, gamma, tau, alpha, train_mode, device, force_cuda)
