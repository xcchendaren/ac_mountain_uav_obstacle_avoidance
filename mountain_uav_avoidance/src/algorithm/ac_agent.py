"""
AC算法代理模块

基于PyTorch实现Actor-Critic算法，支持连续动作空间，包含经验回放和正确的策略梯度。
修正了tanh动作的对数概率计算，经验回放存储原始采样动作。
新增：支持保存和加载经验回放缓冲区，实现断点续训。
经验回放采用混合采样策略：70%均匀采样 + 30%高奖励优先采样，
平衡探索多样性与利用高质量经验。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import pickle
from pathlib import Path
import os


class ActorNetwork(nn.Module):
    """策略网络（Actor）：输出动作均值和对数标准差"""

    def __init__(self, state_dim, action_dim, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # 可学习对数标准差

        # ===== 探索退火：外部可注入的探索偏置（退火时加到 log_std 上）=====
        # 训练初期设为较大正值（如 1.0），随训练进度线性减小到 0
        # 这样探索量从大到小，避免策略过早收敛到局部解
        self.exploration_bonus = 0.0  # 由 ACAgent.set_exploration_bonus() 控制

        # 初始化均值输出接近0
        nn.init.uniform_(self.mean.weight, -3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.mean(x))          # 输出范围 [-1, 1]
        # 对 log_std 进行裁剪，防止梯度爆炸导致无穷大
        # 加上 exploration_bonus 实现退火：训练前期增大探索，后期减小
        log_std = torch.clamp(self.log_std + self.exploration_bonus, min=-2, max=2)
        log_std = log_std.expand_as(mean)
        return mean, log_std

    def sample(self, state):
        """
        采样动作，返回动作（已tanh）、对数概率（已修正）、原始动作（未tanh）
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        raw_action = normal.rsample()  # 重参数化采样（原始动作）
        log_prob = normal.log_prob(raw_action).sum(dim=-1)
        action = torch.tanh(raw_action)
        # tanh修正项：log(1 - tanh^2) 的导数
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        return action, log_prob, raw_action


class CriticNetwork(nn.Module):
    """价值网络（Critic）：输出状态值 V(s)"""

    def __init__(self, state_dim, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.value(x)


class ReplayBuffer:
    """
    经验回放缓冲区（混合采样策略）

    采样方式：70% 均匀随机 + 30% 高奖励优先（top 30% buffer）。
    - 均匀部分保证多样性
    - 高奖励部分加速学习高收益经验
    淘汰策略：池满时奖励低于中位数的新经验被丢弃（保留高质量经验）。
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []          # 存储 (state, action, reward, next_state, done)
        self.rewards = []         # 同步存储奖励，便于计算中位数
        self.threshold = -np.inf  # 奖励阈值
        self.need_update = True   # 是否需要重新计算阈值

        print(f"[ReplayBuffer] 初始化完成，容量: {capacity}")

    # ──────────────────────────────────────────────────────
    #  更新奖励阈值
    # ──────────────────────────────────────────────────────
    def update_threshold(self):
        """根据当前缓冲区中的奖励更新阈值（中位数）"""
        if len(self.rewards) == 0:
            self.threshold = -np.inf
        else:
            self.threshold = np.median(np.array(self.rewards))
        self.need_update = False
        print(f"[ReplayBuffer] 阈值已更新: {self.threshold:.3f} (基于 {len(self.rewards)} 条经验)")

    # ──────────────────────────────────────────────────────
    #  写入经验
    # ──────────────────────────────────────────────────────
    def push(self, state, action, reward, next_state, done):
        """写入一条经验，满时中位数阈值淘汰低质量经验"""
        exp = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
            self.rewards.append(reward)
            return

        # 池已满：中位数阈值淘汰
        # 首次满时自动计算阈值，后续由外部 update_buffer_threshold() 定期更新（每100轮）
        if self.need_update:
            self.update_threshold()
        if reward >= self.threshold:  # 使用最后一次计算的阈值
            min_idx = int(np.argmin(self.rewards))
            self.buffer[min_idx]  = exp
            self.rewards[min_idx] = reward
        # 否则丢弃新经验

    # ──────────────────────────────────────────────────────
    #  采样（混合策略）
    # ──────────────────────────────────────────────────────
    def sample(self, batch_size):
        """
        混合采样：70% 均匀随机 + 30% 高奖励优先。

        设计意图：
        - 均匀部分保证采样多样性，防止偏向高奖励区域
        - 高奖励部分让网络多见成功经验，加速学习到达目标的策略
        """
        n = len(self.buffer)
        if n < batch_size:
            raise ValueError(f"缓冲区经验不足: {n} < {batch_size}")

        n_high = max(1, int(batch_size * 0.30))   # 30%高奖励经验
        n_rand = batch_size - n_high               # 70%均匀经验

        # 高奖励：取奖励前30%的经验（topk），再从中随机选
        rewards_arr = np.array(self.rewards[:n])
        topk_count = max(n_high, int(n * 0.30))
        topk_indices = np.argsort(rewards_arr)[-topk_count:]
        high_indices = np.random.choice(topk_indices, n_high, replace=False).tolist()

        # 均匀：从剩余经验中随机采样
        remaining = list(set(range(n)) - set(high_indices))
        rand_indices = np.random.choice(remaining, min(n_rand, len(remaining)), replace=False).tolist()

        # 补足数量
        all_indices = high_indices + rand_indices
        if len(all_indices) < batch_size:
            extra_pool = list(set(range(n)) - set(all_indices))
            if extra_pool:
                extra = np.random.choice(extra_pool, batch_size - len(all_indices), replace=False).tolist()
                all_indices.extend(extra)

        indices = all_indices[:batch_size]
        batch = [self.buffer[i] for i in indices]
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class ACAgent:
    """
    Actor-Critic 智能体（带经验回放）
    修正版本：正确支持连续动作，经验中存储原始动作，训练时使用修正后的对数概率。
    支持保存/加载模型时同步保存/加载经验回放缓冲区。
    支持动态更新经验池奖励阈值（淘汰低质量经验）。
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        action_low,
        action_high,
        lr_actor=1e-3,
        lr_critic=1e-3,
        gamma=0.99,
        hidden_size=64,
        buffer_capacity=10000,
        batch_size=64,
        device="cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = np.array(action_low)
        self.action_high = np.array(action_high)
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

        self.actor = ActorNetwork(state_dim, action_dim, hidden_size).to(device)
        self.critic = CriticNetwork(state_dim, hidden_size).to(device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def _scale_action(self, action):
        """将[-1,1]的动作缩放到实际范围"""
        return 0.5 * (action + 1.0) * (self.action_high - self.action_low) + self.action_low

    def select_action(self, state):
        """
        根据状态选择动作（用于训练），返回 (缩放后的动作, 原始动作)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_t, _, raw_action_t = self.actor.sample(state_t)
            action_np = action_t.cpu().numpy().flatten()
            raw_action_np = raw_action_t.cpu().numpy().flatten()
        return self._scale_action(action_np), raw_action_np

    def predict(self, state, add_noise=True):
        """
        根据状态选择动作（用于推理），仅返回缩放后的动作

        NOTE: 与 sample() 保持一致的 tanh 处理逻辑：
              mean 已在 ActorNetwork.forward() 中通过 tanh 映射到 [-1, 1]，
              因此只需对 sample 结果做一次 tanh，不能重复应用。
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, log_std = self.actor(state_t)
            if add_noise:
                std = log_std.exp()
                normal = torch.distributions.Normal(mean, std)
                # sample() 与 sample() 一致：Normal(mean, std) → tanh(action)
                raw_action = normal.sample()
                action_t = torch.tanh(raw_action)
            else:
                # mean 已经是 tanh(linear_output)，无需再 tanh
                action_t = mean
            action_np = action_t.cpu().numpy().flatten()
        return self._scale_action(action_np)

    def store_transition(self, state, raw_action, reward, next_state, done):
        """
        存储经验
        Args:
            raw_action: 原始动作（采样后未 tanh 的值），形状 (action_dim,)
        """
        self.replay_buffer.push(state, raw_action, reward, next_state, done)

    def train(self):
        """
        从经验回放中采样批量更新网络。
        混合采样策略（70%均匀 + 30%高奖励优先），标准 Actor-Critic 更新流程。
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        states, raw_actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        states      = states.to(self.device)
        raw_actions = raw_actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # 计算目标值
        with torch.no_grad():
            next_values = self.critic(next_states)
            targets = rewards + self.gamma * next_values * (1 - dones)

        # 更新 Critic
        current_values = self.critic(states)
        critic_loss = F.mse_loss(current_values, targets)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.optimizer_critic.step()

        # 更新 Actor：使用存储的原始动作计算对数概率
        # 注意：这是 Off-Policy Actor-Critic 实现，存储的经验来自旧策略，
        # 但 log_prob(raw_action) 用当前策略 Normal(mean, std) 计算。
        # 理论上需要重要性采样修正，但由于经验池容量较小(10k)且每步训练，
        # 策略变化缓慢，实际中可工作。如需严格修正，可引入 IS 权重 clip。
        mean, log_std = self.actor(states)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        log_probs = normal.log_prob(raw_actions).sum(dim=-1)
        with torch.no_grad():
            advantages = (targets - current_values).detach().squeeze(-1)
        # 优势标准化（稳定训练，防止梯度尺度不一致）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        actor_loss = -(log_probs * advantages).mean()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.optimizer_actor.step()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "value": current_values.mean().item(),
        }

    def update_buffer_threshold(self):
        """强制更新经验回放缓冲区的奖励阈值（由外部调用，例如每100轮）"""
        self.replay_buffer.update_threshold()

    def set_exploration_bonus(self, bonus: float):
        """
        设置探索奖励偏置（退火控制）。
        
        在 ActorNetwork.forward() 中，log_std = clamp(log_std + bonus, -2, 2)
        bonus > 0 → 增大动作标准差 → 更多探索
        bonus = 0 → 纯学习状态（训练收敛后期）
        
        典型用法（在 TrainController._run_one_epoch 中调用）：
            progress = epoch / total_epochs           # 0.0 → 1.0
            bonus = max(0.0, 1.5 * (1 - progress))   # 线性从1.5退火到0
            agent.set_exploration_bonus(bonus)
        """
        self.actor.exploration_bonus = float(bonus)

    def save(self, path, save_buffer=True):
        """
        保存模型，可选保存经验回放缓冲区
        """
        # 将路径转换为 Path 对象并解析为绝对路径
        path = Path(path).resolve()
        parent_dir = path.parent

        # 确保父目录存在，如果创建失败则抛出明确异常
        try:
            parent_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"无法创建父目录 {parent_dir}: {e}")

        # 验证目录是否存在（多次检查确保文件系统同步）
        if not parent_dir.exists():
            # 尝试等待文件系统同步
            import time
            time.sleep(0.1)
            if not parent_dir.exists():
                raise RuntimeError(f"父目录 {parent_dir} 不存在，创建后仍然不存在")
        if not os.access(str(parent_dir), os.W_OK):
            raise RuntimeError(f"父目录 {parent_dir} 不可写")

        print(f"[ACAgent] 保存模型到: {path}")
        print(f"[ACAgent] 父目录: {parent_dir}, 存在: {parent_dir.exists()}")

        # 使用绝对路径字符串
        abs_path = str(parent_dir / path.name)
        print(f"[ACAgent] 使用绝对路径: {abs_path}")

        # 构建模型状态
        model_state = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }

        # 保存模型 - 使用文件句柄而不是路径字符串，避免 torch 的路径处理问题
        try:
            with open(abs_path, 'wb') as f:
                torch.save(model_state, f)
            print(f"[ACAgent] 模型已保存: {abs_path}")
        except Exception as e:
            print(f"[ACAgent] torch.save 失败: {e}")
            raise

        # 保存缓冲区
        if save_buffer:
            buffer_path = parent_dir / (path.name + '.buffer.pkl')
            buffer_abs_path = str(buffer_path)
            try:
                with open(buffer_abs_path, 'wb') as f:
                    pickle.dump(self.replay_buffer.buffer, f)
                print(f"[ACAgent] 缓冲区已保存: {buffer_abs_path}")
            except Exception as e:
                print(f"[ACAgent] 保存缓冲区失败: {e}")
                raise

    def load(self, path, load_buffer=True):
        """
        加载模型，可选加载经验回放缓冲区
        """
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")

        # 使用文件句柄加载，避免路径处理问题
        with open(str(path), 'rb') as f:
            checkpoint = torch.load(f, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])

        if load_buffer:
            # 与 save() 保持一致：父目录 / (模型文件名 + '.buffer.pkl')
            buffer_path = path.parent / (path.name + '.buffer.pkl')
            if buffer_path.exists():
                try:
                    with open(buffer_path, 'rb') as f:
                        loaded_buffer = pickle.load(f)
                        # 注意：加载的缓冲区是旧的列表格式，需要转换为新的带阈值机制的结构
                        self.replay_buffer.buffer = loaded_buffer
                        # 重新构建 rewards 列表
                        self.replay_buffer.rewards = [exp[2] for exp in loaded_buffer]
                        # 重置阈值，让下一次 push 时重新计算
                        self.replay_buffer.threshold = -np.inf
                        self.replay_buffer.need_update = True
                except Exception as e:
                    print(f"[ACAgent] 加载缓冲区失败: {e}")
                    # 不抛出异常，仅记录日志


if __name__ == "__main__":
    state_dim = 9
    action_dim = 3
    action_low = [-5, -5, -5]
    action_high = [5, 5, 5]
    agent = ACAgent(state_dim, action_dim, action_low, action_high)
    state = np.random.randn(state_dim)
    action, raw_action = agent.select_action(state)
    print("缩放后动作:", action)
    print("原始动作:", raw_action)
    action2 = agent.predict(state, add_noise=False)
    print("无噪声动作:", action2)