import random
import math
import numpy as np
import collections
import torch
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, terminated):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, terminated = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), terminated

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


class MDQnet(torch.nn.Module):
    def __init__(self, state_dim, hidden1_dim, hidden2_dim, action_dims):
        """
        多维Q网络，为每个动作维度输出独立的Q值
        :param action_dims: 列表，每个元素表示对应维度的动作数量，如[3, 2]
        """
        super().__init__()
        self.shared_fc1 = torch.nn.Linear(state_dim, hidden1_dim)
        self.shared_fc2 = torch.nn.Linear(hidden1_dim, hidden2_dim)

        # 为每个动作维度创建独立的输出层
        self.action_heads = torch.nn.ModuleList()
        for dim in action_dims:
            self.action_heads.append(torch.nn.Linear(hidden2_dim, dim))

        self.action_dims = action_dims

    def forward(self, x):
        # 共享特征提取
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))

        # 为每个动作维度生成Q值
        q_values = []
        for head in self.action_heads:
            q_values.append(head(x))

        return q_values


class MDDQN:
    def __init__(
        self,
        state_dim,
        hidden1_dim,
        hidden2_dim,
        action_dims,  # 改为动作维度列表，如[3, 2]
        learning_rate,
        gamma,
        epsilon,
        target_update,
        device,
        num_episodes,
    ):
        self.state_dim = state_dim
        self.action_dims = action_dims  # 多维动作空间描述
        self.num_action_dims = len(action_dims)  # 动作维度数量

        # Q网络和目标网络
        self.q_net = MDQnet(state_dim, hidden1_dim, hidden2_dim, action_dims).to(device)
        self.target_q_net = MDQnet(state_dim, hidden1_dim, hidden2_dim, action_dims).to(
            device
        )

        # 优化器
        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), betas=(0.9, 0.999), lr=learning_rate
        )

        # 学习率衰减
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=40000, gamma=0.2
        )

        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.sample_count = 0  # 用于epsilon衰减
        self.epsilon_start = 0.1
        self.epsilon_end = 0.001
        self.epsilon_decay = num_episodes
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):
        """采取多维动作，返回一个列表，每个元素对应一个维度的动作"""
        # epsilon衰减
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * math.exp(-1.0 * self.sample_count / self.epsilon_decay)

        if np.random.random() < self.epsilon:
            # 随机选择每个维度的动作
            action = [np.random.randint(dim) for dim in self.action_dims]
        else:
            # 基于Q值选择最优动作
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            q_values = self.q_net.forward(state)  # 得到每个维度的Q值
            action = [
                q.argmax().item() for q in q_values
            ]  # 每个维度取最大Q值对应的动作

        return action

    def update(self, transition_dict):
        """更新网络，处理多维动作的Q值计算"""
        states = (
            torch.tensor(transition_dict["states"], dtype=torch.float)
            .view(-1, self.state_dim)
            .to(self.device)
        )
        next_states = (
            torch.tensor(transition_dict["next_states"], dtype=torch.float)
            .view(-1, self.state_dim)
            .to(self.device)
        )
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        terminateds = (
            torch.tensor(transition_dict["terminateds"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )

        # 处理多维动作，转换为张量列表
        actions = []
        for dim in range(self.num_action_dims):
            # 提取每个维度的动作并转换为张量
            dim_actions = [a[dim] for a in transition_dict["actions"]]
            actions.append(
                torch.tensor(dim_actions, dtype=torch.long).view(-1, 1).to(self.device)
            )

        # 计算当前Q值
        current_q_list = self.q_net.forward(states)  # 每个元素是一个维度的Q值
        current_q_values = []
        for dim in range(self.num_action_dims):
            # 选取每个维度中实际执行的动作对应的Q值
            current_q = current_q_list[dim].gather(1, actions[dim])
            current_q_values.append(current_q)

        # 计算目标Q值
        with torch.no_grad():
            next_q_list = self.target_q_net.forward(
                next_states
            )  # 目标网络的下一状态Q值
            # 每个维度取最大Q值
            max_next_q_values = [q.max(1)[0].view(-1, 1) for q in next_q_list]

            # 计算每个维度的目标Q值
            target_q_values = []
            for dim in range(self.num_action_dims):
                target_q = rewards + self.gamma * max_next_q_values[dim] * (
                    1 - terminateds
                )
                target_q_values.append(target_q)

        # 计算每个维度的损失并求和
        total_loss = 0.0
        for dim in range(self.num_action_dims):
            loss = F.mse_loss(current_q_values[dim], target_q_values[dim])
            total_loss += loss
        total_loss /= self.num_action_dims  # 平均各维度损失

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # 更新目标网络
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        count = self.count
        self.count += 1

        return total_loss, count
