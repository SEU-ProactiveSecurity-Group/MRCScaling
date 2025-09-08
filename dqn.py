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


class Qnet(torch.nn.Module):
    def __init__(self, state_dim, hidden1_dim, hidden2_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden1_dim)
        self.fc2 = torch.nn.Linear(hidden1_dim, hidden2_dim)
        self.fc3 = torch.nn.Linear(hidden2_dim, action_dim)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        x2 = F.relu(self.fc2(x1))
        return self.fc3(x2)


class DQN:
    def __init__(
        self,
        state_dim,
        hidden1_dim,
        hidden2_dim,
        action_dim,
        learning_rate,
        gamma,
        epsilon,
        target_update,
        device,
        num_episodes,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden1_dim, hidden2_dim, self.action_dim).to(
            device
        )  # Q网络
        # 目标网络
        self.target_q_net = Qnet(
            state_dim, hidden1_dim, hidden2_dim, self.action_dim
        ).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), betas=(0.9, 0.999), lr=learning_rate
        )

        # 使用learning rate decay
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=40000, gamma=0.2
        )

        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.sample_count = (
            0  # 按一定概率随机选动作，即 e-greedy 策略，并且epsilon逐渐衰减
        )
        self.epsilon_start = 0.1
        self.epsilon_end = 0.001
        self.epsilon_decay = num_episodes
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        # epsilon衰减
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * math.exp(-1.0 * self.sample_count / self.epsilon_decay)
        # if self.epsilon > self.epsilon_end:  # 随机性衰减
        #     self.epsilon *= self.epsilon_decay
        #     self.epsilon = max(self.epsilon_end, self.epsilon)xaxa

        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            action = self.q_net.forward(state).argmax().item()

        return action

    def update(self, transition_dict):
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
        actions = torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)

        q_values = self.q_net.forward(states).gather(
            1, actions
        )  # Q值,gather函数用来选取state中的值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net.forward(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (
            1 - terminateds
        )  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数

        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()
        self.scheduler.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络

        count = self.count
        self.count += 1

        return dqn_loss, count
