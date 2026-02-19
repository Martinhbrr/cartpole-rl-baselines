import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state=np.asarray(state,dtype=np.float32).reshape(-1)
        next_state = np.asarray(next_state, dtype=np.float32).reshape(-1) # guarantees shape safety
        self.buffer.append((state, int(action), float(reward), next_state, float(done)))
       
    def sample(self, batch_size, device="cpu"):
        batch=random.sample(self.buffer,batch_size)
        states,actions,rewards,next_states,dones= zip(*batch) #turns rows into columns
        states=torch.tensor(np.stack(states),dtype=torch.float32,device=device)
        next_states=torch.tensor(np.stack(next_states),dtype=torch.float32,device=device) #preserves 2D array
        actions=torch.tensor(actions,dtype=torch.int64,device=device)
        rewards=torch.tensor(rewards,dtype=torch.float32, device=device)
        dones=torch.tensor(dones,dtype=torch.float32, device=device)

        return states,actions,rewards,next_states,dones
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        obs_dim,
        n_actions,
        gamma=0.99,
        lr=1e-3,
        buffer_size=50000,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=20000,
        target_update_freq=1000,
        device="cpu",
    ):
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.q_net = QNetwork(obs_dim, n_actions).to(device)
        self.target_net = QNetwork(obs_dim, n_actions).to(device)

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(buffer_size)

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        self.total_steps = 0

    def act(self, state):
        state=np.asanyarray(state,dtype=np.float32).reshape(-1)
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, obs_dim)
            q = self.q_net(s)  # (1, n_actions)
            action = torch.argmax(q, dim=1).item()  # convert tensor -> python int

        return action



    def observe(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state,action,reward,next_state,done)
        self.total_steps +=1

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        states,actions,rewards,next_states,dones = \
            self.replay_buffer.sample(self.batch_size,device=self.device)
        
        q_values= self.q_net(states)
        q_sa=q_values.gather(1,actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q = next_q_values.max(dim=1).values
            target = rewards + self.gamma * (1 - dones) * max_next_q

        loss=torch.nn.functional.mse_loss(q_sa,target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())



    def update_epsilon(self):
        decay_ratio=min(self.total_steps/self.epsilon_decay_steps, 1.0)
        self.epsilon=self.epsilon_start - decay_ratio*(self.epsilon_start-self.epsilon_end)

