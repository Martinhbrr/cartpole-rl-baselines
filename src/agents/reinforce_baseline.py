import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F
class PolicyNet(nn.Module): #decision maker
    def __init__(self, obs_dim: int, hidden: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ValueNet(nn.Module): # evaluator
    def __init__(self, obs_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ReinforceBaselineAgent:
    def __init__(self):
        obs_dim=4
        act_dim=2
        hidden=128
        gamma=0.99
        self.gamma=gamma
        self.obs_dim=obs_dim
        self.act_dim=act_dim

        self.policy=PolicyNet(obs_dim,hidden,act_dim)
        self.value_net=ValueNet(obs_dim,hidden)

        lr_policy=1e-3
        lr_value=1e-3
        self.opt_policy=optim.Adam(self.policy.parameters(),lr=lr_policy)
        self.opt_value=optim.Adam(self.value_net.parameters(),lr=lr_value)

        self.states=[]
        self.log_probs=[]
        self.rewards=[]

        self.device=torch.device("cpu")
        

    def act(self,state):
        s = torch.tensor(state, dtype=torch.float32, device=self.device)
        logits = self.policy(s)
        dist = Categorical(logits=logits)
        a = dist.sample()
        self.log_probs.append(dist.log_prob(a))
        return int(a.item())

    def observe(self,state,action,reward,next_state,done):
        self.states.append(state)
        self.rewards.append(reward)


    def end_episode(self):
        if len(self.rewards) == 0:
            return

        returns = []
        G = 0.0
        for r in reversed(self.rewards):
            G = float(r) + self.gamma * G
            returns.append(G)
        returns.reverse()

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        if isinstance(self.states[0], torch.Tensor):
            states = torch.stack([s.to(self.device) for s in self.states])
        else:
            states = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)

        values = self.value_net(states)

        advantages = returns - values
        advantages_actor = advantages.detach()
        advantages_actor = (advantages_actor - advantages_actor.mean()) / (advantages_actor.std() + 1e-8)

        log_probs = torch.stack(self.log_probs).to(self.device)

        actor_loss = -(log_probs * advantages_actor).mean()
        critic_loss = F.mse_loss(values, returns)

        self.opt_policy.zero_grad()
        actor_loss.backward()
        self.opt_policy.step()

        self.opt_value.zero_grad()
        critic_loss.backward()
        self.opt_value.step()

        self.states.clear()
        self.log_probs.clear()
        self.rewards.clear()

