import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


class PolicyNet(nn.Module):  # decision maker
    def __init__(self, obs_dim: int, hidden: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits


class ValueNet(nn.Module):  # evaluator
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
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: int = 128,
        gamma: float = 0.99,
        lr_policy: float = 1e-3,
        lr_value: float = 1e-3,
        device: str = "cpu",
    ):
        self.gamma = float(gamma)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.device = torch.device(device)

        self.policy = PolicyNet(self.obs_dim, hidden, self.act_dim).to(self.device)
        self.value_net = ValueNet(self.obs_dim, hidden).to(self.device)

        self.opt_policy = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.opt_value = optim.Adam(self.value_net.parameters(), lr=lr_value)

        self.states: list[np.ndarray] = []
        self.log_probs: list[torch.Tensor] = []
        self.rewards: list[float] = []

    def act(self, state) -> int:
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        s_t = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, obs_dim)

        logits = self.policy(s_t).squeeze(0)  # (act_dim,)
        dist = Categorical(logits=logits)
        a = dist.sample()

        self.log_probs.append(dist.log_prob(a))
        return int(a.item())

    def observe(self, state, action, reward, next_state, done):
        # action/next_state/done not needed for reinforce+baseline, but kept for a consistent interface
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        self.states.append(s)
        self.rewards.append(float(reward))

    def end_episode(self):
        if len(self.rewards) == 0:
            return 0.0, 0.0

        # Compute returns
        returns = []
        G = 0.0
        for r in reversed(self.rewards):
            G = float(r) + self.gamma * G
            returns.append(G)
        returns.reverse()
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Stack states
        states_t = torch.tensor(np.stack(self.states), dtype=torch.float32, device=self.device)

        # Critic values and advantages
        values = self.value_net(states_t)  # (T,)
        advantages = returns_t - values

        # normalize advantages for the actor
        adv_actor = advantages.detach()
        adv_actor = (adv_actor - adv_actor.mean()) / (adv_actor.std() + 1e-8)

        log_probs_t = torch.stack(self.log_probs).to(self.device)  # (T,)

        actor_loss = -(log_probs_t * adv_actor).mean()
        critic_loss = F.mse_loss(values, returns_t)

        # Update policy
        self.opt_policy.zero_grad()
        actor_loss.backward()
        self.opt_policy.step()

        # Update value function
        self.opt_value.zero_grad()
        critic_loss.backward()
        self.opt_value.step()

        actor_loss_value = float(actor_loss.detach().cpu().item())
        critic_loss_value = float(critic_loss.detach().cpu().item())

        # Clear episode storage
        self.states.clear()
        self.log_probs.clear()
        self.rewards.clear()

        return actor_loss_value, critic_loss_value
