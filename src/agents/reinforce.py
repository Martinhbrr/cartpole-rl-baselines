"""
agent that uses the Monte-Carlo Policy-Gradient control to learn an optimal policy directly without
 going through values
"""
import torch
from torch.distributions import Categorical


class ReinforceAgent:
    """
    REINFORCE (Monte Carlo policy gradient).
    Stores (log_prob_t, reward_t) for one episode, then updates the policy at episode end.
    """

    def __init__(self, gamma, policy, optimizer, device="cpu", normalize_returns=True):
        self.gamma = float(gamma)
        self.policy = policy
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.normalize_returns = normalize_returns

        self.rewards = []
        self.log_probs = []

        # Ensure policy is on the right device
        self.policy.to(self.device)

    def act(self, obs):
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, obs_dim)

        logits = self.policy(x)                # (1, n_actions)
        dist = Categorical(logits=logits)      # use logits directly
        action_t = dist.sample()               # tensor([0]) or tensor([1])
        log_prob = dist.log_prob(action_t)     # tensor([..])

        self.log_probs.append(log_prob.squeeze(0))  # store shape ()/(scalar)
        return int(action_t.item())

    def observe(self, s, a, r, s2, done):
        # For REINFORCE we only need reward
        self.rewards.append(float(r))

    def end_episode(self):
        # 1) compute discounted returns
        returns = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.append(G)
        returns.reverse()

        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # 2) normalize returns
        if self.normalize_returns and len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std(unbiased=False) + 1e-8)

        # 3) compute loss
        log_probs_t = torch.stack(self.log_probs)  
        loss = -(log_probs_t * returns_t).sum()

        # 4) gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 5) clear episode buffers
        self.rewards.clear()
        self.log_probs.clear()

        return float(loss.item())


