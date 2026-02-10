import numpy as np
import torch
import torch.nn as nn
from src.envs import make_env
from src.logging import save_run
from src.agents.reinforce import ReinforceAgent


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
    def forward(self, x):
        return self.net(x)  # logits

def main():
    env_id = "CartPole-v1"
    seed = 0
    num_episodes = 800
    max_steps = 500
    gamma = 0.99
    lr = 1e-3

    env = make_env(env_id, seed=seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy = PolicyNet(obs_dim, n_actions, hidden=128)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    agent = ReinforceAgent(gamma=gamma, policy=policy, optimizer=optimizer)
    rewards = []
    losses = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0

        for _ in range(max_steps):
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.observe(obs, action, reward, next_obs, done)
            ep_reward += reward
            obs = next_obs
            if done:
                break

        loss = agent.end_episode()
        rewards.append(ep_reward)
        losses.append(loss)

        if (ep + 1) % 25 == 0:
            print(f"Episode {ep+1:4d} | mean(last 25)={np.mean(rewards[-25:]):7.2f} | last={ep_reward:6.1f}")

    run_data = {
        "env_id": env_id,
        "agent": "reinforce",
        "seed": seed,
        "num_episodes": int(num_episodes),
        "hyperparameters": {
            "gamma": gamma,
            "lr": lr,
            "hidden": 128,
            "max_steps": max_steps,
        },
        "reward_mean_last_100": float(np.mean(rewards[-100:])),
        "rewards": [float(r) for r in rewards],
        "losses": [float(l) for l in losses],
    }

    out_path = save_run(run_data, prefix="reinforce")
    print(f"\nSaved run to {out_path}")
    print(f"Mean reward (last 100): {run_data['reward_mean_last_100']:.2f}")


if __name__ == "__main__":
    main()
