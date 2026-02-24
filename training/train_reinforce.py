import argparse
import numpy as np
import torch
import torch.nn as nn

from src.envs import make_env
from src.logging import save_run
from src.agents.reinforce import ReinforceAgent


def seed_everything(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mean_last(rewards, n: int) -> float:
    if len(rewards) == 0:
        return float("nan")
    n = min(n, len(rewards))
    return float(np.mean(rewards[-n:]))


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    args = parser.parse_args()

    seed_everything(args.seed)

    env = make_env(args.env_id, seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = PolicyNet(obs_dim, n_actions, hidden=args.hidden)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    agent = ReinforceAgent(gamma=args.gamma, policy=policy, optimizer=optimizer)

    rewards = []
    losses = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_reward = 0.0

        for _ in range(args.max_steps):
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.observe(obs, action, reward, next_obs, done)

            ep_reward += reward
            obs = next_obs

            if done:
                break

        loss = agent.end_episode()
        rewards.append(float(ep_reward))
        losses.append(float(loss))

        if (ep + 1) % 25 == 0:
            print(f"Episode {ep+1:4d} | mean(last 25)={np.mean(rewards[-25:]):7.2f} | last={ep_reward:6.1f}")

    run_data = {
        "env_id": args.env_id,
        "agent": "reinforce",
        "seed": int(args.seed),
        "num_episodes": int(args.episodes),
        "hyperparameters": {
            "gamma": args.gamma,
            "lr": args.lr,
            "hidden": args.hidden,
            "max_steps": args.max_steps,
        },
        "reward_mean_last_100": mean_last(rewards, 100),
        "rewards": rewards,
        "losses": losses,
    }

    out_path = save_run(run_data, prefix="reinforce")
    print(f"\nSaved run to {out_path}")
    print(f"Mean reward (last 100): {run_data['reward_mean_last_100']:.2f}")


if __name__ == "__main__":
    main()