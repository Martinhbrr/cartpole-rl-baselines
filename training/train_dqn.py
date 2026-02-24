import argparse
import numpy as np
import torch

from src.envs import make_env
from src.logging import save_run
from src.agents.dqn import DQNAgent


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=500)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--buffer_size", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon_decay_steps", type=int, default=20_000)
    parser.add_argument("--target_update_freq", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    seed_everything(args.seed)

    env = make_env(args.env_id, seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        gamma=args.gamma,
        lr=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        target_update_freq=args.target_update_freq,
        device=args.device,
    )

    rewards = []
    epsilons = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_reward = 0.0

        for _ in range(args.max_steps):
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.observe(obs, action, reward, next_obs, done)
            agent.update()
            agent.update_epsilon()

            ep_reward += reward
            obs = next_obs

            if done:
                break

        rewards.append(float(ep_reward))
        epsilons.append(float(agent.epsilon))

        if (ep + 1) % 25 == 0:
            print(
                f"Episode {ep+1:4d} | mean(last 25)={np.mean(rewards[-25:]):7.2f} "
                f"| last={ep_reward:6.1f} | eps={agent.epsilon:6.3f} | steps={agent.total_steps}"
            )

    run_data = {
        "env_id": args.env_id,
        "agent": "dqn",
        "seed": int(args.seed),
        "num_episodes": int(args.episodes),
        "hyperparameters": {
            "gamma": args.gamma,
            "lr": args.lr,
            "buffer_size": args.buffer_size,
            "batch_size": args.batch_size,
            "epsilon_start": args.epsilon_start,
            "epsilon_end": args.epsilon_end,
            "epsilon_decay_steps": args.epsilon_decay_steps,
            "target_update_freq": args.target_update_freq,
            "max_steps": args.max_steps,
            "device": args.device,
        },
        "reward_mean_last_100": mean_last(rewards, 100),
        "rewards": rewards,
        "epsilons": epsilons,
        "total_steps": int(agent.total_steps),
    }

    out_path = save_run(run_data, prefix="dqn")
    print(f"\nSaved run to {out_path}")
    print(f"Mean reward (last 100): {run_data['reward_mean_last_100']:.2f}")


if __name__ == "__main__":
    main()