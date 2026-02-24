import argparse
import numpy as np

from src.envs import make_env
from src.utils import run_episodes
from src.agents.random_agent import RandomAgent
from src.logging import save_run


def seed_everything(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def mean_last(rewards: np.ndarray, n: int) -> float:
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
    args = parser.parse_args()

    seed_everything(args.seed)

    env = make_env(args.env_id, seed=args.seed)
    agent = RandomAgent(env.action_space)

    rewards = run_episodes(env, agent, num_episodes=args.episodes, max_steps=args.max_steps)

    run_data = {
        "env_id": args.env_id,
        "agent": "random",
        "seed": int(args.seed),
        "num_episodes": int(len(rewards)),
        "hyperparameters": {
            "max_steps": int(args.max_steps),
        },
        "reward_mean_last_100": mean_last(rewards, 100),
        "reward_mean_last_20": mean_last(rewards, 20),
        "rewards": rewards.astype(np.float32).tolist(),
    }

    out_path = save_run(run_data, prefix="random")
    print(f"Saved run to {out_path}")
    print(f"Mean reward (last 100): {run_data['reward_mean_last_100']:.2f}")


if __name__ == "__main__":
    main()