import numpy as np
from src.envs import make_env
from src.utils import run_episodes
from src.agents.random_agent import RandomAgent
from src.logging import save_run

def main():
    env = make_env("CartPole-v1", seed=0)
    agent = RandomAgent(env.action_space)

    rewards = run_episodes(env, agent, num_episodes=200)

    run_data = {
        "env_id": "CartPole-v1",
        "agent": "random",
        "num_episodes": int(len(rewards)),
        "reward_mean_last_20": float(np.mean(rewards[-20:])),
        "rewards": rewards.tolist(),
    }

    out_path = save_run(run_data, prefix="random")
    print(f"Saved run to {out_path}")
    print(f"Mean reward (last 20): {run_data['reward_mean_last_20']:.2f}")

if __name__ == "__main__":
    main()
