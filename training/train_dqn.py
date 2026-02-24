import numpy as np

from src.envs import make_env
from src.logging import save_run
from src.agents.dqn import DQNAgent 


def main():
    env_id = "CartPole-v1"
    seed = 0
    num_episodes = 800
    max_steps = 500

    gamma = 0.99
    lr = 1e-3
    buffer_size = 50_000
    batch_size = 64
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_steps = 20_000
    target_update_freq = 1000
    device = "cpu"

    env = make_env(env_id, seed=seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        gamma=gamma,
        lr=lr,
        buffer_size=buffer_size,
        batch_size=batch_size,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        target_update_freq=target_update_freq,
        device=device,
    )

    rewards = []
    epsilons = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0

        for _ in range(max_steps):
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

        rewards.append(ep_reward)
        epsilons.append(float(agent.epsilon))

        if (ep + 1) % 25 == 0:
            print(
                f"Episode {ep+1:4d} | mean(last 25)={np.mean(rewards[-25:]):7.2f} "
                f"| last={ep_reward:6.1f} | eps={agent.epsilon:6.3f} | steps={agent.total_steps}"
            )

    run_data = {
        "env_id": env_id,
        "agent": "dqn",
        "seed": seed,
        "num_episodes": int(num_episodes),
        "hyperparameters": {
            "gamma": gamma,
            "lr": lr,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "epsilon_decay_steps": epsilon_decay_steps,
            "target_update_freq": target_update_freq,
            "max_steps": max_steps,
            "device": device,
        },
        "reward_mean_last_100": float(np.mean(rewards[-100:])),
        "rewards": [float(r) for r in rewards],
        "epsilons": epsilons,
        "total_steps": int(agent.total_steps),
    }

    out_path = save_run(run_data, prefix="dqn")
    print(f"\nSaved run to {out_path}")
    print(f"Mean reward (last 100): {run_data['reward_mean_last_100']:.2f}")


if __name__ == "__main__":
    main()