import numpy as np

from src.envs import make_env
from src.logging import save_run
from src.agents.reinforce_baseline import ReinforceBaselineAgent


def main():
    env_id = "CartPole-v1"
    seed = 0
    num_episodes = 800
    max_steps = 500

    gamma = 0.99
    hidden = 128
    lr_policy = 1e-3
    lr_value = 1e-3
    device = "cpu"

    env = make_env(env_id, seed=seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = ReinforceBaselineAgent(
        obs_dim=obs_dim,
        act_dim=n_actions,
        hidden=hidden,
        gamma=gamma,
        lr_policy=lr_policy,
        lr_value=lr_value,
        device=device,
    )

    rewards = []
    actor_losses = []
    critic_losses = []

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

        actor_loss, critic_loss = agent.end_episode()
        rewards.append(ep_reward)
        actor_losses.append(float(actor_loss))
        critic_losses.append(float(critic_loss))

        if (ep + 1) % 25 == 0:
            print(
                f"Episode {ep+1:4d} | mean(last 25)={np.mean(rewards[-25:]):7.2f} | last={ep_reward:6.1f} "
                f"| actor_loss={np.mean(actor_losses[-25:]):8.3f} | critic_loss={np.mean(critic_losses[-25:]):8.3f}"
            )

    run_data = {
        "env_id": env_id,
        "agent": "reinforce_baseline",
        "seed": seed,
        "num_episodes": int(num_episodes),
        "hyperparameters": {
            "gamma": gamma,
            "hidden": hidden,
            "lr_policy": lr_policy,
            "lr_value": lr_value,
            "max_steps": max_steps,
            "device": device,
        },
        "reward_mean_last_100": float(np.mean(rewards[-100:])),
        "rewards": [float(r) for r in rewards],
        "actor_losses": [float(x) for x in actor_losses],
        "critic_losses": [float(x) for x in critic_losses],
    }

    out_path = save_run(run_data, prefix="reinforce_baseline")
    print(f"\nSaved run to {out_path}")
    print(f"Mean reward (last 100): {run_data['reward_mean_last_100']:.2f}")


if __name__ == "__main__":
    main()