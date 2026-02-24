import argparse
import numpy as np
import torch

from src.envs import make_env
from src.logging import save_run
from src.agents.reinforce_baseline import ReinforceBaselineAgent


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
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr_policy", type=float, default=1e-3)
    parser.add_argument("--lr_value", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    seed_everything(args.seed)

    env = make_env(args.env_id, seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = ReinforceBaselineAgent(
        obs_dim=obs_dim,
        act_dim=n_actions,
        hidden=args.hidden,
        gamma=args.gamma,
        lr_policy=args.lr_policy,
        lr_value=args.lr_value,
        device=args.device,
    )

    rewards = []
    actor_losses = []
    critic_losses = []

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

        actor_loss, critic_loss = agent.end_episode()
        rewards.append(float(ep_reward))
        actor_losses.append(float(actor_loss))
        critic_losses.append(float(critic_loss))

        if (ep + 1) % 25 == 0:
            print(
                f"Episode {ep+1:4d} | mean(last 25)={np.mean(rewards[-25:]):7.2f} | last={ep_reward:6.1f} "
                f"| actor_loss={np.mean(actor_losses[-25:]):8.3f} | critic_loss={np.mean(critic_losses[-25:]):8.3f}"
            )

    run_data = {
        "env_id": args.env_id,
        "agent": "reinforce_baseline",
        "seed": int(args.seed),
        "num_episodes": int(args.episodes),
        "hyperparameters": {
            "gamma": args.gamma,
            "hidden": args.hidden,
            "lr_policy": args.lr_policy,
            "lr_value": args.lr_value,
            "max_steps": args.max_steps,
            "device": args.device,
        },
        "reward_mean_last_100": mean_last(rewards, 100),
        "rewards": rewards,
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
    }

    out_path = save_run(run_data, prefix="reinforce_baseline")
    print(f"\nSaved run to {out_path}")
    print(f"Mean reward (last 100): {run_data['reward_mean_last_100']:.2f}")


if __name__ == "__main__":
    main()