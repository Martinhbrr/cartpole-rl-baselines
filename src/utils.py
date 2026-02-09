import numpy as np

def run_episodes(env, agent, num_episodes=200, max_steps=500):
    rewards = []

    for _ in range(num_episodes):
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

        rewards.append(ep_reward)

    return np.array(rewards, dtype=np.float32)
