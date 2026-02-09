import gymnasium as gym

def make_env(env_id="CartPole-v1", seed=0):
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env
