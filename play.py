import os
import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from plastech_env import PlasTechEnv

def make_env():
    """Utility function for multiprocessed env."""
    env = PlasTechEnv()
    env = DummyVecEnv([lambda: env])
    return env

def play_agent():
    env = make_env()
    model_path = os.path.join("models", "dqn_plastech")
    model = DQN.load(model_path, env=env)

    num_episodes = 10  # Set the desired number of episodes
    for episode in range(num_episodes):
        obs = env.reset()
        total_rewards = 0
        done = False
        step = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_rewards += reward[0]
            print(f"Action: {action}, Reward: {reward}, Total Rewards: {total_rewards}")
            step += 1
            if step >= 100:  # Prevents infinite loop in case of missing done signal
                break
        print(f"Episode {episode + 1} completed in {step} steps. Total Rewards = {total_rewards}")

    env.close()

if __name__ == "__main__":
    play_agent()
