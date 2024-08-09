import os
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from plastech_env import PlasTechEnv

def make_env():
    """Utility function for creating a multiprocessed environment."""
    env = PlasTechEnv()
    env = DummyVecEnv([lambda: env])  # Use DummyVecEnv to wrap the environment
    return env

def play_model():
    """Load the model and simulate its performance."""
    model_path = os.path.join("models", "dqn_plastech")
    env = make_env()

    # Load the trained model
    model = DQN.load(model_path, env=env)

    # Number of episodes to play
    num_episodes = 10

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_rewards = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            total_rewards += rewards
            env.render()  # Comment this out if the environment does not support rendering
            print(f"Action: {action}, Reward: {rewards}, Total Rewards: {total_rewards}")

            if done:
                print(f"Episode {episode + 1}: Total Rewards = {total_rewards}")
                break

    env.close()

if __name__ == "__main__":
    play_model()
