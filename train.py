import os
import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from plastech_env import PlasTechEnv

def make_env():
    """Utility function for multiprocessed env."""
    env = PlasTechEnv()
    env = DummyVecEnv([lambda: env])
    return env

def train_agent():
    """Setup and train the DQN agent."""
    # Create the environment
    env = make_env()

    # Initialize the model
    model = DQN(
        MlpPolicy,
        env,
        learning_rate=0.0005,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        verbose=1,
        tensorboard_log="./plastech_dqn_tensorboard/"
    )

    # Train the model
    model.learn(total_timesteps=int(1e5))

    # Save the model
    model_path = os.path.join("models", "dqn_plastech")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Close the environment
    env.close()

if __name__ == "__main__":
    train_agent()
