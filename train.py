import os
import gym
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from plastech_env import PlasTechEnv

def make_env():
    """
    Utility function to create and wrap the environment.
    """
    env = PlasTechEnv()
    env = DummyVecEnv([lambda: env])
    return env

def train_agent():
    """
    Function to set up and train the DQN agent with the PlasTech environment.
    """
    env = make_env()

    model = DQN(
        MlpPolicy,
        env,
        verbose=1,
        tensorboard_log="./plastech_dqn_tensorboard/"
    )
    
    print("Starting training...")
    model.learn(total_timesteps=int(1e5))
    print("Training completed.")

    # Save the model
    model_path = "./models/dqn_plastech"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Close the environment
    env.close()
    print("Environment closed successfully.")

if __name__ == "__main__":
    train_agent()
