import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from plastech_env import PlasTechEnv  # Import the custom environment we created

def train_agent():
    # Create the environment
    env = PlasTechEnv()
    env = DummyVecEnv([lambda: env])  # Vectorized environments for stability-baselines3

    # Define the model
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the model
    model.learn(total_timesteps=20000)

    # Save the model
    model.save("ppo_plastech")

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"Evaluation: mean_reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    return model

if __name__ == "__main__":
    trained_model = train_agent()
