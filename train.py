import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from plastech_env import PlasTechEnv

def train_agent():
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)  # Ensure the directory exists

    # Create the environment
    env = PlasTechEnv()
    env = DummyVecEnv([lambda: env])

    # Define and train the model
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=20000)

    # Save the model
    model_path = os.path.join(model_dir, "ppo_plastech")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    env.close()

if __name__ == "__main__":
    train_agent()
