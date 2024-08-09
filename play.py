from stable_baselines3 import PPO
import os
import tensorflow as tf
from plastech_env import PlasTechEnv

loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

def simulate_production():
    model_path = "models/ppo_plastech"

    if not os.path.exists(model_path + '.zip'):
        raise FileNotFoundError(f"No model file found at {model_path}.zip. Please train the model first.")

    # Load the environment
    env = PlasTechEnv()
    
    # Load the trained model
    model = PPO.load(model_path)

    obs = env.reset()
    for _ in range(200):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        if dones:
            obs = env.reset()

    env.close()

if __name__ == "__main__":
    simulate_production()
