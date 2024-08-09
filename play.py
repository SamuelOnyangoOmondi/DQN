from stable_baselines3 import PPO
import os
import tensorflow as tf
from plastech_env import PlasTechEnv

def simulate_and_evaluate_production():
    model_path = "models/ppo_plastech"
    if not os.path.exists(model_path + '.zip'):
        raise FileNotFoundError(f"No model file found at {model_path}.zip. Please train the model first.")

    env = PlasTechEnv()
    model = PPO.load(model_path)
    obs = env.reset()

    all_labels = []  # Collect true labels
    all_logits = []  # Collect model predictions

    for _ in range(200):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        
        # Collect data for loss calculation (example placeholders, adjust as needed)
        all_labels.append(info.get('true_label'))
        all_logits.append(action)  # If 'action' is not directly comparable to labels, adjust accordingly

        if dones:
            obs = env.reset()

    env.close()
    
    # Convert collected data to suitable format and calculate loss
    labels = tf.constant(all_labels)
    logits = tf.constant(all_logits)
    loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    print("Loss:", loss)

if __name__ == "__main__":
    simulate_and_evaluate_production()
