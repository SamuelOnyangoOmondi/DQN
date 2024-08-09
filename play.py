from plastech_env import PlasTechEnv
from stable_baselines3 import DQN

def play():
    """Load the model and run the environment interactively with the trained agent for a fixed number of episodes."""
    env = PlasTechEnv()
    model = DQN.load("models/dqn_plastech")
    num_episodes = 5  # Define the number of episodes to play

    for episode in range(num_episodes):
        obs = env.reset()
        total_rewards = 0
        for step in range(100):  # Define max steps per episode
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_rewards += reward
            env.render()
            if done:
                print(f"Episode {episode + 1} completed in {step + 1} steps. Total Rewards = {total_rewards}")
                break
        if not done:
            print(f"Episode {episode + 1} reached the step limit. Total Rewards = {total_rewards}")

    env.close()

if __name__ == "__main__":
    play()
