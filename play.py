from plastech_env import PlasTechEnv
from stable_baselines3 import DQN

def play():
    """Load the model and run the environment interactively with the trained agent."""
    env = PlasTechEnv()
    model = DQN.load("models/dqn_plastech")

    obs = env.reset()
    for _ in range(200):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()

if __name__ == "__main__":
    play()
