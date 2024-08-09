import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from plastech_env import PlasTechEnv

def play():
    """
    Loads the trained DQN model and runs it on the PlasTech environment to evaluate its performance.
    This function also visualizes the state of the environment after each action taken by the agent.
    """
    env = PlasTechEnv()
    model = DQN.load("./models/dqn_plastech")

    obs = env.reset()
    plt.figure(figsize=(5, 5))

    for i in range(100):
        plt.clf()
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        plt.imshow(env.render(mode='rgb_array'))
        plt.title(f"Step: {i + 1}, Action: {action}, Reward: {rewards}")
        plt.pause(0.1)  # pause to update the plot
        if dones:
            obs = env.reset()

    plt.show()
    env.close()

if __name__ == "__main__":
    play()
