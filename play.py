import gym
from stable_baselines3 import PPO
from plastech_env import PlasTechEnv

def simulate():
    # Load the environment
    env = PlasTechEnv()
    
    # Load the trained model
    model = PPO.load("ppo_plastech")

    # Run a simulation
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            break

    env.close()

if __name__ == "__main__":
    simulate()
