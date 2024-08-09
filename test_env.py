import gym
from plastech_env import PlasTechEnv

def test_environment():
    env = PlasTechEnv()
    env.reset()

    print("Testing environment initialization...")
    initial_state = env.reset()
    print("Initial state:", initial_state)

    for _ in range(10):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        print(f"Action taken: {action} -> Next state: {next_state}, Reward: {reward}, Done: {done}")

    env.close()
    print("Environment closed successfully.")

if __name__ == "__main__":
    test_environment()
