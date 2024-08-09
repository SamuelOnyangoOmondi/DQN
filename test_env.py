import gym
from plastech_env import PlasTechEnv

def test_environment():
    """Create an instance of the environment and test its functionality."""
    env = PlasTechEnv()
    state = env.reset()
    print("Testing environment initialization...")
    print("Initial state:", state)

    for _ in range(10):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        print(f"Action: {action}, Next state: {next_state}, Reward: {reward}, Done: {done}")
        if done:
            state = env.reset()
            print("Environment reset after done.")
            print("State after reset:", state)

    env.close()
    print("Environment closed successfully.")

if __name__ == "__main__":
    test_environment()
