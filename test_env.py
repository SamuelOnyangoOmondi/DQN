import gym
from plastech_env import PlasTechEnv

def test_environment():
    # Create an instance of the environment
    env = PlasTechEnv()
    env.reset()

    print("Testing environment initialization...")
    # Test initial state
    initial_state = env.reset()
    print("Initial state:", initial_state)

    # Perform random actions and print the outcome
    for _ in range(10):
        action = env.action_space.sample()  # Randomly sample an action
        next_state, reward, done, info = env.step(action)
        print(f"Action taken: {action} -> Next state: {next_state}, Reward: {reward}, Done: {done}")

    # Close the environment
    env.close()
    print("Environment closed successfully.")

if __name__ == "__main__":
    test_environment()
