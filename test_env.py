import gym
from plastech_env import PlasTechEnv

def test_environment():
    """
    Function to test the PlasTech environment by initializing it, resetting it, and taking random actions
    to see the responses and rewards.
    """
    env = PlasTechEnv()
    env.reset()

    print("Testing environment initialization...")
    initial_state = env.reset()
    print("Initial state:")
    env.render()

    # Perform random actions and print the outcome
    for _ in range(10):
        action = env.action_space.sample()  # Randomly sample an action
        next_state, reward, done, info = env.step(action)
        print(f"Action taken: {action} -> Reward: {reward}, Done: {done}")
        env.render()  # Visualize the state after the action
        if done:
            print("Resetting environment...")
            env.reset()

    # Close the environment
    env.close()
    print("Environment closed successfully.")

if __name__ == "__main__":
    test_environment()
