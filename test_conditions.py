from plastech_env import PlasTechEnv

def test_initial_conditions():
    """Test environment with manual initial conditions."""
    env = PlasTechEnv()
    test_positions = [[0, 0], [1, 1], [2, 2]]

    for position in test_positions:
        env.reset()
        env.agent_position = position  # Manually set the initial agent position
        print(f"Testing from position {position}")
        state = env._get_obs()  # Get the observation after setting the position
        env.render()  # Display the grid

        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}")

    env.close()

if __name__ == "__main__":
    test_initial_conditions()
