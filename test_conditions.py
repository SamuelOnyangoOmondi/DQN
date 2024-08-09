from plastech_env import PlasTechEnv

def test_initial_conditions():
    """
    Tests the environment with specific initial conditions.
    """
    env = PlasTechEnv()
    test_positions = [[0, 0], [1, 0], [0, 1], [1, 1]]
    for position in test_positions:
        print(f"Testing from position {position}")
        env.reset(initial_position=position)
        env.render()

if __name__ == "__main__":
    test_initial_conditions()
