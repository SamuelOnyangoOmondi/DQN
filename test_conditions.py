from plastech_env import PlasTechEnv

def test_initial_conditions():
    env = PlasTechEnv()
    test_positions = [[0, 0], [0, 1], [1, 0], [2, 2]]  # Varying initial positions

    for position in test_positions:
        print(f"\nTesting from position {position}")
        env.reset(initial_position=position)
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 10:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            print(f"Step {steps}: Action {action}, Reward {reward}, Total Reward {total_reward}")
            env.render()

        print(f"Completed test from position {position} with total reward {total_reward}")

if __name__ == "__main__":
    test_initial_conditions()
