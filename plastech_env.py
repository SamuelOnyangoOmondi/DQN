import gym
from gym import spaces
import numpy as np

class PlasTechEnv(gym.Env):
    """
    Custom Environment for Plas-tech mission. This environment simulates a 6x6 grid where the agent, a waste collection vehicle,
    must collect plastic waste and deliver it to any of the recycling facilities without retracing steps unnecessarily.
    """
    metadata = {'render.modes': ['human', 'console']}

    def __init__(self):
        super(PlasTechEnv, self).__init__()
        # Define a 6x6 grid environment
        self.grid_size = 6
        self.agent_start_position = [0, 0]  # Agent starts at the top-left corner
        self.agent_position = self.agent_start_position.copy()

        # Define action space (4 directions)
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Down, 2: Left, 3: Right

        # Define observation space (flattened grid)
        self.observation_space = spaces.Box(low=0, high=3,
                                            shape=(self.grid_size * self.grid_size,), dtype=np.uint8)

        # Initialize the grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.n_recycling_facilities = 2  # Number of recycling facilities
        self._place_objects()

    def _place_objects(self):
        """
        Randomly place plastic waste and recycling facilities on the grid.
        Plastic waste is marked as 1, recycling facilities as 2.
        """
        self.grid.fill(0)  # Clear grid
        self.grid[tuple(self.agent_start_position)] = 3  # Agent's position marked as 3

        # Place plastic waste randomly
        n_plastic = np.random.randint(5, 10)  # Random number of plastic pieces between 5 and 10
        plastic_positions = np.random.choice(self.grid_size * self.grid_size, n_plastic, replace=False)
        self.grid.flat[plastic_positions] = 1

        # Place recycling facilities
        for _ in range(self.n_recycling_facilities):
            while True:
                facility_position = np.random.randint(self.grid_size * self.grid_size)
                if self.grid.flat[facility_position] == 0:  # Ensure no overlap with plastic or other facilities
                    self.grid.flat[facility_position] = 2
                    break

    def step(self, action):
        """
        Apply the action to move the agent and calculate reward and if the episode is done.
        """
        # Move agent based on action
        if action == 0 and self.agent_position[0] > 0:  # Up
            self.agent_position[0] -= 1
        elif action == 1 and self.agent_position[0] < self.grid_size - 1:  # Down
            self.agent_position[0] += 1
        elif action == 2 and self.agent_position[1] > 0:  # Left
            self.agent_position[1] -= 1
        elif action == 3 and self.agent_position[1] < self.grid_size - 1:  # Right
            self.agent_position[1] += 1

        # Check what's at the new position
        current_cell = self.grid[self.agent_position[0], self.agent_position[1]]
        reward = 0
        done = False

        if current_cell == 1:  # Collected plastic
            reward = 10
            self.grid[self.agent_position[0], self.agent_position[1]] = 0  # Remove plastic from grid
        elif current_cell == 2:  # Recycling facility
            reward = 50  # Large reward for delivering to recycling facility
            done = True  # End episode when recycling facility is reached

        # Return observation, reward, done status, and additional info
        return self.grid.flatten(), reward, done, {}

    def reset(self):
        """
        Reset the environment to start a new episode.
        """
        self.agent_position = self.agent_start_position.copy()
        self._place_objects()
        return self.grid.flatten()

    def render(self, mode='human'):
        """
        Render the environment to the console or a human-friendly format.
        """
        if mode == 'console':
            print(self.grid)
        else:
            # For more sophisticated rendering (e.g., using matplotlib), implement here
            pass

# Example instantiation and stepping through the environment
if __name__ == "__main__":
    env = PlasTechEnv()
    env.reset()
    env.render(mode='console')
    for _ in range(10):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
        env.render(mode='console')
        if done:
            break
