import gym
from gym import spaces
import numpy as np

class PlasTechEnv(gym.Env):
    """
    Custom Environment for Plas-tech simulation, following gym interface.
    This is a simple grid environment where the agent must collect plastic waste and process it.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(PlasTechEnv, self).__init__()
        self.grid_size = 6
        self.action_space = spaces.Discrete(5)  # 0: up, 1: down, 2: left, 3: right, 4: collect
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.grid_size, self.grid_size, 3), dtype=np.uint8)

        # Initialize grid and agent location
        self.reset()

    def reset(self):
        self.state = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        # Randomly place plastic waste
        num_waste = np.random.randint(1, 5)  # Randomly 1 to 4 pieces of plastic
        for _ in range(num_waste):
            x, y = np.random.randint(0, self.grid_size, size=2)
            self.state[x, y, 0] = 1  # Channel 0 for plastic waste

        # Set one processing area
        self.state[5, 5, 1] = 1  # Bottom-right corner as processing area, channel 1

        # Initialize agent position
        self.agent_pos = [0, 0]  # Start at top-left corner
        self.state[self.agent_pos[0], self.agent_pos[1], 2] = 1  # Channel 2 for agent position
        return self.state

    def step(self, action):
        if action == 0 and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.grid_size - 1:
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.grid_size - 1:
            self.agent_pos[1] += 1
        elif action == 4:
            # Collect plastic if present
            pass  # Implementation of collection logic here

        # Update state with new agent position
        # Reset previous position
        self.state[:, :, 2] = 0
        self.state[self.agent_pos[0], self.agent_pos[1], 2] = 1
        
        # Reward logic
        reward = 0
        done = False
        info = {}
        return self.state, reward, done, info

    def render(self, mode='human', close=False):
        # Rendering logic for human viewing
        pass

    def close(self):
        pass

