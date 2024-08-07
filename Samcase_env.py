import numpy as np
import gym
from gym import spaces

class HospitalEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, size=5):
        super(HospitalEnv, self).__init__()
        self.size = size
        self.action_space = spaces.Discrete(4)  # Up, down, left, right
        self.observation_space = spaces.Box(low=0, high=size - 1, shape=(2,), dtype=np.int32)
        self.state = np.array([0, 0], dtype=np.int32)  # Starting position
        self.goal = np.array([size - 1, size - 1], dtype=np.int32)  # Goal position

    def reset(self):
        self.state = np.array([0, 0], dtype=np.int32)
        return self.state.copy()

    def step(self, action):
        movements = [np.array([0, -1]), np.array([0, 1]), np.array([-1, 0]), np.array([1, 0])]
        self.state = np.clip(self.state + movements[action], 0, self.size - 1)
        done = np.array_equal(self.state, self.goal)
        reward = 1 if done else -0.1
        return self.state.copy(), reward, done, {}

    def render(self, mode='human'):
        grid = np.full((self.size, self.size), '.', dtype=str)
        grid[tuple(self.state)] = 'A'
        grid[tuple(self.goal)] = 'G'
        for row in grid:
            print(' '.join(row))
