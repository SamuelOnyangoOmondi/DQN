import gym
from gym import spaces
import numpy as np

class HospitalEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, size=5, render_mode=None):
        super(HospitalEnv, self).__init__()
        self.size = size
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        # Using a Box instead of Dict for direct neural network compatibility
        self.observation_space = spaces.Box(low=0, high=size - 1, shape=(2,), dtype=np.int32)
        self.state = np.array([0, 0], dtype=np.int32)  # Starting position at top-left corner
        self.goal = np.array([size - 1, size - 1], dtype=np.int32)  # Goal at bottom-right corner
        self.render_mode = render_mode

    def reset(self):
        self.state = np.array([0, 0], dtype=np.int32)  # Reset state to the starting position
        return self._get_obs(), {}

    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"
        movements = [np.array([0, 1]), np.array([1, 0]), np.array([0, -1]), np.array([-1, 0])]
        self.state = np.clip(self.state + movements[action], 0, self.size - 1)
        done = np.array_equal(self.state, self.goal)
        reward = 1 if done else -1 / self.size
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return self.state.copy()  # Returning a copy of the current state

    def render(self, mode='human'):
        grid = np.zeros((self.size, self.size), dtype=str)
        grid.fill(' ')
        sx, sy = self.state
        gx, gy = self.goal
        grid[sy][sx] = 'A'  # Agent
        grid[gy][gx] = 'G'  # Goal
        print("\n".join(" ".join(row) for row in grid))

    def close(self):
        pass
