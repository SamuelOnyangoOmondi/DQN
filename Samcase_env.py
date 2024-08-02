import gym
from gym import spaces
import numpy as np

class HospitalEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, size=5, render_mode=None):
        super(HospitalEnv, self).__init__()
        self.size = size
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Dict({
            "position": spaces.Discrete(size*size),
            "goal": spaces.Discrete(size*size)
        })
        self.state = None
        self.goal = 24  # Goal is fixed at the bottom-right corner (medicine cabinet)
        self.render_mode = render_mode

    def reset(self):
        self.state = 0  # Start at top-left corner
        return self._get_obs(), {}

    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"
        x, y = self.state % self.size, self.state // self.size
        if action == 0:    y = max(0, y - 1)  # up
        elif action == 1:  y = min(self.size - 1, y + 1)  # down
        elif action == 2:  x = max(0, x - 1)  # left
        elif action == 3:  x = min(self.size - 1, x + 1)  # right
        self.state = y * self.size + x
        done = self.state == self.goal
        reward = 1 if done else -1/self.size
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return {'position': self.state, 'goal': self.goal}

    def render(self, mode='human'):
        grid = np.zeros((self.size, self.size), dtype=str)
        grid.fill(' ')
        sx, sy = self.state % self.size, self.state // self.size
        gx, gy = self.goal % self.size, self.goal // self.size
        grid[sy][sx] = 'A'  # Agent
        grid[gy][gx] = 'G'  # Goal
        print("\n".join(" ".join(row) for row in grid))

    def close(self):
        pass
