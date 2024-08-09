import numpy as np
import gym
from gym import spaces
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PlasTechEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        super(PlasTechEnv, self).__init__()
        self.grid_size = 6
        self.action_space = spaces.Discrete(4)  # 0: up, 1: down, 2: left, 3: right
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size, 3), dtype=np.uint8)
        self.agent_position = [0, 0]
        self.goal_position = [self.grid_size - 1, self.grid_size - 1]
        self.obstacles = self._generate_obstacles()
        self.state = None
        self.visited_states = set()

    def _generate_obstacles(self):
        obstacles = []
        for _ in range(int(self.grid_size * self.grid_size * 0.2)):
            obs = np.random.randint(0, self.grid_size, size=2).tolist()
            if obs != self.agent_position and obs != self.goal_position:
                obstacles.append(obs)
        return obstacles

    def reset(self):
        self.agent_position = [0, 0]
        self.state = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        self.visited_states.clear()
        self.visited_states.add(tuple(self.agent_position))
        return self._get_obs()

    def step(self, action):
        old_position = self.agent_position.copy()
        # Movement actions
        if action == 0 and self.agent_position[0] > 0:
            self.agent_position[0] -= 1
        elif action == 1 and self.agent_position[0] < self.grid_size - 1:
            self.agent_position[0] += 1
        elif action == 2 and self.agent_position[1] > 0:
            self.agent_position[1] -= 1
        elif action == 3 and self.agent_position[1] < self.grid_size - 1:
            self.agent_position[1] += 1

        reward = -0.1
        done = False
        if self.agent_position == self.goal_position:
            reward += 10
            done = True
            logging.info("Goal reached! Reward: +10")
        elif self.agent_position in self.obstacles:
            reward -= 5
            done = True
            logging.info(f"Hit an obstacle at {self.agent_position}! Reward: -5")
        elif tuple(self.agent_position) not in self.visited_states:
            reward += 0.5
            self.visited_states.add(tuple(self.agent_position))
            logging.info("Exploring new state! Reward: +0.5")
        logging.info(f"Action: {action}, Position: {self.agent_position}, Reward: {reward}")
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        self.state.fill(0)
        self.state[self.agent_position[0], self.agent_position[1], 0] = 1
        self.state[self.goal_position[0], self.goal_position[1], 1] = 1
        for obs in self.obstacles:
            self.state[obs[0], obs[1], 2] = 1
        return self.state

    def render(self, mode='human', close=False):
        if close:
            return
        grid = np.zeros((self.grid_size, self.grid_size))
        grid[self.goal_position[0], self.goal_position[1]] = 0.5
        for obs in self.obstacles:
            grid[obs[0], obs[1]] = 1
        grid[self.agent_position[0], self.agent_position[1]] = 0.3
        plt.imshow(grid, cmap='hot', interpolation='nearest')
        plt.title("Environment State")
        plt.show()

    def close(self):
        pass
