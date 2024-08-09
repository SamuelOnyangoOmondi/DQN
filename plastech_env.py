import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt

class PlasTechEnv(gym.Env):
    """
    PlasTech Environment simulating the collection of plastic waste.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(PlasTechEnv, self).__init__()
        self.grid_size = 6  # Size of the grid
        self.action_space = spaces.Discrete(4)  # Four possible actions: up, down, left, right
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size, 3), dtype=np.uint8)

        self.goal_position = [self.grid_size - 1, self.grid_size - 1]  # Position of the goal
        self.obstacles = self._generate_obstacles()  # Generate obstacles within the grid
        self.agent_position = [0, 0]  # Starting position of the agent
        self.state = None
        self.figure, self.ax = plt.subplots()

    def _generate_obstacles(self):
        # Generates obstacles at random positions within the grid, excluding the start and goal positions
        obstacles = []
        while len(obstacles) < self.grid_size:  # Ensuring a fixed number of obstacles
            obs = np.random.randint(0, self.grid_size, size=2).tolist()
            if obs != self.agent_position and obs != self.goal_position and obs not in obstacles:
                obstacles.append(obs)
        return obstacles

    def reset(self, initial_position=None):
        # Resets the environment to start a new episode
        if initial_position:
            self.agent_position = initial_position
        else:
            self.agent_position = [0, 0]  # Reset to the default starting position
        self.state = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        return self._update_state()

    def step(self, action):
        # Updates the environment according to the action taken by the agent
        move = [[-1, 0], [1, 0], [0, -1], [0, 1]][action]
        new_position = [self.agent_position[0] + move[0], self.agent_position[1] + move[1]]
        if 0 <= new_position[0] < self.grid_size and 0 <= new_position[1] < self.grid_size:
            self.agent_position = new_position

        reward = -0.1  # Default step cost
        done = False

        if self.agent_position == self.goal_position:
            reward += 10  # Reward for reaching the goal
            done = True
        elif self.agent_position in self.obstacles:
            reward -= 5  # Penalty for hitting an obstacle
            done = True

        return self._update_state(), reward, done, {}

    def _update_state(self):
        # Updates the state representation of the environment
        self.state.fill(0)
        self.state[self.agent_position[0], self.agent_position[1]] = [1, 0, 0]  # Agent in red
        self.state[self.goal_position[0], self.goal_position[1]] = [0, 1, 0]  # Goal in green
        for obs in self.obstacles:
            self.state[obs[0], obs[1]] = [0, 0, 1]  # Obstacles in blue
        return self.state

    def render(self, mode='human'):
        # Visualization of the environment
        if mode == 'human':
            plt.imshow(self.state)
            plt.title("PlasTech Environment")
            plt.show()

    def close(self):
        plt.close(self.figure)
