import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt

class PlasTechEnv(gym.Env):
    """
    Custom environment simulating plastic waste collection in a grid setup. The agent must navigate through
    obstacles to reach a goal position where it processes the waste, optimizing the path for efficiency.
    This environment includes visualization capabilities to display the state of the grid, the agent's position,
    obstacles, and the goal.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(PlasTechEnv, self).__init__()
        self.grid_size = 6  # Define grid size
        self.action_space = spaces.Discrete(4)  # Actions: 0: up, 1: down, 2: left, 3: right
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.grid_size, self.grid_size, 3), dtype=np.uint8)

        # Initialize agent, goal, and obstacles positions
        self.agent_position = [0, 0]
        self.goal_position = [self.grid_size - 1, self.grid_size - 1]
        self.obstacles = self._generate_obstacles()
        self.state = None
        self.visited_states = set()  # Track visited states for exploration reward

    def _generate_obstacles(self):
        """
        Generates obstacles on the grid, avoiding the initial and goal positions.
        """
        obstacles = []
        for _ in range(int(self.grid_size * self.grid_size * 0.2)):  # Approximately 20% of the grid are obstacles
            obs = np.random.randint(0, self.grid_size, size=2).tolist()
            if obs != self.agent_position and obs != self.goal_position and obs not in obstacles:
                obstacles.append(obs)
        return obstacles

    def reset(self):
        """
        Resets the environment to the initial state.
        """
        self.agent_position = [0, 0]
        self.state = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        self.visited_states.clear()
        self.visited_states.add(tuple(self.agent_position))
        return self._get_obs()

    def step(self, action):
        """
        Apply the selected action and update the environment's state, calculating the reward and checking if the goal is reached.
        """
        old_position = self.agent_position.copy()
        if action == 0 and self.agent_position[0] > 0:
            self.agent_position[0] -= 1
        elif action == 1 and self.agent_position[0] < self.grid_size - 1:
            self.agent_position[0] += 1
        elif action == 2 and self.agent_position[1] > 0:
            self.agent_position[1] -= 1
        elif action == 3 and self.agent_position[1] < self.grid_size - 1:
            self.agent_position[1] += 1

        reward = -0.1  # Default penalty for each move
        done = False

        if self.agent_position == self.goal_position:
            reward += 10  # Large reward for reaching the goal
            done = True
        elif self.agent_position in self.obstacles:
            reward -= 5  # Large penalty for hitting an obstacle
            done = True
        elif tuple(self.agent_position) not in self.visited_states:
            reward += 0.5  # Reward for exploring a new state
            self.visited_states.add(tuple(self.agent_position))

        if self.agent_position == old_position:
            reward -= 1  # Penalty for no movement

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        """
        Returns the current state of the environment, marking the agent, goal, and obstacles.
        """
        self.state.fill(0)
        self.state[self.agent_position[0], self.agent_position[1], 0] = 1  # Agent's position
        self.state[self.goal_position[0], self.goal_position[1], 1] = 1  # Goal position
        for obs in self.obstacles:
            self.state[obs[0], obs[1], 2] = 1  # Obstacles
        return self.state

    def render(self, mode='human', close=False):
        """
        Renders the current state of the environment visually using matplotlib.
        """
        if close:
            plt.close()
        else:
            plt.figure(figsize=(5, 5))
            img = np.zeros((self.grid_size, self.grid_size))
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    if self.state[row, col, 0] == 1:
                        img[row, col] = 1  # Agent is marked with 1
                    elif self.state[row, col, 1] == 1:
                        img[row, col] = 0.5  # Goal is marked with 0.5
                    elif self.state[row, col, 2] == 1:
                        img[row, col] = 0.8  # Obstacles are marked with 0.8
            plt.imshow(img, cmap='hot')
            plt.grid(True)
            plt.xticks(range(self.grid_size))
            plt.yticks(range(self.grid_size))
            plt.title('PlasTech Environment Visualization')
            plt.show()

    def close(self):
        """
        Optional: Close any open renderings.
        """
        plt.close()
