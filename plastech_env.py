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
        self.action_space = spaces.Discrete(4)  # 0: up, 1: down, 2: left, 3: right
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size, 3), dtype=np.uint8)

        # Define positions in a structured way
        self.agent_position = [0, 0]
        self.goal_position = [self.grid_size - 1, self.grid_size - 1]
        self.obstacles = self._generate_obstacles()
        self.state = None
        self.visited_states = set()  # To track visited states for exploration reward

    def _generate_obstacles(self):
        # Randomly place obstacles, avoiding the agent's start and goal positions
        obstacles = []
        for _ in range(int(self.grid_size * self.grid_size * 0.2)):  # 20% of the grid
            obs = np.random.randint(0, self.grid_size, size=2).tolist()
            if obs != self.agent_position and obs != self.goal_position:
                obstacles.append(obs)
        return obstacles

    def reset(self):
        self.agent_position = [0, 0]  # Reset agent to the starting position
        self.state = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        self.visited_states.clear()
        self.visited_states.add(tuple(self.agent_position))
        return self._get_obs()

    def step(self, action):
        # Apply the selected action and update the environment's state
        old_position = self.agent_position.copy()
        if action == 0:  # up
            self.agent_position[0] = max(0, self.agent_position[0] - 1)
        elif action == 1:  # down
            self.agent_position[0] = min(self.grid_size - 1, self.agent_position[0] + 1)
        elif action == 2:  # left
            self.agent_position[1] = max(0, self.agent_position[1] - 1)
        elif action == 3:  # right
            self.agent_position[1] = min(self.grid_size - 1, self.agent_position[1] + 1)

        reward = -0.1  # Default penalty for each move
        done = False

        if self.agent_position == self.goal_position:
            reward += 10  # Increased reward for reaching the goal
            done = True
        elif self.agent_position in self.obstacles:
            reward -= 5  # Penalty for hitting an obstacle
            done = True
        elif tuple(self.agent_position) not in self.visited_states:
            reward += 0.5  # Reward for exploring new state
            self.visited_states.add(tuple(self.agent_position))

        if self.agent_position == old_position:
            reward -= 1  # Additional penalty for no movement

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        # Update state based on current agent position
        self.state.fill(0)
        self.state[self.agent_position[0], self.agent_position[1], 0] = 1  # Agent's position
        self.state[self.goal_position[0], self.goal_position[1], 1] = 1  # Goal position
        for obs in self.obstacles:
            self.state[obs[0], obs[1], 2] = 1  # Obstacles
        return self.state

    def render(self, mode='human', close=False):
        # Visual representation of the environment
        if close:
            plt.close()
        else:
            plt.figure(figsize=(5, 5))
            img = np.zeros((self.grid_size, self.grid_size))
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    if self.state[row, col, 0] == 1:
                        img[row, col] = 1  # Agent
                    elif self.state[row, col, 1] == 1:
                        img[row, col] = 0.5  # Goal
                    elif self.state[row, col, 2] == 1:
                        img[row, col] = 0.8  # Obstacle
            plt.imshow(img, cmap='hot')
            plt.grid('on', which='both')
            tick_labels = range(self.grid_size)
            plt.xticks(tick_labels)
            plt.yticks(tick_labels)
            plt.title('PlasTech Environment')
            plt.show()

    def close(self):
        pass
