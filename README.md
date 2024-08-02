---

# Deep Q-Network (DQN) for Hospital Environment Navigation

This project implements a Deep Q-Network (DQN), a type of deep reinforcement learning model, to navigate a simulated hospital environment. The agent learns to move from a starting point to a goal within a grid, avoiding potential obstacles and optimizing its path based on rewards.

## Features

- **Custom Environment**: Utilizes a gym-like environment tailored for simulating a hospital navigation task.
- **DQN Implementation**: Uses a neural network to approximate Q-values with state input being the agent's position on the grid.
- **Training and Evaluation**: Includes scripts for training the DQN agent and evaluating its performance in the environment.

## Environment Setup

The environment, defined in `Samcase_env.py`, simulates a hospital with a grid where the agent must navigate to a specific point. The agent receives a positive reward for reaching the goal and a negative reward for taking longer paths.

## Installation

Follow these steps to set up the project environment and run the DQN model:

1. **Clone the Repository**:
   ```bash
   git clone (https://github.com/SamuelOnyangoOmondi/DQN.git)
   cd DQN
   ```

2. **Set Up a Virtual Environment** (Optional but recommended):
   ```bash
   python -m venv venv
   # Activate virtual environment
   # On Windows
   venv\Scripts\activate
   # On Unix or MacOS
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Training Script**:
   ```bash
   python train.py
   ```

## Usage

- **Training the Agent**:
  Use the `train.py` script to train the agent. You can adjust hyperparameters within the script such as learning rate, number of episodes, etc.

- **Evaluating the Agent**:
  After training, you can evaluate the agent's performance using the `play.py` script, which demonstrates the agent navigating through the environment.

## Files and Directories

- `Samcase_env.py`: Defines the custom environment.
- `train.py`: Script for training the DQN agent.
- `play.py`: Script for demonstrating the trained agent.
- `requirements.txt`: Contains all necessary Python packages.

## Contributing

Feel free to fork the repository and submit pull requests. You can also open issues for bugs or feature requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contact

For any queries, you can reach out to [Samuel Omondi](s.omondi@alustudent.com).

---
