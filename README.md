# Plas-tech: Deep Reinforcement Learning for Plastic Waste Management

This repository showcases a Deep Reinforcement Learning (DRL) project focused on optimizing the collection and conversion of plastic waste into valuable resources like cooking gas. Utilizing a custom environment and a model based on Deep Q-Networks (DQN), this project aims to simulate the efficient management and processing of plastic waste within a grid-like environment.

## Features

- **Custom Environment**: A gym-like environment specifically designed to simulate the collection and processing of plastic waste.
- **DQN Implementation**: Uses a neural network to approximate Q-values, where the state input includes the agent’s position and status on the grid.
- **Visualizations**: Provides visual feedback on the agent's performance and environment status during training and evaluation phases.
- **Comprehensive Testing**: Scripts to test the environment and specific agent conditions, ensuring robust performance.
- **Training and Evaluation**: Scripts for training the DRL agent and evaluating its efficiency in real-time scenarios.

## Project Structure

- `plastech_env.py`: Defines the custom environment for managing plastic waste.
- `train.py`: Script to train the DQN agent.
- `play.py`: Script to demonstrate the trained agent's performance in the environment.
- `test_env.py`: Script to test the environment's configuration and ensure it's set up correctly.
- `test_conditions.py`: Script to test specific conditions and agent behaviors in the environment.
- `requirements.txt`: Lists dependencies required for the project.
- `models/`: Directory to store the trained models for future use or reference.

## Environment Setup

The `PlasTechEnv` environment simulates a grid where an agent navigates to collect plastic waste efficiently. The agent earns positive rewards for successful collection and conversion actions, while inefficient actions result in penalties.

## Installation

To install and run the DQN model, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SamuelOnyangoOmondi/DQN.git
   cd DQN
   ```

2. **Set Up a Virtual Environment** (recommended):
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
  Execute `train.py` to train the agent. Hyperparameters like learning rate and episode count can be adjusted in the script.

- **Evaluating the Agent**:
  Use `play.py` to visualize the agent's navigation and waste management strategy in the environment post-training.

- **Testing Specific Conditions**:
  Run `test_conditions.py` to evaluate the agent's behavior under various initial conditions.

## Video Submission

A detailed walkthrough of the project setup, execution, and a demonstration of the trained agent is available in this video submission:
[Watch the Video](https://example.com/plastech_video)

## Contributing

Contributions are highly appreciated. Please fork this repository, create your feature branch, and submit pull requests with any enhancements. Open issues for any bugs or feature suggestions you may encounter.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contact

For more information or inquiries, please contact [Samuel Omondi](mailto:s.omondi@alustudent.com).
