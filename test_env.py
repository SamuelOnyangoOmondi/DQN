import numpy as np
from tensorflow.keras.models import Sequential
from rl.agents.dqn import DQNAgent
from plastech_env import PlasTechEnv
from train import build_model, build_agent

if __name__ == "__main__":
    # Initialize the environment
    env = PlasTechEnv()
    np.random.seed(123)
    env.seed(123)

    # Get the number of actions from the environment's action space
    num_actions = env.action_space.n

    # Rebuild the model and agent for testing
    model = build_model((env.observation_space.shape[0],), num_actions)
    dqn = build_agent(model, num_actions)

    # Load the trained weights
    dqn.load_weights('dqn_plastech_weights.h5f')

    # Evaluate the agent's performance
    scores = dqn.test(env, nb_episodes=10, visualize=True)
    print(np.mean(scores.history['episode_reward']))

    # Optionally, render the last episode step by step (if rendering is supported)
    env.render(mode='human')
