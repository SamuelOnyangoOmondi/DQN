import numpy as np
import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from plastech_env import PlasTechEnv
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

def build_model(state_shape, num_actions):
    """
    Build a neural network model that predicts the Q-values for each action in a given state.
    """
    model = Sequential([
        Flatten(input_shape=(1,) + state_shape),
        Dense(24, activation='relu'),
        Dense(24, activation='relu'),
        Dense(num_actions, activation='linear')
    ])
    return model

def build_agent(model, num_actions):
    """
    Compile the DQN agent with the given model and action space.
    """
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                  nb_actions=num_actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

if __name__ == "__main__":
    # Initialize the environment
    env = PlasTechEnv()
    np.random.seed(123)
    env.seed(123)

    # Get the number of actions from the environment's action space
    num_actions = env.action_space.n
    # Build the model
    model = build_model((env.observation_space.shape[0],), num_actions)
    print(model.summary())

    # Build and compile the agent
    dqn = build_agent(model, num_actions)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # Start training the agent
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

    # After training is complete, save the model weights
    dqn.save_weights('dqn_plastech_weights.h5f', overwrite=True)

    # Optionally, load the weights and continue training or start testing
    # dqn.load_weights('dqn_plastech_weights.h5f')
