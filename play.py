import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl2.agents.dqn import DQNAgent
from rl2.memory import SequentialMemory
from rl2.policy import EpsGreedyQPolicy
from Samcase_env import HospitalEnv

# Load the model and environment
env = HospitalEnv()
model = Sequential([
    Flatten(input_shape=(1,) + env.observation_space.shape),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(env.action_space.n, activation='linear')
])
memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.load_weights('dqn_weights.h5f')

# Play the game
observation, _ = env.reset()
done = False
while not done:
    action = dqn.forward(observation)
    observation, reward, done, info = env.step(action)
    env.render()
