import numpy as np
from Samcase_env import HospitalEnv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl2.agents.dqn import DQNAgent
from rl2.policy import BoltzmannQPolicy
from rl2.memory import SequentialMemory

# Initialize environment
env = HospitalEnv()
nb_actions = env.action_space.n

# Model Configuration
model = Sequential([
    Flatten(input_shape=(1,) + env.observation_space.shape),
    Dense(24, activation='relu'),
    Dense(24, activation='relu'),
    Dense(nb_actions, activation='linear')
])

# Agent Setup
memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, policy=policy, target_model_update=1e-2, nb_steps_warmup=10)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.load_weights('dqn_hospital_weights.h5f')

# Simulation
for episode in range(10):
    state = env.reset()
    done = False
    while not done:
        action = dqn.forward(state)
        next_state, reward, done, _ = env.step(action)
        env.render()
        state = next_state
