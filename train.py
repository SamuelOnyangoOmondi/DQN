import gym
import numpy as np
from Samcase_env import HospitalEnv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl2.agents.dqn import DQNAgent
from rl2.policy import BoltzmannQPolicy
from rl2.memory import SequentialMemory

# Setup the environment
env = HospitalEnv()

# Model Configuration
model = Sequential([
    Flatten(input_shape=(1,) + env.observation_space.shape),
    Dense(24, activation='relu'),
    Dense(24, activation='relu'),
    Dense(env.action_space.n, activation='linear')
])

# Agent Configuration
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

# Training
dqn.fit(env, nb_steps=5000, visualize=False, verbose=2)

# Save Model Weights
dqn.save_weights('dqn_hospital_weights.h5f', overwrite=True)
