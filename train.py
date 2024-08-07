import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent  # If using keras-rl
# from rl2.agents.dqn import DQNAgent  # If rl2 was a typo and the correct one is rl
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from Samcase_env import HospitalEnv

# Setup the environment
env = HospitalEnv()
nb_actions = env.action_space.n

# Build the model
model = Sequential([
    Flatten(input_shape=(1,) + env.observation_space.shape),
    Dense(24, activation='relu'),
    Dense(24, activation='relu'),
    Dense(nb_actions, activation='linear')
])

# Configure and compile the agent
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

# Train the agent
dqn.fit(env, nb_steps=5000, visualize=False, verbose=2)

# Save weights
dqn.save_weights('dqn_hospital_weights.h5f', overwrite=True)
