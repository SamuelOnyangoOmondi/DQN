import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from Samcase_env import HospitalEnv

# Initialize environment
env = HospitalEnv()

# Build the model
model = Sequential([
    Flatten(input_shape=(1,) + env.observation_space.shape),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(env.action_space.n, activation='linear')
])

# Setup DQN agent
memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

# Train the agent
observation, _ = env.reset()
observation = np.array([observation])  # Add batch dimension
print("Training observation shape:", observation.shape)  # Debug print
dqn.fit(env, nb_steps=5000, visualize=False, verbose=2)

# Save model weights
dqn.save_weights('dqn_weights.h5f', overwrite=True)
