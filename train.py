import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from Samcase_env import HospitalEnv

# Setup the environment
env = HospitalEnv()

# Build the model
model = Sequential([
    Flatten(input_shape=(1,) + env.observation_space.shape),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(env.action_space.n, activation='linear')
])

# Configure and compile the agent
memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

# Train the agent
observation, _ = env.reset()
observation = np.array([observation])  # Ensure proper shape
print("Training observation shape:", observation.shape)

# Debugging: Check the shape right before model prediction
def debug_predict(model, observation):
    print("Predicting with shape:", observation.shape)
    return model.predict_on_batch(observation)

# Replace model prediction call with debug function in your DQN code if necessary
dqn.fit(env, nb_steps=5000, visualize=False, verbose=2)

# Save weights
dqn.save_weights('dqn_weights.h5f', overwrite=True)
