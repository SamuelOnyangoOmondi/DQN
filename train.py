from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

from Samcase_env import HospitalEnv

# Setup the environment
env = HospitalEnv()

# Make sure the observation space is properly initialized in HospitalEnv
# For example, the observation_space could be something like this in HospitalEnv:
# self.observation_space = spaces.Box(low=0, high=4, shape=(2,), dtype=np.int32)
# which represents a 5x5 grid (0 to 4, both x and y coordinates).

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
dqn.fit(env, nb_steps=5000, visualize=False, verbose=2)

# Save weights
dqn.save_weights('dqn_weights.h5f', overwrite=True)
