import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent  # Adjust this if your package name is different
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from Samcase_env import HospitalEnv

# Initialize the environment
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

# Load the trained weights
try:
    dqn.load_weights('dqn_hospital_weights.h5f')
except Exception as e:
    print(f"Failed to load weights: {e}")

# Run the game
for episode in range(5):  # Play for 5 episodes
    state = env.reset()
    state = np.array([state])  # Ensure the state conforms to what the model expects
    total_reward = 0
    done = False

    while not done:
        action = dqn.forward(state)
        state, reward, done, info = env.step(action)
        state = np.array([state])  # Update state to match model's expectations
        total_reward += reward
        env.render()  # Optional: Render the environment to see it in action

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

env.close()
