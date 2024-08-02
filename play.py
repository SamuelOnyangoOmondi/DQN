from Sam'scase_env import HospitalEnv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory

# Load the model and environment
env = HospitalEnv()
model = Sequential([
    Flatten(input_shape=(1,) + env.observation_space.shape),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(env.action_space.n, activation='linear')
])
dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=SequentialMemory(limit=50000, window_length=1),
               target_model_update=1e-2)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.load_weights('dqn_weights.h5f')

# Play the game
observation = env.reset()
done = False
while not done:
    action = dqn.forward(observation)
    observation, reward, done, info = env.step(action)
    env.render()

env.close()
