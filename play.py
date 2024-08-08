from tensorflow.keras.models import Sequential
from rl.agents.dqn import DQNAgent
from plastech_env import PlasTechEnv
from train import build_model, build_agent

if __name__ == "__main__":
    # Initialize the environment
    env = PlasTechEnv()
    np.random.seed(123)
    env.seed(123)

    # Rebuild the model and agent for playing
    num_actions = env.action_space.n
    model = build_model((env.observation_space.shape[0],), num_actions)
    dqn = build_agent(model, num_actions)

    # Load the trained weights
    dqn.load_weights('dqn_plastech_weights.h5f')

    # Run a single episode to see the agent in action
    obs = env.reset()
    done = False
    while not done:
        action = dqn.forward(obs)
        obs, reward, done, info = env.step(action)
        env.render(mode='console')  # or 'human' if more sophisticated rendering is implemented

    print("Simulation complete.")
