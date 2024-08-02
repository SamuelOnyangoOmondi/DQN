from Samcase_env import HospitalEnv

env = HospitalEnv()
observation, _ = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    obs = np.array([obs])  # Ensure proper shape if passed to any model
    env.render()
    print(f"Step: {action}, State: {obs}, Reward: {reward}")

env.close()
