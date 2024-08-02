from Samcase_env import HospitalEnv

env = HospitalEnv()
env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    print(f"Step: {action}, State: {obs}, Reward: {reward}")

env.close()
