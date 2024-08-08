from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from plastech_env import PlasTechEnv

def simulate_production():
    # Load the environment
    env = PlasTechEnv()
    
    # Load the trained model
    model = PPO.load("ppo_plastech")

    # Prepare for rendering
    images = []
    obs = env.reset()

    for _ in range(200):  # simulate for 200 steps
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        img = env.render(mode='rgb_array')
        images.append(img)
        if dones:
            obs = env.reset()

    env.close()
    return images

def display_frames_as_gif(frames):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    ani = matplotlib.animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    plt.show()

if __name__ == "__main__":
    frames = simulate_production()
    display_frames_as_gif(frames)
