from stable_baselines3 import PPO
import time
from training import make_env


def visualize_trained_agent(model_path="./results/6x6-Grid/ppo_adversarial_finetuned_6x6.zip", episodes=5, env_grid = "MiniGrid-DoorKey-6x6-v0"):
    print(f"\nModel Path: {model_path}...")
    model = PPO.load(model_path)
    env = make_env(render_mode="human", env_grid=env_grid)

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            time.sleep(0.1)

        print(f"Episode {ep+1} Reward: {ep_reward:.2f}")

    env.close()
    print("Visualization finished.")


visualize_trained_agent(env_grid = "MiniGrid-DoorKey-6x6-v0")