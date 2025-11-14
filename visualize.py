import gymnasium as gym
from gymnasium import spaces
from minigrid.wrappers import FullyObsWrapper, FlatObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
import numpy as np
import time
import os
from dotenv import load_dotenv
from training import make_env


# 🎥 Visualize trained model
def visualize_trained_agent(model_path="./results/6x6-Grid/ppo_adversarial_from_scratch.zip", episodes=5, env_grid = "MiniGrid-DoorKey-6x6-v0"):
    print(f"\n🎥 Visualizing final model from {model_path}...")
    model = PPO.load(model_path)
    env = make_env(render_mode="human", env_grid=env_grid)

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True) # Use deterministic for viz
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            time.sleep(0.1) # Slow down for human eyes

        print(f"🏁 Episode {ep+1} Reward: {ep_reward:.2f}")

    env.close()
    print("🎬 Visualization finished.")

# --- 7. VISUALIZE THE NEWLY TRAINED AGENT ---
visualize_trained_agent(env_grid = "MiniGrid-DoorKey-6x6-v0")