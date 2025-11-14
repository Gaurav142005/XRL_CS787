import gymnasium as gym
from gymnasium import spaces
from minigrid.wrappers import FullyObsWrapper, FlatObsWrapper
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

# --- 1. ENVIRONMENT SETUP (Must match training) ---

class RemoveDropActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.drop_action = 4
        n_actions = self.action_space.n - 1
        self.action_space = spaces.Discrete(n_actions)

    def step(self, action):
        if action >= self.drop_action:
            action += 1
        return self.env.step(action)

class DoorKeyRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_pos = None
        self.last_action = None 

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_pos = tuple(self.env.unwrapped.agent_pos)
        self.last_action = None 
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        base_env = self.env.unwrapped
        reward -= 0.01
        if tuple(base_env.agent_pos) == self.last_pos:
            reward -= 0.02
        if action == self.last_action:
            reward -= 0.1  
        self.last_action = action 
        
        if info.get("picked_up", False):
            reward += 0.2
        if info.get("door_opened", False):
            reward += 0.3
        self.last_pos = tuple(base_env.agent_pos)
        return obs, reward, terminated, truncated, info

def make_env(render_mode=None, env_grid="MiniGrid-DoorKey-8x8-v0"):
    env = gym.make(env_grid, render_mode=render_mode)    
    env = RemoveDropActionWrapper(env)
    env = DoorKeyRewardWrapper(env)
    env = FullyObsWrapper(env)
    env = FlatObsWrapper(env)
    return env

# --- 2. EVALUATION CONFIGURATION ---

N_EPISODES = 200

models_to_evaluate = {
    "Adversarial (From Scratch)": "ppo_from_scratch_8x8.zip",
    "Adversarial (Fine-Tuned)": "ppo_adversarial_finetuned_8x8.zip"
}

results = {
    "rewards": {},
    "steps": {},
    "times": {},
    "success_rate": {}
}

# --- 3. DATA COLLECTION LOOP ---

for model_name, model_path in models_to_evaluate.items():
    print(f"Evaluating {model_name}...")
    
    try:
        model = PPO.load(model_path)
    except FileNotFoundError:
        print(f"Skipping {model_name}: File not found.")
        continue

    eval_env = make_env()
    
    ep_rewards = []
    ep_steps = []
    ep_times = []
    success_count = 0

    for ep in range(N_EPISODES):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0
        step_count = 0
        start_time = time.perf_counter()
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            total_reward += reward
            step_count += 1
            
            if terminated:
                done = True
            elif truncated:
                done = True
        
        end_time = time.perf_counter()
        
        ep_rewards.append(total_reward)
        ep_steps.append(step_count)
        ep_times.append(end_time - start_time)
        
        if terminated: 
            success_count += 1

    results["rewards"][model_name] = ep_rewards
    results["steps"][model_name] = ep_steps
    results["times"][model_name] = ep_times
    results["success_rate"][model_name] = (success_count / N_EPISODES) * 100

# --- 4. SEPARATE PLOTTING ---

colors = ['#1f77b4', '#ff7f0e'] # Consistent colors for the two models

# === PLOT 1: Reward vs Episode ===
plt.figure(figsize=(8, 5))
for i, (name, rewards) in enumerate(results["rewards"].items()):
    # Plot raw data faintly
    plt.plot(rewards, alpha=0.2, color=colors[i])
    # Plot Moving Average
    series = pd.Series(rewards)
    rolling_mean = series.rolling(window=10).mean()
    plt.plot(rolling_mean, label=f"{name} (Mov Avg)", color=colors[i], linewidth=2)

plt.title("Reward vs. Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_1_rewards.png")
plt.show()

# === PLOT 2: Wall-Clock Time vs Episode ===
plt.figure(figsize=(8, 5))
for i, (name, times) in enumerate(results["times"].items()):
    plt.plot(times, label=name, alpha=0.8, color=colors[i])

plt.title("Inference Time per Episode (Wall Clock)")
plt.xlabel("Episode")
plt.ylabel("Time (Seconds)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_2_times.png")
plt.show()

# === PLOT 3: Mean Steps to Completion (Bar Chart) ===
plt.figure(figsize=(7, 5))
model_names = list(models_to_evaluate.keys())
steps_means = [np.mean(results["steps"][name]) for name in model_names]
steps_stds = [np.std(results["steps"][name]) for name in model_names]

# Plot bars with error bars (standard deviation)
bars = plt.bar(model_names, steps_means, yerr=steps_stds, capsize=10, 
               color=['skyblue', 'salmon'], alpha=0.9)

plt.title("Mean Steps to Completion (Lower is Better)")
plt.ylabel("Steps")
plt.grid(True, axis='y', alpha=0.3)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig("plot_3_steps_bar.png")
plt.show()

# === PLOT 4: Success Rate (Bar Chart) ===
plt.figure(figsize=(7, 5))
success_rates = [results["success_rate"][name] for name in model_names]

bars_success = plt.bar(model_names, success_rates, color=['skyblue', 'salmon'], alpha=0.9)

plt.title("Success Rate (Goal Reached)")
plt.ylabel("Success %")
plt.ylim(0, 110) # Give some headroom for the text
plt.grid(True, axis='y', alpha=0.3)

for bar in bars_success:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig("plot_4_success_rate.png")
plt.show()