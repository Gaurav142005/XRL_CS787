import gymnasium as gym
from gymnasium import spaces
from minigrid.wrappers import FullyObsWrapper, FlatObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
import numpy as np
import time
import os
import google.generativeai as genai
from dotenv import load_dotenv
from adversarial_callback import AdversarialCallback 
load_dotenv()

# configuring API key for Gemini
try:
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"] 
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Gemini API Key configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    exit()

class RemoveDropActionWrapper(gym.Wrapper):
    # Removes the 'drop' action from the action space so for better training.
    def __init__(self, env):
        super().__init__(env)
        # action 4 : 'drop'
        self.drop_action = 4
        n_actions = self.action_space.n - 1
        self.action_space = spaces.Discrete(n_actions)

    def step(self, action):
        if action >= self.drop_action:
            action += 1
        return self.env.step(action)


# Reward function wrapper
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
        
        # every step has a small negative reward to encourage efficiency
        reward -= 0.01
        
        # penalty to encourage movement
        if tuple(base_env.agent_pos) == self.last_pos:
            reward -= 0.05
        
        # agent was picking up at an empty cell which made its' performance worse so the below addition
        # penaliseze useless PICKUP actions
        PICKUP_ACTION = 3
        if action == PICKUP_ACTION and not info.get("picked_up", False):
            reward -= 0.1

        self.last_action = action
        
        # High rewards for key pickup and door opening
        if info.get("picked_up", False):
            reward += 0.2
        if info.get("door_opened", False):
            reward += 0.3
            
        self.last_pos = tuple(base_env.agent_pos)
        return obs, reward, terminated, truncated, info

# Environment Creation
def make_env(render_mode=None, env_grid = "MiniGrid-DoorKey-6x6-v0"):
    env = gym.make(env_grid, render_mode=render_mode)     
    env = RemoveDropActionWrapper(env)
    env = DoorKeyRewardWrapper(env)
    env = FullyObsWrapper(env)
    env = FlatObsWrapper(env)
    return env


train_env = DummyVecEnv([make_env])


# Baseline Training
print("Baseline training start")
model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gae_lambda=0.95,
    gamma=0.99,
    ent_coef=0.01,
)
new_log_dir = "./training_logs/ppo_adversarial_from_scratch_logs/"
model.set_logger(configure(new_log_dir, ["stdout", "csv", "tensorboard"]))
model.learn(
    total_timesteps=500_000,
    callback=None
)
final_model_path = "./results/6x6-Grid/ppo_adversarial_from_scratch_6x6.zip"
model.save(final_model_path)
print(f"Model saved as '{final_model_path}'.")
# end Baseline Training



#Visualisation function
def visualize_trained_agent(model_path="./results/6x6-Grid/ppo_adversarial_from_scratch_6x6.zip", episodes=5, env_grid = "MiniGrid-DoorKey-6x6-v0"):
    print(f"\n Model path: {model_path}")
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

model_path = "./results/6x6-Grid/ppo_adversarial_from_scratch_6x6.zip"

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit()
    
model = PPO.load(model_path, env=train_env)


adversarial_cb = AdversarialCallback(k=25, lambda_penalty=1.0, verbose=1)


new_log_dir = "./training_logs/ppo_adversarial_finetuned_logs/"
model.set_logger(configure(new_log_dir, ["stdout", "csv", "tensorboard"]))

model.learn(
    total_timesteps=100_000,      
    callback=adversarial_cb,      
    reset_num_timesteps=False     
)

final_model_path = "./results/6x6-Grid/ppo_adversarial_finetuned_6x6.zip"
model.save(final_model_path)
print(f"Model saved as '{final_model_path}'.")



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