from stable_baselines3 import PPO
import time
import gymnasium as gym
from gymnasium import spaces
from minigrid.wrappers import FullyObsWrapper, FlatObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
import numpy as np


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


def visualize_trained_agent(model_path="results/6x6-Grid/ppo_adversarial_finetuned_6x6.zip", episodes=5, env_grid = "MiniGrid-DoorKey-6x6-v0"):
    print(f"\nModel Path: {model_path}")
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