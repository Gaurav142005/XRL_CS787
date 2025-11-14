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
# --- Import your new callback ---
from adversarial_callback import AdversarialCallback 
load_dotenv()

# --- 1. CONFIGURE YOUR GEMINI API KEY ---
try:
    # Use your environment variable name
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"] 
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Gemini API Key configured successfully.")
except KeyError:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    print("Please create a .env file and add GOOGLE_API_KEY='your_key_here'")
    exit()
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    exit()

class RemoveDropActionWrapper(gym.Wrapper):
    """
    Removes the 'drop' action from the action space.
    Remaps actions so the agent cannot select 'drop'.
    """
    def __init__(self, env):
        super().__init__(env)
        # In MiniGrid, action 4 is 'drop'
        self.drop_action = 4
        
        # Remove 'drop' from the discrete action space (reduce size by 1)
        # New action space is 6: 0:left, 1:right, 2:fwd, 3:pickup, 4:toggle, 5:done
        n_actions = self.action_space.n - 1
        self.action_space = spaces.Discrete(n_actions)

    def step(self, action):
        # Remap actions from agent (0-5) to environment (0-6, skipping 4)
        if action >= self.drop_action:
            action += 1 # 4 -> 5 (toggle), 5 -> 6 (done)
        return self.env.step(action)


# 🧠 Custom Reward Shaping Wrapper
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
        
        # Small penalty for existing
        reward -= 0.01
        
        # Penalty for staying in the same spot (encourages exploration)
        if tuple(base_env.agent_pos) == self.last_pos:
            reward -= 0.05
        
        # Penalize attempting to pick up when nothing was picked up
        PICKUP_ACTION = 3 # This is the agent's action
        if action == PICKUP_ACTION and not info.get("picked_up", False):
            reward -= 0.1

        self.last_action = action
        
        # Reward for good actions
        if info.get("picked_up", False):
            reward += 0.2
        if info.get("door_opened", False):
            reward += 0.3
            
        self.last_pos = tuple(base_env.agent_pos)
        return obs, reward, terminated, truncated, info

# Environment Creation
def make_env(render_mode=None, env_grid = "MiniGrid-DoorKey-8x8-v0"):
    env = gym.make(env_grid, render_mode=render_mode)     
    env = RemoveDropActionWrapper(env)  # <-- Add this line
    env = DoorKeyRewardWrapper(env)
    env = FullyObsWrapper(env)
    env = FlatObsWrapper(env)
    return env

#  Training Environment
train_env = DummyVecEnv([make_env])


# --- TRAIN FROM SCRATCH ---
print("🧠 Initializing new PPO model for training from scratch...")
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
print("New model initialized.")
new_log_dir = "./ppo_scratch_logs/"
model.set_logger(configure(new_log_dir, ["stdout", "csv", "tensorboard"]))
print(f"Logging new training phase to {new_log_dir}")
print("🚀 Starting base training loop...")
model.learn(
    total_timesteps=500_000, # 500k steps for base training
    callback=None
)
final_model_path = "ppo_from_scratch_8x8.zip"
model.save(final_model_path)
print(f"Base training complete. Model saved as '{final_model_path}'.")
# --- END OF SCRATCH TRAINING ---



# 🎥 Visualize trained model
def visualize_trained_agent(model_path="ppo_from_scratch_8x8.zip", episodes=5, env_grid = "MiniGrid-DoorKey-8x8-v0"):
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
visualize_trained_agent(env_grid = "MiniGrid-DoorKey-8x8-v0")


model_path = "ppo_from_scratch_8x8.zip" # Assumes you have this file
print(f" Loading existing model from {model_path} for fine-tuning...")

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    print("Please run the 'TRAIN FROM SCRATCH' block first, or check the path.")
    exit()
    
model = PPO.load(model_path, env=train_env)
print("Model loaded successfully.")

# --- 3. INITIALIZE THE ADVERSARIAL CALLBACK ---
# k=25: Check progress every 25 steps.
# lambda_penalty=1.0: Apply the full penalty from the LLM.
adversarial_cb = AdversarialCallback(k=25, lambda_penalty=1.0, verbose=1)

# --- 4. SET UP NEW LOGGER ---
new_log_dir = "./ppo_adversarial_finetuned_logs/"
model.set_logger(configure(new_log_dir, ["stdout", "csv", "tensorboard"]))
print(f"Logging fine-tuning phase to {new_log_dir}")

# --- 5. RUN THE JOINT TRAINING LOOP ---
print("🚀 Starting adversarial FINE-TUNING loop...")
model.learn(
    total_timesteps=100_000,      # *** CHANGED: Train for 100k steps ***
                                  # (1000 was too short to learn)
    callback=adversarial_cb,      # Callback is now ACTIVE
    reset_num_timesteps=False     # Continue counting timesteps
)

# --- 6. SAVE THE FINAL ADVERSARIALLY-TRAINED MODEL ---
final_model_path = "ppo_adversarial_finetuned_8x8.zip" # New save name
model.save(final_model_path)
print(f"✅ Adversarial fine-tuning complete. Model saved as '{final_model_path}'.")


# 🎥 Visualize trained model
def visualize_trained_agent(model_path="ppo_adversarial_finetuned_8x8.zip", episodes=5, env_grid = "MiniGrid-DoorKey-8x8-v0"):
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
visualize_trained_agent(env_grid = "MiniGrid-DoorKey-8x8-v0")