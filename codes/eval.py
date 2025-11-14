import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper, FlatObsWrapper
from stable_baselines3 import PPO
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import time
import warnings
import os

# (This must be identical to your training.py setup)

class RemoveDropActionWrapper(gym.Wrapper):
    """
    Removes the 'drop' action from the action space.
    """
    def __init__(self, env):
        super().__init__(env)
        # action 4 is 'drop'
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
        
        PICKUP_ACTION = 3
        if action == PICKUP_ACTION and not info.get("picked_up", False):
            reward -= 0.25

        self.last_action = action
        
        if info.get("picked_up", False):
            reward += 0.2
        if info.get("door_opened", False):
            reward += 0.3
            
        self.last_pos = tuple(base_env.agent_pos)
        return obs, reward, terminated, truncated, info

def make_env(render_mode=None, env_grid="MiniGrid-DoorKey-6x6-v0"):
    env = gym.make(env_grid, render_mode=render_mode)
    env = RemoveDropActionWrapper(env)
    env = DoorKeyRewardWrapper(env)
    env = FullyObsWrapper(env)
    env = FlatObsWrapper(env)
    return env

N_EPISODES = 500
ENV_ID = "MiniGrid-DoorKey-6x6-v0"

models_to_evaluate = {
    "Base Model (Scratch)": "results/ppo_adversarial_from_scratch_6x6.zip",
    "LLM Fine-Tuned Model": "results/ppo_adversarial_finetuned_6x6.zip"
}

results = {}

# Create a vectorized environment for evaluation
# We wrap it in a function for DummyVecEnv
def env_fn():
    return make_env(env_grid=ENV_ID)

eval_env = DummyVecEnv([env_fn])

print(f"Starting Evaluation")
print(f"Environment: {ENV_ID}")
print(f"Models: {', '.join(models_to_evaluate.keys())}")
print(f"Episodes per model: {N_EPISODES}\n")

start_time = time.time()

for model_name, model_path in models_to_evaluate.items():
    print(f"Evaluating: {model_name} (from {model_path})...")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}. Skipping.")
        results[model_name] = (None, None)
        continue
        
    try:
        model = PPO.load(model_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            mean_reward, std_reward = evaluate_policy(
                model, 
                eval_env, 
                n_eval_episodes=N_EPISODES,
                deterministic=True,     
                render=False             
            )
            
        results[model_name] = (mean_reward, std_reward)
        print(f"Done.")
        
    except Exception as e:
        print(f"ERROR: An error occurred during evaluation for {model_name}: {e}")
        results[model_name] = (None, None)

end_time = time.time()
print(f"\nEvaluation finished in {end_time - start_time:.2f} seconds.")

print("\nFinal Comparison")
print(f"Average reward over {N_EPISODES} deterministic episodes (Mean +/- Std):")
print("-" * 40)
for name, (mean, std) in results.items():
    if mean is not None:
        print(f"  - {name}: \t {mean:.2f} +/- {std:.2f}")
    else:
        print(f"  - {name}: \t FAILED TO EVALUATE")

print("-" * 40)