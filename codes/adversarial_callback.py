import google.generativeai as genai
from stable_baselines3.common.callbacks import BaseCallback
import json
import time
import numpy as np

# Helper function to describe the environment state in the form of a text 
def get_textual_state(env):
    unwrapped_env = env.unwrapped
    agent_pos = unwrapped_env.agent_pos
    key_pos = None
    door_pos = None
    door_obj = None 

    # Find key and door positions/objects
    for i in range(unwrapped_env.grid.width):
        for j in range(unwrapped_env.grid.height):
            cell = unwrapped_env.grid.get(i, j)
            if cell:  # Check if cell is not empty
                if cell.type == 'key':
                    key_pos = (i, j)
                elif cell.type == 'door':
                    door_pos = (i, j)
                    door_obj = cell 

    # Build the state description
    state_desc = f"Agent at {agent_pos}. "
    
    if key_pos:
        state_desc += f"Key at {key_pos}. "
        
    if door_pos and door_obj:
        # We check the 'is_locked' property of the door_obj we found
        state_desc += f"Door at {door_pos} and is {'locked' if door_obj.is_locked else 'unlocked'}. "
    
    state_desc += f"Agent is carrying key: {'yes' if unwrapped_env.carrying else 'no'}."
    
    return state_desc

# Function to calculate heuristics
def get_state_heuristics(unwrapped_env):
    """Calculates heuristics about the current state."""
    agent_pos = unwrapped_env.agent_pos
    key_pos, door_pos = None, None

    # Find key and door
    for i in range(unwrapped_env.grid.width):
        for j in range(unwrapped_env.grid.height):
            cell = unwrapped_env.grid.get(i, j)
            if cell:
                if cell.type == 'key':
                    key_pos = (i, j)
                elif cell.type == 'door':
                    door_pos = (i, j)

    # Determine the current objective
    if not unwrapped_env.carrying and key_pos:
        objective_name = "key"
        objective_pos = key_pos
    elif unwrapped_env.carrying and door_pos:
        objective_name = "door"
        objective_pos = door_pos
    else:
        # No objective (e.g., key is gone, or in an env with no key/door)
        return {"objective": "none", "distance": 0}

    if objective_pos:
        # Calculate Manhattan distance (it's a grid world)
        distance = abs(agent_pos[0] - objective_pos[0]) + abs(agent_pos[1] - objective_pos[1])
        return {"objective": objective_name, "distance": int(distance)}
    
    return {"objective": "none", "distance": 0}


class AdversarialCallback(BaseCallback):
    """
    A custom callback that uses a caching LLM to apply targeted penalties.

    :param k: How often (in steps) to check the agent's progress.
    :param lambda_penalty: The scaling factor for the LLM's penalty.
    :param model_name: The name of the Gemini model to use.
    :param verbose: Verbosity level.
    """
    def __init__(self, k: int, lambda_penalty: float, model_name: str = "gemini-2.5-flash", verbose: int = 1):
        super(AdversarialCallback, self).__init__(verbose)
        self.k = k
        self.lambda_penalty = lambda_penalty
        
        self.critique_cache = {}    #  Cache for LLM responses 
        self.last_heuristics = {"objective": "none", "distance": 0}  #  Store last state's heuristics 
        
        try:
            self.gemini_model = genai.GenerativeModel(model_name)
            if self.verbose > 0:
                print(f"Gemini model '{model_name}' initialized for callback.")
        except Exception as e:
            print(f"CRITICAL: Failed to initialize Gemini model. {e}")
            self.gemini_model = None

    def _on_reset(self) -> None:
        """Called when the environment is reset."""
        # Reset last_heuristics to the new state
        self.last_heuristics = get_state_heuristics(self.training_env.envs[0].unwrapped)

    def _on_step(self) -> bool:
        
        # 1. FREQUENCY CHECK 
        if self.n_calls % self.k != 0:  # Only run our logic every 'k' steps
            return True # Do nothing for other steps

        if not self.gemini_model:
            if self.verbose > 0:
                print("DEBUG: Gemini model not initialized, skipping callback logic.")
            return True

        # 2. GET CURRENT STATE & PROGRESS 
        current_env = self.training_env.envs[0].unwrapped
        current_heuristics = get_state_heuristics(current_env)
        
        progress_str = "no_change"
        objective = current_heuristics.get("objective", "none")
        
        # Compare to last heuristics check
        if objective == self.last_heuristics.get("objective", "none") and objective != "none":
            if current_heuristics["distance"] < self.last_heuristics["distance"]:
                progress_str = "moved_closer"
            elif current_heuristics["distance"] > self.last_heuristics["distance"]:
                progress_str = "moved_further"

        # 3. CREATE CACHE KEY 
        # Get the action the agent just took (from the policy)
        action_idx = self.locals["actions"][0]
        
        # This map reflects the agent's 6 possible actions AFTER the 'drop' wrapper
        action_map = {0: 'turn_left', 1: 'turn_right', 2: 'move_forward', 3: 'pickup', 4: 'toggle', 5: 'done'}
        action_str = action_map.get(action_idx, "unknown_action")

        cache_key = (objective, progress_str, action_str)
        llm_feedback = None

        # 4. CHECK CACHE 
        if cache_key in self.critique_cache:
            if self.verbose > 0:
                print(f"DEBUG [Step {self.n_calls}]: Cache HIT for {cache_key}.")
            llm_feedback = self.critique_cache[cache_key]
        
        # 5. CALL API (if cache miss) 
        else:
            if self.verbose > 0:
                print(f"DEBUG [Step {self.n_calls}]: Cache MISS for {cache_key}. Calling Gemini API...")
            
            prompt = self.build_adversary_prompt(objective, progress_str, action_str)
            
            try:
                response = self.gemini_model.generate_content(prompt)
                raw_text = response.text
                
                json_start = raw_text.find('{')
                json_end = raw_text.rfind('}') + 1
                
                if json_start == -1 or json_end == 0:
                    raise Exception(f"No JSON object found in response: {raw_text}")
                
                json_str = raw_text[json_start:json_end]
                llm_feedback = json.loads(json_str)
            
                self.critique_cache[cache_key] = llm_feedback
                if self.verbose > 0:
                    print(f"DEBUG: API Success. Saved critique for {cache_key} to cache.")

            except Exception as e:
                if self.verbose > 0:
                    print(f"Error during LLM call for key {cache_key}: {e}")
                
                llm_feedback = {"critique": f"API Error: {e}", "penalty_score": 0.0}

            #  3. ADD THE DELAY 
            # Wait 10 seconds *after* any API call (success or fail) to respect the free tier rate limit.
            if self.verbose > 0:
                print("DEBUG: Waiting 10s to respect API rate limit...")
            time.sleep(10)

        # 6. APPLY PENALTY 
        penalty = float(llm_feedback.get("penalty_score", 0.0))
        
        if penalty > 0:
            critique = llm_feedback.get('critique', 'No critique')
            if self.verbose > 0:
                print(f"DEBUG: Applying penalty {penalty} for: {critique}")
            
            buffer = self.model.rollout_buffer
            
            # This applies the penalty to the last step 
            last_step_index = (buffer.pos - 1) % buffer.buffer_size
            penalty_to_apply = self.lambda_penalty * penalty
            
            buffer.rewards[last_step_index] -= penalty_to_apply
            
            self.logger.record("adversary/applied_penalty", penalty_to_apply)
            self.logger.record("adversary/critique_text", critique)
        
        self.logger.record("adversary/cache_size", len(self.critique_cache))
        
        # 7. UPDATE LAST HEURISTICS FOR NEXT CHECK 
        self.last_heuristics = current_heuristics
        
        return True

    def build_adversary_prompt(self, objective, progress, action):
        """
        Builds a simple, heuristic-based prompt for the LLM.
        """
        prompt = f"""
        You are an AI Coach observing a 'Player' in a MiniGrid environment.
        Your goal is to provide a 'penalty_score' (0.0 to 1.0) for a specific situation.

         CURRENT SITUATION 
        -   Agent's Current Objective: '{objective}' (e.g., 'key', 'door', 'none')
        -   Agent's Progress (since last check): '{progress}' (e.g., 'moved_closer', 'moved_further', 'no_change')
        -   Agent's Last Action: '{action}'

         SCORING RUBRIC 
        -   **Severe Suboptimality (Score: 0.8 - 1.0):**
            -   Progress is 'moved_further'. (Moving away from the goal).
            -   Action is 'toggle' but Objective is 'key'. (Trying to open door without key).
        -   **Moderate Suboptimality (Score: 0.4 - 0.7):**
            -   Progress is 'no_change' AND Action is 'move_forward'. (Likely walked into a wall).
            -   Progress is 'no_change' AND Action is 'turn_left' or 'turn_right'. (Wasting time oscillating).
            -   Action is 'pickup' but Objective is not 'key' or agent is already carrying it.
        -   **Optimal Behavior (Score: 0.0):**
            -   Progress is 'moved_closer'. (Good!)
            -   Action is 'pickup' and Objective is 'key'.
            -   Action is 'toggle' and Objective is 'door'.
            -   Any other reasonable, non-penalized action.

        TASK 
        Based *only* on the situation and rubric, provide a critique and penalty.
        Return your analysis in this *exact* JSON format:
        
        {{
          "critique": "A brief explanation of why this situation is good or bad, based on the rubric.",
          "penalty_score": 0.0
        }}

         EXAMPLE 
        Situation: Objective='key', Progress='moved_further', Action='move_forward'
        {{
          "critique": "Severe: The agent's objective is the 'key', but it 'moved_further' away from it.",
          "penalty_score": 0.9
        }}

        Now, analyze this situation: Objective='{objective}', Progress='{progress}', Action='{action}'
        Return only the JSON object.
        """
        return prompt