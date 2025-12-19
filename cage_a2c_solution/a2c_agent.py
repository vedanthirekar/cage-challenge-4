"""
A2C agent for CAGE Challenge 4
Loads trained model and provides actions for each agent
"""

import os
import numpy as np
from CybORG.Agents import BaseAgent


class A2CAgent(BaseAgent):
    """
    A2C-trained agent for CAGE Challenge 4
    Shared model across all agents
    """

    # Shared model instance
    _shared_model = None
    _shared_model_path = None
    # Shared observation buffer across all agents
    _obs_buffer = {}
    _action_cache = {}

    def __init__(self, checkpoint_path: str = None, name: str = None):
        super().__init__(name)
        self.checkpoint_path = checkpoint_path
        self.step_count = 0
        self.fallback_logged = False

        # Load shared model once
        # SB3 saves as .zip, so check for both path and path.zip
        if checkpoint_path and (os.path.exists(checkpoint_path) or os.path.exists(checkpoint_path + ".zip")):
            if A2CAgent._shared_model is None or A2CAgent._shared_model_path != checkpoint_path:
                self._load_model(checkpoint_path)
                A2CAgent._shared_model = self.model
                A2CAgent._shared_model_path = checkpoint_path
            else:
                self.model = A2CAgent._shared_model
        else:
            print(f"⚠️  Model path not found: {checkpoint_path}")
            self.model = None

    def _load_model(self, checkpoint_path: str):
        """Load trained A2C model"""
        try:
            from stable_baselines3 import A2C

            abs_path = os.path.abspath(checkpoint_path)
            print(f"🔍 Loading A2C model from: {abs_path}")

            # SB3 saves as .zip but load() expects path without extension
            # Check if either the path or path.zip exists
            if os.path.exists(abs_path) or os.path.exists(abs_path + ".zip"):
                self.model = A2C.load(abs_path)
                print("✅ Successfully loaded A2C model")
            else:
                print(f"❌ Model not found: {abs_path}")
                self.model = None

        except Exception as e:
            print(f"⚠️  Could not load A2C model: {e}")
            self.model = None

    def get_action(self, observation, action_space):
        """Get action from trained model or fallback"""
        self.step_count += 1

        # Store this agent's observation in shared buffer
        if isinstance(observation, np.ndarray):
            A2CAgent._obs_buffer[self.name] = observation.astype(np.int32)
        else:
            A2CAgent._obs_buffer[self.name] = np.array([int(observation)], dtype=np.int32)

        # Check if we have cached action for this agent
        if self.name in A2CAgent._action_cache:
            action = A2CAgent._action_cache[self.name]
            del A2CAgent._action_cache[self.name]
            return action

        # When we have all 5 agents' observations, get combined action
        expected_agents = ["blue_agent_0", "blue_agent_1", "blue_agent_2", 
                          "blue_agent_3", "blue_agent_4"]
        
        if all(agent in A2CAgent._obs_buffer for agent in expected_agents):
            if self.model is not None:
                try:
                    # Combine observations in correct order
                    obs_list = [A2CAgent._obs_buffer[agent] for agent in expected_agents]
                    combined_obs = np.concatenate(obs_list).astype(np.int32)

                    # Get action from model
                    actions, _ = self.model.predict(combined_obs, deterministic=True)

                    if self.step_count <= 5:
                        print(f"✅ Using TRAINED A2C MODEL")
                        print(f"   Obs shape: {combined_obs.shape}, Actions: {actions}")

                    # Cache actions for all agents
                    for i, agent in enumerate(expected_agents):
                        A2CAgent._action_cache[agent] = int(actions[i])

                    # Clear observation buffer
                    A2CAgent._obs_buffer.clear()

                    # Return action for this agent
                    return A2CAgent._action_cache.pop(self.name)

                except Exception as e:
                    if not self.fallback_logged:
                        print(f"⚠️  Inference failed: {e}")
                        print(f"   Obs shapes: {[obs.shape for obs in obs_list]}")
                        self.fallback_logged = True
                    A2CAgent._obs_buffer.clear()
                    A2CAgent._action_cache.clear()
                    return self._fallback_action(action_space)
            else:
                if not self.fallback_logged:
                    print(f"⚠️  No model loaded, using fallback")
                    self.fallback_logged = True
                A2CAgent._obs_buffer.clear()
                return self._fallback_action(action_space)
        else:
            # Still waiting for other agents' observations
            return 0

    def _fallback_action(self, action_space):
        """Fallback action when no trained model"""
        cycle = self.step_count % 15
        if cycle < 5:
            return 0
        if cycle < 8:
            return 1
        if cycle < 10:
            return 2
        if cycle < 12:
            return 4
        return 0

    def end_episode(self):
        pass

    def set_initial_values(self, action_space, observation):
        pass

    def train(self, results):
        pass
