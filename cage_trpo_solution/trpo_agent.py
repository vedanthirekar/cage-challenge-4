"""
TRPO agent for CAGE Challenge 4
"""

import os
import numpy as np
from CybORG.Agents import BaseAgent


class TRPOAgent(BaseAgent):
    """
    TRPO-trained agent for CAGE Challenge 4
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
        if checkpoint_path and (os.path.exists(checkpoint_path) or os.path.exists(checkpoint_path + ".zip")):
            if TRPOAgent._shared_model is None or TRPOAgent._shared_model_path != checkpoint_path:
                self._load_model(checkpoint_path)
                TRPOAgent._shared_model = self.model
                TRPOAgent._shared_model_path = checkpoint_path
            else:
                self.model = TRPOAgent._shared_model
        else:
            print(f"⚠️  Model path not found: {checkpoint_path}")
            self.model = None

    def _load_model(self, checkpoint_path: str):
        """Load trained TRPO model"""
        try:
            from sb3_contrib import TRPO

            abs_path = os.path.abspath(checkpoint_path)
            print(f"🔍 Loading TRPO model from: {abs_path}")

            if os.path.exists(abs_path) or os.path.exists(abs_path + ".zip"):
                self.model = TRPO.load(abs_path)
                print("✅ Successfully loaded TRPO model")
            else:
                print(f"❌ Model not found: {abs_path}")
                self.model = None

        except Exception as e:
            print(f"⚠️  Could not load TRPO model: {e}")
            self.model = None

    def get_action(self, observation, action_space):
        """Get action from trained model or fallback"""
        self.step_count += 1

        # Store this agent's observation in shared buffer
        if isinstance(observation, np.ndarray):
            TRPOAgent._obs_buffer[self.name] = observation.astype(np.int32)
        else:
            TRPOAgent._obs_buffer[self.name] = np.array([int(observation)], dtype=np.int32)

        # Check if we have cached action for this agent
        if self.name in TRPOAgent._action_cache:
            action = TRPOAgent._action_cache[self.name]
            del TRPOAgent._action_cache[self.name]
            return action

        # When we have all 5 agents' observations, get combined action
        expected_agents = ["blue_agent_0", "blue_agent_1", "blue_agent_2",
                          "blue_agent_3", "blue_agent_4"]

        if all(agent in TRPOAgent._obs_buffer for agent in expected_agents):
            if self.model is not None:
                try:
                    # Combine observations in correct order
                    obs_list = [TRPOAgent._obs_buffer[agent] for agent in expected_agents]
                    combined_obs = np.concatenate(obs_list).astype(np.int32)

                    # Get action from model
                    actions, _ = self.model.predict(combined_obs, deterministic=True)

                    if self.step_count <= 5:
                        print(f"✅ Using TRAINED TRPO MODEL")

                    # Cache actions for all agents
                    for i, agent in enumerate(expected_agents):
                        TRPOAgent._action_cache[agent] = int(actions[i])

                    # Clear observation buffer
                    TRPOAgent._obs_buffer.clear()

                    # Return action for this agent
                    return TRPOAgent._action_cache.pop(self.name)

                except Exception as e:
                    if not self.fallback_logged:
                        print(f"⚠️  Inference failed: {e}")
                        self.fallback_logged = True
                    TRPOAgent._obs_buffer.clear()
                    TRPOAgent._action_cache.clear()
                    return self._fallback_action(action_space)
            else:
                if not self.fallback_logged:
                    print(f"⚠️  No model loaded, using fallback")
                    self.fallback_logged = True
                TRPOAgent._obs_buffer.clear()
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
