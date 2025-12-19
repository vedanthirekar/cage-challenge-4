"""
A2C agent for CAGE Challenge 4
Uses SB3 with PettingZoo integration
"""

import os
import numpy as np
from CybORG.Agents import BaseAgent


class A2CAgent(BaseAgent):
    """
    A2C-trained agent for CAGE Challenge 4
    """

    def __init__(self, checkpoint_path: str = None, name: str = None):
        super().__init__(name)
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.step_count = 0
        self.fallback_logged = False

        if checkpoint_path and (os.path.exists(checkpoint_path) or 
                               os.path.exists(checkpoint_path + ".zip")):
            self._load_model(checkpoint_path)
        else:
            print(f"⚠️  Model path not found: {checkpoint_path}")

    def _load_model(self, checkpoint_path: str):
        """Load trained A2C model"""
        try:
            from stable_baselines3 import A2C

            abs_path = os.path.abspath(checkpoint_path)
            print(f"🔍 Loading A2C model from: {abs_path}")

            if os.path.exists(abs_path) or os.path.exists(abs_path + ".zip"):
                self.model = A2C.load(abs_path)
                print(f"✅ Successfully loaded A2C model for {self.name}")
            else:
                print(f"❌ Model not found: {abs_path}")
                self.model = None

        except Exception as e:
            print(f"⚠️  Could not load A2C model: {e}")
            self.model = None

    def get_action(self, observation, action_space):
        """Get action from trained model or fallback"""
        self.step_count += 1

        if self.model is not None:
            try:
                # Convert observation to numpy array
                if not isinstance(observation, np.ndarray):
                    observation = np.array(observation)

                # Get action from model (deterministic for evaluation)
                action, _ = self.model.predict(observation, deterministic=True)

                if self.step_count <= 5:
                    print(f"✅ {self.name} using TRAINED A2C MODEL")

                return int(action)

            except Exception as e:
                if not self.fallback_logged:
                    print(f"⚠️  {self.name} inference failed: {e}")
                    self.fallback_logged = True
                return self._fallback_action(action_space)
        else:
            if not self.fallback_logged:
                print(f"⚠️  {self.name} no model loaded, using fallback")
                self.fallback_logged = True
            return self._fallback_action(action_space)

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
