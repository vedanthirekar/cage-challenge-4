"""
DQN agent for CAGE Challenge 4
"""

import os
import numpy as np
from CybORG.Agents import BaseAgent


class DQNAgent(BaseAgent):
    """
    DQN-trained agent for CAGE Challenge 4
    """

    def __init__(self, checkpoint_path: str = None,
                 policy_id: str = "policy_small", name: str = None):
        super().__init__(name)
        self.checkpoint_path = checkpoint_path
        self.policy_id = policy_id
        self.algorithm = None
        self.step_count = 0
        self.fallback_logged = False

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str):
        """Load trained DQN model"""
        try:
            from ray.rllib.algorithms.dqn import DQN
            from ray.tune import register_env
            import ray

            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, log_to_driver=False)

            def create_env(env_config):
                from CybORG import CybORG
                from CybORG.Agents import (SleepAgent, EnterpriseGreenAgent,
                                           FiniteStateRedAgent)
                from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
                from CybORG.Agents.Wrappers import EnterpriseMAE

                sg = EnterpriseScenarioGenerator(
                    blue_agent_class=SleepAgent,
                    green_agent_class=EnterpriseGreenAgent,
                    red_agent_class=FiniteStateRedAgent,
                    steps=env_config.get('episode_length', 100),
                )
                cyborg = CybORG(scenario_generator=sg)
                return EnterpriseMAE(cyborg)

            register_env("CAGE4", create_env)

            abs_path = os.path.abspath(checkpoint_path)
            print(f"🔍 Loading DQN model from: {abs_path}")

            if os.path.exists(abs_path):
                self.algorithm = DQN.from_checkpoint(abs_path)
                print("✅ Successfully loaded DQN model")
            else:
                print(f"❌ Checkpoint not found: {abs_path}")
                self.algorithm = None

        except Exception as e:
            print(f"⚠️  Could not load DQN model: {e}")
            self.algorithm = None

    def get_action(self, observation, action_space):
        """Get action from trained model or fallback"""
        self.step_count += 1

        if self.algorithm is not None:
            try:
                obs_array = self._process_observation(observation)
                action = self.algorithm.compute_single_action(
                    obs_array,
                    policy_id=self.policy_id,
                    explore=False
                )

                if self.step_count == 1:
                    print(f"✅ {self.name}: Using TRAINED DQN MODEL")

                if isinstance(action, (int, np.integer)):
                    return max(0, min(action, action_space.n - 1))
                return int(action) % action_space.n

            except Exception:
                if not self.fallback_logged:
                    print(f"⚠️  {self.name}: Inference failed, using fallback")
                    self.fallback_logged = True
                return self._fallback_action(action_space)
        else:
            if not self.fallback_logged:
                print(f"⚠️  {self.name}: No model - using fallback")
                self.fallback_logged = True
            return self._fallback_action(action_space)

    def _process_observation(self, observation):
        """Process observation for DQN"""
        if isinstance(observation, np.ndarray):
            return observation.astype(np.int32)
        return np.array([int(observation)], dtype=np.int32)

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
