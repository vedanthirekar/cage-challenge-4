"""
APPO agent for CAGE Challenge 4
"""

import os
from CybORG.Agents import BaseAgent


class APPOAgent(BaseAgent):
    """
    APPO-trained agent for CAGE Challenge 4
    """

    # Shared algorithm instance
    _shared_algo = None
    _shared_checkpoint_path = None

    def __init__(self, checkpoint_path: str = None, name: str = None):
        super().__init__(name)
        self.checkpoint_path = checkpoint_path
        self.policy_id = "policy_large" if name == "blue_agent_4" else "policy_small"
        self.step_count = 0

        # Load shared algorithm once
        if checkpoint_path and os.path.exists(checkpoint_path):
            if APPOAgent._shared_algo is None or APPOAgent._shared_checkpoint_path != checkpoint_path:
                self._load_algorithm(checkpoint_path)
                APPOAgent._shared_algo = self.algo
                APPOAgent._shared_checkpoint_path = checkpoint_path
            else:
                self.algo = APPOAgent._shared_algo
        else:
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            self.algo = None

    def _load_algorithm(self, checkpoint_path: str):
        """Load trained APPO algorithm"""
        try:
            import ray
            from ray.rllib.algorithms.appo import APPO
            from ray.tune import register_env
            from CybORG import CybORG
            from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
            from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
            from CybORG.Agents.Wrappers import EnterpriseMAE

            abs_path = os.path.abspath(checkpoint_path)
            print(f"🔍 Loading APPO model from: {abs_path}")

            # Initialize Ray if not already
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, log_to_driver=False)

            # Register environment (required for checkpoint loading)
            def create_env(env_config):
                sg = EnterpriseScenarioGenerator(
                    blue_agent_class=SleepAgent,
                    green_agent_class=EnterpriseGreenAgent,
                    red_agent_class=FiniteStateRedAgent,
                    steps=env_config.get('episode_length', 500),
                )
                cyborg = CybORG(scenario_generator=sg)
                return EnterpriseMAE(cyborg)

            register_env("CAGE4", create_env)

            # Load checkpoint
            self.algo = APPO.from_checkpoint(abs_path)
            print("✅ Successfully loaded APPO model")

        except Exception as e:
            print(f"⚠️  Could not load APPO model: {e}")
            self.algo = None

    def get_action(self, observation, action_space):
        """Get action from trained model or fallback"""
        self.step_count += 1

        if self.algo is not None:
            try:
                action = self.algo.compute_single_action(
                    observation,
                    policy_id=self.policy_id,
                    explore=False
                )

                if self.step_count <= 5:
                    print(f"✅ {self.name} using TRAINED APPO MODEL (policy: {self.policy_id})")

                return int(action)

            except Exception as e:
                if self.step_count <= 5:
                    print(f"⚠️  {self.name} inference failed: {e}")
                return self._fallback_action(action_space)
        else:
            if self.step_count <= 5:
                print(f"⚠️  {self.name} no model loaded, using fallback")
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
