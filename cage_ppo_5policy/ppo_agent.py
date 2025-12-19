"""
PPO Agent with 5 Separate Policies for CAGE Challenge 4
"""

import os
from CybORG.Agents import BaseAgent


class PPOAgent(BaseAgent):
    """Agent using PPO with dedicated policy per agent"""
    
    _algo = None
    _checkpoint = None
    
    def __init__(self, checkpoint_path: str = None, name: str = None):
        super().__init__(name)
        self.checkpoint_path = checkpoint_path
        self.policy_id = f"policy_{name}"
        self.step_count = 0
        
        # Load algorithm once (shared across all agent instances)
        if checkpoint_path and os.path.exists(checkpoint_path):
            if PPOAgent._algo is None or PPOAgent._checkpoint != checkpoint_path:
                self._load(checkpoint_path)
                PPOAgent._algo = self.algo
                PPOAgent._checkpoint = checkpoint_path
            else:
                self.algo = PPOAgent._algo
        else:
            self.algo = None
            print(f"⚠️ No checkpoint at: {checkpoint_path}")
    
    def _load(self, path):
        """Load the trained algorithm"""
        import ray
        from ray.rllib.algorithms.ppo import PPO
        from ray.tune import register_env
        from CybORG import CybORG
        from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
        from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
        from CybORG.Agents.Wrappers import EnterpriseMAE
        
        print(f"🔍 Loading PPO from: {os.path.abspath(path)}")
        
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False)
        
        def env_creator(config):
            sg = EnterpriseScenarioGenerator(
                blue_agent_class=SleepAgent,
                green_agent_class=EnterpriseGreenAgent,
                red_agent_class=FiniteStateRedAgent,
                steps=500,
            )
            return EnterpriseMAE(CybORG(scenario_generator=sg))
        
        register_env("CybORG_CAGE4", env_creator)
        
        try:
            self.algo = PPO.from_checkpoint(path)
            print("✅ Loaded PPO model")
        except Exception as e:
            print(f"❌ Failed to load: {e}")
            self.algo = None
    
    def get_action(self, observation, action_space):
        """Get action from the agent's dedicated policy"""
        self.step_count += 1
        
        if self.algo:
            try:
                action = self.algo.compute_single_action(
                    observation,
                    policy_id=self.policy_id,
                    explore=False
                )
                if self.step_count <= 3:
                    print(f"✅ {self.name} using {self.policy_id}")
                return int(action)
            except Exception as e:
                if self.step_count <= 3:
                    print(f"⚠️ {self.name} error: {e}")
        
        # Fallback
        return 0
    
    def end_episode(self):
        pass
    
    def set_initial_values(self, action_space, observation):
        pass
    
    def train(self, results):
        pass
