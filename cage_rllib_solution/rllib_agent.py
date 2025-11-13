"""
RLlib agent for CAGE Challenge 4
"""

import os
import numpy as np
from gym import Space
from CybORG.Agents import BaseAgent


class RLlibAgent(BaseAgent):
    """
    RLlib-trained agent for CAGE Challenge 4
    """
    
    def __init__(self, checkpoint_path: str = None, policy_id: str = "policy_small", name: str = None):
        super().__init__(name)
        self.checkpoint_path = checkpoint_path
        self.policy_id = policy_id
        self.algorithm = None
        self.step_count = 0
        self.fallback_logged = False
        
        # Try to load trained model if available
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_model(checkpoint_path)
    
    def _load_model(self, checkpoint_path: str):
        """Load trained RLlib model"""
        try:
            from ray.rllib.algorithms.ppo import PPO
            from ray.tune import register_env
            import ray
            
            # Initialize Ray if not already initialized
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, log_to_driver=False)
            
            # Register the CAGE4 environment (needed for model loading)
            def create_env(env_config):
                from CybORG import CybORG
                from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
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
            
            # Convert to absolute path to avoid URI issues
            abs_path = os.path.abspath(checkpoint_path)
            print(f"🔍 Attempting to load model from: {abs_path}")
            print(f"🔍 Path exists: {os.path.exists(abs_path)}")
            
            if os.path.exists(abs_path):
                self.algorithm = PPO.from_checkpoint(abs_path)
                print(f"✅ Successfully loaded trained model from {abs_path}")
            else:
                print(f"❌ Checkpoint path does not exist: {abs_path}")
                print(f"⚠️  {self.name} will use FALLBACK behavior")
                self.algorithm = None
                
        except Exception as e:
            print(f"⚠️  Could not load model from {checkpoint_path}: {e}")
            print(f"⚠️  Full error: {type(e).__name__}: {str(e)}")
            print(f"⚠️  {self.name} will use FALLBACK behavior")
            self.algorithm = None
    
    def get_action(self, observation, action_space):
        """Get action from trained model or intelligent fallback"""
        self.step_count += 1
        
        if self.algorithm is not None:
            try:
                # Use trained RLlib model
                obs_array = self._process_observation(observation)
                action = self.algorithm.compute_single_action(
                    obs_array, 
                    policy_id=self.policy_id
                )
                
                # Log first action to confirm model usage
                if self.step_count == 1:
                    print(f"✅ {self.name}: Using TRAINED MODEL")
                
                # Ensure action is valid
                if isinstance(action, (int, np.integer)):
                    return max(0, min(action, action_space.n - 1))
                else:
                    return int(action) % action_space.n
                    
            except Exception as e:
                if not self.fallback_logged:
                    print(f"⚠️  {self.name}: Model inference failed, using FALLBACK behavior")
                    self.fallback_logged = True
                return self._fallback_action(action_space)
        else:
            # Use intelligent fallback when no trained model
            if not self.fallback_logged:
                print(f"⚠️  {self.name}: No trained model loaded - using FALLBACK behavior")
                self.fallback_logged = True
            return self._fallback_action(action_space)
    
    def _process_observation(self, observation):
        """Process observation for RLlib"""
        if isinstance(observation, np.ndarray):
            # Ensure it's the right type and shape
            obs = observation.astype(np.int32)  # Keep as int for MultiDiscrete
            return obs
        elif isinstance(observation, dict):
            # Flatten observation dictionary
            obs_values = []
            for key, value in sorted(observation.items()):
                if isinstance(value, (int, float)):
                    obs_values.append(int(value))  # Keep as int
                elif isinstance(value, (list, np.ndarray)):
                    obs_values.extend([int(v) for v in np.array(value).flatten()])
                elif isinstance(value, dict):
                    for k, v in sorted(value.items()):
                        if isinstance(v, (int, float)):
                            obs_values.append(int(v))  # Keep as int
                        elif isinstance(v, (list, np.ndarray)):
                            obs_values.extend([int(x) for x in np.array(v).flatten()])
            return np.array(obs_values, dtype=np.int32)  # MultiDiscrete expects int
        else:
            return np.array([int(observation)], dtype=np.int32)
    
    def _fallback_action(self, action_space):
        """Intelligent action selection when no trained model is available"""
        # Cycle through defensive actions
        cycle = self.step_count % 15
        
        if cycle < 5:
            return 0  # Monitor
        elif cycle < 8:
            return 1  # Analyse
        elif cycle < 10:
            return 2  # Remove
        elif cycle < 12:
            return 4  # Deploy Decoy
        else:
            return 0  # Monitor (default)