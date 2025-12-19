"""
Reward Shaping Wrapper for CAGE Challenge 4

Adds positive rewards for:
1. Monitor detects malicious activity (+0.5)
2. Successfully restore host (+1.0)
3. Remove red session - red session count decreases (+2.0)
"""

from typing import Any, Dict, Tuple
import numpy as np

from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import EnterpriseMAE
from CybORG.Simulator.Actions.AbstractActions import Monitor, Restore, Remove


class RewardShapingMAE(EnterpriseMAE):
    """
    Wrapper that adds reward shaping on top of EnterpriseMAE.
    
    Additional rewards:
    - +0.5 for Monitor detecting malicious activity
    - +1.0 for successful Restore action
    - +2.0 for each red session removed
    """
    
    def __init__(self, env: CybORG, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self._prev_red_session_count = self._count_red_sessions()
        
        # Reward values (can be tuned)
        self.monitor_detect_reward = 0.5
        self.restore_success_reward = 1.0
        self.remove_session_reward = 2.0
    
    def _count_red_sessions(self) -> int:
        """Count total number of active red sessions across all red agents."""
        state = self.env.environment_controller.state
        total = 0
        for agent_name in state.sessions:
            if 'red' in agent_name:
                total += len([s for s in state.sessions[agent_name].values() if s.active])
        return total
    
    def reset(self, *args, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset and initialize red session tracking."""
        obs, info = super().reset(*args, **kwargs)
        self._prev_red_session_count = self._count_red_sessions()
        return obs, info
    
    def step(
        self,
        action_dict: Dict[str, Any] = None,
        messages: Dict[str, Any] = None,
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        """
        Take a step and add shaped rewards.
        """
        # Store actions before step (convert indices to Action objects)
        actions_taken = {}
        if action_dict:
            for agent, action_idx in action_dict.items():
                if isinstance(action_idx, int):
                    actions_taken[agent] = self._action_space[agent]["actions"][action_idx]
                else:
                    actions_taken[agent] = action_idx
        
        # Execute the step
        obs, rewards, terminated, truncated, info = super().step(
            action_dict=action_dict, messages=messages
        )
        
        # Calculate shaped rewards
        shaped_rewards = self._calculate_shaped_rewards(actions_taken, rewards)
        
        # Update red session count for next step
        self._prev_red_session_count = self._count_red_sessions()
        
        return obs, shaped_rewards, terminated, truncated, info
    
    def _calculate_shaped_rewards(
        self, 
        actions_taken: Dict[str, Any],
        base_rewards: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Add shaped rewards to base rewards.
        """
        shaped_rewards = dict(base_rewards)
        state = self.env.environment_controller.state
        
        # Track bonus for distribution
        total_bonus = 0.0
        
        # 1. Monitor detection reward
        for agent, action in actions_taken.items():
            if isinstance(action, Monitor):
                # Check if monitor found anything (malicious processes/connections)
                # We check if there were events detected
                for blue_agent in self.possible_agents:
                    sess = state.sessions.get(blue_agent, {}).get(0)
                    if sess and hasattr(sess, 'sus_pids') and sess.sus_pids:
                        # Monitor found suspicious activity
                        total_bonus += self.monitor_detect_reward
                        break
        
        # 2. Restore success reward
        for agent, action in actions_taken.items():
            if isinstance(action, Restore):
                # Restore always returns True if it executes
                # Check if the action was valid (not Sleep replacement)
                if hasattr(action, 'hostname') and action.hostname:
                    total_bonus += self.restore_success_reward
        
        # 3. Red session removal reward
        current_red_sessions = self._count_red_sessions()
        sessions_removed = self._prev_red_session_count - current_red_sessions
        if sessions_removed > 0:
            total_bonus += sessions_removed * self.remove_session_reward
        
        # Distribute bonus equally to all blue agents
        if total_bonus > 0:
            bonus_per_agent = total_bonus / len(shaped_rewards)
            for agent in shaped_rewards:
                shaped_rewards[agent] += bonus_per_agent
        
        return shaped_rewards


def create_shaped_env(env_config: dict):
    """Factory function to create reward-shaped environment."""
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=env_config.get('episode_length', 500),
    )
    cyborg = CybORG(scenario_generator=sg)
    return RewardShapingMAE(cyborg)
