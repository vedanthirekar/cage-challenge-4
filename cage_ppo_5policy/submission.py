"""
Submission for CAGE Challenge 4 - PPO with 5 Separate Policies
"""

import os
from CybORG import CybORG
from CybORG.Agents import BaseAgent
from CybORG.Agents.Wrappers import EnterpriseMAE

from ppo_agent import PPOAgent

SUBMISSION_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SUBMISSION_DIR, "trained_model")


class Submission:
    NAME: str = "PPO with 5 Separate Policies"
    TEAM: str = "AI Cyber Defense"
    TECHNIQUE: str = "PPO (5 dedicated policies)"
    
    AGENTS: dict[str, BaseAgent] = {
        "blue_agent_0": PPOAgent(checkpoint_path=MODEL_PATH, name="blue_agent_0"),
        "blue_agent_1": PPOAgent(checkpoint_path=MODEL_PATH, name="blue_agent_1"),
        "blue_agent_2": PPOAgent(checkpoint_path=MODEL_PATH, name="blue_agent_2"),
        "blue_agent_3": PPOAgent(checkpoint_path=MODEL_PATH, name="blue_agent_3"),
        "blue_agent_4": PPOAgent(checkpoint_path=MODEL_PATH, name="blue_agent_4"),
    }
    
    @staticmethod
    def wrap(env: CybORG) -> EnterpriseMAE:
        return EnterpriseMAE(env)
