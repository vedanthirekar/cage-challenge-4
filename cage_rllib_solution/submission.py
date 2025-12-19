"""
Submission file for CAGE Challenge 4 using RLlib
"""

import os
from CybORG import CybORG
from CybORG.Agents import BaseAgent
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from CybORG.Agents.Wrappers.EnterpriseMAE import EnterpriseMAE

from rllib_agent import RLlibAgent

# Get absolute path to trained model relative to this file
SUBMISSION_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SUBMISSION_DIR, "trained_model_shaped")


class Submission:
    """
    CAGE Challenge 4 submission using RLlib-trained agents
    """
    
    # Submission metadata
    NAME: str = "RLlib PPO Multi-Agent Defense"
    TEAM: str = "AI Cyber Defense Team"
    TECHNIQUE: str = "Multi-Agent PPO"
    
    # Define 5 blue agents with trained policies
    AGENTS: dict[str, BaseAgent] = {
        "blue_agent_0": RLlibAgent(
            checkpoint_path=MODEL_PATH,
            policy_id="policy_small",
            name="blue_agent_0"
        ),
        "blue_agent_1": RLlibAgent(
            checkpoint_path=MODEL_PATH,
            policy_id="policy_small",
            name="blue_agent_1"
        ),
        "blue_agent_2": RLlibAgent(
            checkpoint_path=MODEL_PATH,
            policy_id="policy_small",
            name="blue_agent_2"
        ),
        "blue_agent_3": RLlibAgent(
            checkpoint_path=MODEL_PATH,
            policy_id="policy_small",
            name="blue_agent_3"
        ),
        "blue_agent_4": RLlibAgent(
            checkpoint_path=MODEL_PATH,
            policy_id="policy_large",
            name="blue_agent_4"
        ),
    }
    
    @staticmethod
    def wrap(env: CybORG) -> MultiAgentEnv:
        """Wrap environment with EnterpriseMAE"""
        return EnterpriseMAE(env)