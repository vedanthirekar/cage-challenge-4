"""
Submission file for CAGE Challenge 4 using RLlib APPO
"""

import os
from CybORG import CybORG
from CybORG.Agents import BaseAgent
from CybORG.Agents.Wrappers import EnterpriseMAE

from appo_agent import APPOAgent

# Get absolute path to trained model
SUBMISSION_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SUBMISSION_DIR, "trained_model")


class Submission:
    """
    CAGE Challenge 4 submission using APPO-trained agents
    """

    NAME: str = "RLlib APPO Multi-Agent Defense"
    TEAM: str = "AI Cyber Defense Team"
    TECHNIQUE: str = "APPO (Asynchronous Proximal Policy Optimization)"

    AGENTS: dict[str, BaseAgent] = {
        "blue_agent_0": APPOAgent(checkpoint_path=MODEL_PATH, name="blue_agent_0"),
        "blue_agent_1": APPOAgent(checkpoint_path=MODEL_PATH, name="blue_agent_1"),
        "blue_agent_2": APPOAgent(checkpoint_path=MODEL_PATH, name="blue_agent_2"),
        "blue_agent_3": APPOAgent(checkpoint_path=MODEL_PATH, name="blue_agent_3"),
        "blue_agent_4": APPOAgent(checkpoint_path=MODEL_PATH, name="blue_agent_4"),
    }

    @staticmethod
    def wrap(env: CybORG) -> EnterpriseMAE:
        """Wrap environment with EnterpriseMAE"""
        return EnterpriseMAE(env)
