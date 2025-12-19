"""
Submission file for CAGE Challenge 4 using RLlib IMPALA
"""

import os
from CybORG import CybORG
from CybORG.Agents import BaseAgent
from CybORG.Agents.Wrappers import EnterpriseMAE

from impala_agent import IMPALAAgent

# Get absolute path to trained model
SUBMISSION_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SUBMISSION_DIR, "trained_model")


class Submission:
    """
    CAGE Challenge 4 submission using IMPALA-trained agents
    """

    NAME: str = "RLlib IMPALA Multi-Agent Defense"
    TEAM: str = "AI Cyber Defense Team"
    TECHNIQUE: str = "IMPALA (Importance Weighted Actor-Learner Architecture)"

    AGENTS: dict[str, BaseAgent] = {
        "blue_agent_0": IMPALAAgent(checkpoint_path=MODEL_PATH, name="blue_agent_0"),
        "blue_agent_1": IMPALAAgent(checkpoint_path=MODEL_PATH, name="blue_agent_1"),
        "blue_agent_2": IMPALAAgent(checkpoint_path=MODEL_PATH, name="blue_agent_2"),
        "blue_agent_3": IMPALAAgent(checkpoint_path=MODEL_PATH, name="blue_agent_3"),
        "blue_agent_4": IMPALAAgent(checkpoint_path=MODEL_PATH, name="blue_agent_4"),
    }

    @staticmethod
    def wrap(env: CybORG) -> EnterpriseMAE:
        """Wrap environment with EnterpriseMAE"""
        return EnterpriseMAE(env)
