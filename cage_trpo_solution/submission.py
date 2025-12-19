"""
Submission file for CAGE Challenge 4 using SB3-Contrib TRPO
"""

import os
from CybORG import CybORG
from CybORG.Agents import BaseAgent
from CybORG.Agents.Wrappers import BlueFlatWrapper

from trpo_agent import TRPOAgent

# Get absolute path to trained model
SUBMISSION_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SUBMISSION_DIR, "trained_model", "model_final")


class Submission:
    """
    CAGE Challenge 4 submission using TRPO-trained agents
    """

    NAME: str = "SB3-Contrib TRPO Multi-Agent Defense"
    TEAM: str = "AI Cyber Defense Team"
    TECHNIQUE: str = "TRPO (Trust Region Policy Optimization)"

    AGENTS: dict[str, BaseAgent] = {
        "blue_agent_0": TRPOAgent(checkpoint_path=MODEL_PATH, name="blue_agent_0"),
        "blue_agent_1": TRPOAgent(checkpoint_path=MODEL_PATH, name="blue_agent_1"),
        "blue_agent_2": TRPOAgent(checkpoint_path=MODEL_PATH, name="blue_agent_2"),
        "blue_agent_3": TRPOAgent(checkpoint_path=MODEL_PATH, name="blue_agent_3"),
        "blue_agent_4": TRPOAgent(checkpoint_path=MODEL_PATH, name="blue_agent_4"),
    }

    @staticmethod
    def wrap(env: CybORG) -> BlueFlatWrapper:
        """Wrap environment with BlueFlatWrapper"""
        return BlueFlatWrapper(env)
