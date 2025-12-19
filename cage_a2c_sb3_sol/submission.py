"""
Submission file for CAGE Challenge 4 using SB3 A2C
"""

import os
from CybORG import CybORG
from CybORG.Agents import BaseAgent
from CybORG.Agents.Wrappers import BlueFlatWrapper

from a2c_agent import A2CAgent

# Get absolute path to trained model
SUBMISSION_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SUBMISSION_DIR, "trained_model", "model_final")


class Submission:
    """
    CAGE Challenge 4 submission using A2C-trained agents
    """

    NAME: str = "SB3 A2C Multi-Agent Defense (PettingZoo)"
    TEAM: str = "AI Cyber Defense Team"
    TECHNIQUE: str = "A2C (Advantage Actor-Critic)"

    AGENTS: dict[str, BaseAgent] = {
        "blue_agent_0": A2CAgent(checkpoint_path=MODEL_PATH, name="blue_agent_0"),
        "blue_agent_1": A2CAgent(checkpoint_path=MODEL_PATH, name="blue_agent_1"),
        "blue_agent_2": A2CAgent(checkpoint_path=MODEL_PATH, name="blue_agent_2"),
        "blue_agent_3": A2CAgent(checkpoint_path=MODEL_PATH, name="blue_agent_3"),
        "blue_agent_4": A2CAgent(checkpoint_path=MODEL_PATH, name="blue_agent_4"),
    }

    @staticmethod
    def wrap(env: CybORG) -> BlueFlatWrapper:
        """Wrap environment with BlueFlatWrapper"""
        return BlueFlatWrapper(env)
