"""
Submission file for CAGE Challenge 4 using RLlib DQN
"""

import os
from CybORG import CybORG
from CybORG.Agents import BaseAgent
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from CybORG.Agents.Wrappers.EnterpriseMAE import EnterpriseMAE

from dqn_agent import DQNAgent

# Get absolute path to trained model
SUBMISSION_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SUBMISSION_DIR, "trained_model")


class Submission:
    """
    CAGE Challenge 4 submission using DQN-trained agents
    """

    NAME: str = "RLlib DQN Multi-Agent Defense"
    TEAM: str = "AI Cyber Defense Team"
    TECHNIQUE: str = "Multi-Agent DQN"

    AGENTS: dict[str, BaseAgent] = {
        "blue_agent_0": DQNAgent(
            checkpoint_path=MODEL_PATH,
            policy_id="policy_small",
            name="blue_agent_0"
        ),
        "blue_agent_1": DQNAgent(
            checkpoint_path=MODEL_PATH,
            policy_id="policy_small",
            name="blue_agent_1"
        ),
        "blue_agent_2": DQNAgent(
            checkpoint_path=MODEL_PATH,
            policy_id="policy_small",
            name="blue_agent_2"
        ),
        "blue_agent_3": DQNAgent(
            checkpoint_path=MODEL_PATH,
            policy_id="policy_small",
            name="blue_agent_3"
        ),
        "blue_agent_4": DQNAgent(
            checkpoint_path=MODEL_PATH,
            policy_id="policy_large",
            name="blue_agent_4"
        ),
    }

    @staticmethod
    def wrap(env: CybORG) -> MultiAgentEnv:
        """Wrap environment with EnterpriseMAE"""
        return EnterpriseMAE(env)
