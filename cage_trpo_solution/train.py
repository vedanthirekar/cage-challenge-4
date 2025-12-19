"""
Training script for CAGE Challenge 4 using SB3-Contrib TRPO
TRPO (Trust Region Policy Optimization) - more conservative than PPO
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import numpy as np
from collections import deque

from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper

from sb3_contrib import TRPO
import gymnasium as gym
from gymnasium import spaces


class MultiAgentWrapper(gym.Env):
    """
    Wraps multi-agent environment for single-agent SB3 training.
    Collects observations from all agents and outputs combined actions.
    """

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.agents = env.agents
        self.num_agents = len(self.agents)

        # Get observation and action spaces
        self.obs_space_small = env.observation_space("blue_agent_0")
        self.act_space_small = env.action_space("blue_agent_0")
        self.obs_space_large = env.observation_space("blue_agent_4")
        self.act_space_large = env.action_space("blue_agent_4")

        # Combined observation space: concatenate all agent observations
        obs_dims = (
            self.obs_space_small.shape[0] * 4 +  # Agents 0-3
            self.obs_space_large.shape[0]         # Agent 4
        )
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(obs_dims,), dtype=np.int32
        )

        # Combined action space: tuple of discrete actions
        self.action_space = spaces.MultiDiscrete(
            [self.act_space_small.n] * 4 + [self.act_space_large.n]
        )

        self.episode_rewards = deque(maxlen=100)
        self.episode_reward = 0.0

    def reset(self, seed=None, options=None):
        """Reset environment and return combined observation"""
        obs_dict, info = self.env.reset(seed=seed)
        self.episode_reward = 0.0
        return self._combine_observations(obs_dict), {}

    def step(self, actions):
        """Take step with combined actions"""
        action_dict = {
            agent: int(actions[i])
            for i, agent in enumerate(self.agents)
        }

        obs_dict, rewards_dict, terminated_dict, truncated_dict, info_dict = (
            self.env.step(action_dict)
        )

        combined_reward = sum(rewards_dict.values())
        self.episode_reward += combined_reward

        done = any(terminated_dict.values()) or any(truncated_dict.values())

        if done:
            self.episode_rewards.append(self.episode_reward)

        combined_obs = self._combine_observations(obs_dict)

        return combined_obs, combined_reward, done, False, {}

    def _combine_observations(self, obs_dict):
        """Concatenate observations from all agents"""
        obs_list = []
        for agent in self.agents:
            obs = obs_dict[agent]
            if isinstance(obs, np.ndarray):
                obs_list.append(obs)
            else:
                obs_list.append(np.array(obs))
        return np.concatenate(obs_list).astype(np.int32)

    def get_episode_rewards(self):
        """Get average episode reward"""
        if len(self.episode_rewards) == 0:
            return 0.0
        return np.mean(self.episode_rewards)


def create_env():
    """Create CAGE Challenge 4 environment"""
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500,
    )
    cyborg = CybORG(scenario_generator=sg)
    cyborg = BlueFlatWrapper(cyborg)
    return MultiAgentWrapper(cyborg)


def train_trpo(total_timesteps=100000, checkpoint_dir="cage_trpo_solution/trained_model"):
    """Train TRPO agent on CAGE Challenge 4"""

    print("\n" + "=" * 60)
    print("CAGE Challenge 4 - TRPO Training")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    # Create environment
    env = create_env()
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create TRPO model
    model = TRPO(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        n_steps=2048,  # TRPO typically uses larger batches
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        cg_max_steps=15,  # Conjugate gradient steps
        cg_damping=0.1,   # Damping for numerical stability
        line_search_shrinking_factor=0.8,
        line_search_max_iter=10,
        target_kl=0.01,   # Trust region constraint
        verbose=0,
    )

    print("\nModel created. Starting training...")
    print("-" * 60)

    # Train with periodic checkpointing
    checkpoint_interval = 10000
    best_reward = float('-inf')

    for step in range(0, total_timesteps, checkpoint_interval):
        remaining = min(checkpoint_interval, total_timesteps - step)

        # Train
        model.learn(total_timesteps=remaining, reset_num_timesteps=False)

        # Get average reward
        avg_reward = env.get_episode_rewards()

        # Log progress
        if avg_reward > best_reward:
            best_reward = avg_reward
            print(f"[{step + remaining:7d}/{total_timesteps}] Reward: {avg_reward:7.1f} ✓")
        else:
            print(f"[{step + remaining:7d}/{total_timesteps}] Reward: {avg_reward:7.1f}")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"model_{step + remaining}")
        model.save(checkpoint_path)

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "model_final")
    model.save(final_model_path)

    print("-" * 60)
    print(f"✅ Training complete!")
    print(f"Best reward: {best_reward:.1f}")
    print(f"Final model saved to: {final_model_path}")

    env.close()
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train TRPO on CAGE Challenge 4")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="cage_trpo_solution/trained_model",
        help="Checkpoint directory",
    )

    args = parser.parse_args()

    model = train_trpo(
        total_timesteps=args.timesteps,
        checkpoint_dir=args.checkpoint_dir,
    )
