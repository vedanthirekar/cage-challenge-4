"""
Training script for CAGE Challenge 4 using SB3 A2C with PettingZoo
Based on official TrainingSB3.py example
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import time
import torch as T

from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper

import pettingzoo
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec
from gymnasium.spaces import Space

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env


class GenericPzShim(ParallelEnv):
    """Wrapper to make CybORG compatible with PettingZoo"""
    
    metadata = {
        "render_modes": [],
        "name": "CybORG v4 - CAGE Challenge 4",
        "is_parallelizable": True,
        "has_manual_policy": False,
    }

    def __init__(self, env: BlueFlatWrapper, **kwargs):
        super().__init__()
        self.env = env

    def reset(self, seed: int | None = None, *args, **kwargs):
        return self.env.reset(seed=seed)

    def step(self, *args, **kwargs):
        return self.env.step(*args, **kwargs)

    @property
    def agents(self):
        return self.env.agents

    @property
    def possible_agents(self):
        return self.env.possible_agents

    @property
    def action_spaces(self) -> dict[str, Space]:
        return self.env.action_spaces()

    def action_space(self, agent_name) -> Space:
        return self.env.action_space(agent_name)

    @property
    def observation_spaces(self) -> dict[str, Space]:
        return self.env.observation_spaces()

    def observation_space(self, agent_name) -> Space:
        return self.env.observation_space(agent_name)

    @property
    def action_masks(self) -> dict[str, T.tensor]:
        return {
            a: T.tensor(self.env.action_masks[a], dtype=T.bool) 
            for a in self.env.agents
        }


import gymnasium as gym


class SB3Wrapper(gym.Env):
    """Wrapper to convert PettingZoo AEC environment to Gymnasium for SB3"""

    def __init__(self, aec_env):
        super().__init__()
        self.aec_env = aec_env
        self.aec_env.reset()
        
        # Set observation and action spaces from first agent
        self.observation_space = self.aec_env.observation_space(self.aec_env.agent_selection)
        self.action_space = self.aec_env.action_space(self.aec_env.agent_selection)

    def reset(self, seed=None, options=None):
        self.aec_env.reset(seed=seed)
        obs = self.aec_env.observe(self.aec_env.agent_selection)
        return obs, {}

    def step(self, action):
        self.aec_env.step(action)
        obs, reward, terminated, truncated, info = self.aec_env.last()
        return obs, reward, terminated, truncated, info


class ProgressCallback(BaseCallback):
    """Callback to log training progress"""
    
    def __init__(self, check_freq: int, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_reward = 0
        
    def _on_step(self) -> bool:
        # Track episode rewards
        self.episode_reward += self.locals.get('rewards', [0])[0]
        
        if self.locals.get('dones', [False])[0]:
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0
            
        # Log progress
        if self.n_calls % self.check_freq == 0:
            if len(self.episode_rewards) > 0:
                avg_reward = sum(self.episode_rewards[-10:]) / min(10, len(self.episode_rewards))
                print(f"[{self.n_calls:7d}] Avg reward (last 10 eps): {avg_reward:7.1f}")
        
        return True


def create_env(**kwargs):
    """Create CAGE Challenge 4 environment with PettingZoo wrapper"""
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500,
    )
    cyborg = CybORG(scenario_generator=sg)
    cyborg = BlueFlatWrapper(cyborg, **kwargs)
    cyborg = GenericPzShim(cyborg)
    return cyborg


def train_a2c(
    total_timesteps=100000,
    checkpoint_dir="cage_a2c_sb3_sol/trained_model",
    seed=0,
    pad_spaces=True
):
    """Train A2C agent on CAGE Challenge 4"""

    print("\n" + "=" * 60)
    print("CAGE Challenge 4 - A2C Training (SB3 + PettingZoo)")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Pad spaces: {pad_spaces}")

    # Create environment
    env = create_env(pad_spaces=pad_spaces)
    
    # Convert to AEC (turn-based) for SB3
    aec_env = parallel_to_aec(env)
    
    # Wrap for SB3 compatibility
    env = SB3Wrapper(aec_env)

    print(f"\nObservation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create A2C model
    logdir = os.path.join(checkpoint_dir, "logs")
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=7e-4,
        n_steps=5,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_rms_prop=True,
        normalize_advantage=False,
        verbose=1,
        tensorboard_log=logdir,
    )
    model.set_random_seed(seed)

    print("\nModel created. Starting training...")
    print("-" * 60)

    # Create callback for progress logging
    callback = ProgressCallback(check_freq=10000)

    # Train
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save final model
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    final_model_path = os.path.join(checkpoint_dir, f"model_final_{timestamp}")
    model.save(final_model_path)

    # Also save as model_final for easy loading
    model.save(os.path.join(checkpoint_dir, "model_final"))

    print("-" * 60)
    print(f"✅ Training complete!")
    print(f"Model saved to: {final_model_path}")

    env.close()
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train A2C on CAGE Challenge 4"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="cage_a2c_sb3_sol/trained_model",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--pad-spaces",
        action="store_true",
        default=True,
        help="Pad observation/action spaces",
    )

    args = parser.parse_args()

    model = train_a2c(
        total_timesteps=args.timesteps,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        pad_spaces=args.pad_spaces,
    )
