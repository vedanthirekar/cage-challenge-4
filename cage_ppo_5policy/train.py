"""
CAGE Challenge 4 - PPO with 5 Separate Policies
Each agent gets its own dedicated policy for maximum specialization
"""

import os
import warnings
warnings.filterwarnings("ignore")

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env

from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import EnterpriseMAE


def env_creator(config):
    """Create the CAGE Challenge 4 environment"""
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500,
    )
    cyborg = CybORG(scenario_generator=sg)
    return EnterpriseMAE(cyborg)


def policy_mapping(agent_id, episode, worker, **kwargs):
    """Map each agent to its own dedicated policy"""
    return f"policy_{agent_id}"


def main(iterations=50):
    # Initialize Ray
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    
    # Register environment
    register_env("CybORG_CAGE4", env_creator)
    
    # Create temp env to get observation/action spaces
    temp_env = env_creator({})
    
    # Create 5 separate policies - one for each agent
    policies = {}
    for agent_id in temp_env.agents:
        policies[f"policy_{agent_id}"] = PolicySpec(
            observation_space=temp_env.observation_space(agent_id),
            action_space=temp_env.action_space(agent_id),
        )
    
    print(f"Created {len(policies)} policies:")
    for name, spec in policies.items():
        print(f"  {name}: obs={spec.observation_space.shape}, act={spec.action_space.n}")
    
    # PPO Configuration
    config = (
        PPOConfig()
        .environment(env="CybORG_CAGE4")
        .framework("torch")
        .env_runners(
            num_env_runners=4,
            rollout_fragment_length=500,
            explore=True,
        )
        .reporting(
            min_sample_timesteps_per_iteration=1000,
        )
        .training(
            # Lower LR to reduce overfitting
            lr=1e-4,
            gamma=0.99,
            lambda_=0.95,
            # Tighter clipping to prevent large policy changes
            clip_param=0.1,
            # Higher entropy for more exploration (key for generalization)
            entropy_coeff=0.1,
            vf_loss_coeff=0.5,
            # Larger batch for more stable gradients
            train_batch_size=8000,
            minibatch_size=256,
            # Fewer epochs to prevent overfitting
            num_epochs=5,
            grad_clip=0.5,
            # Stricter KL constraint
            kl_coeff=0.5,
            kl_target=0.005,
            # Smaller network = less capacity to overfit
            model={
                "fcnet_hiddens": [128, 128],
                "fcnet_activation": "relu",
                "vf_share_layers": False,
            },
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping,
        )
        .resources(num_gpus=0)
        .debugging(log_level="WARN")
    )
    
    # Build algorithm
    algo = config.build()
    
    print(f"\nTraining PPO with 5 separate policies for {iterations} iterations...")
    print("-" * 60)
    
    best_reward = float("-inf")
    
    for i in range(iterations):
        result = algo.train()
        
        reward = result.get("env_runners", {}).get("episode_reward_mean", 0)
        
        if reward > best_reward:
            best_reward = reward
            marker = " ✓"
        else:
            marker = ""
        
        print(f"[{i+1:3d}/{iterations}] Reward: {reward:8.1f}{marker}")
    
    # Save checkpoint
    save_dir = "cage_ppo_5policy/trained_model"
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = algo.save(save_dir)
    
    print("-" * 60)
    print(f"Best reward: {best_reward:.1f}")
    print(f"Saved to: {save_dir}")
    
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    main(args.iterations)
