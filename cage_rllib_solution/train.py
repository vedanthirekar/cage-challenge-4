"""
Training script for CAGE Challenge 4 using RLlib PPO
"""

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env

from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import EnterpriseMAE

import warnings
warnings.filterwarnings("ignore")


def create_env(env_config):
    """Create CAGE Challenge 4 environment"""
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=env_config.get('episode_length', 500),
    )
    cyborg = CybORG(scenario_generator=sg)
    return EnterpriseMAE(cyborg)


def train_agents(num_iterations=25, episode_length=500, checkpoint_dir="cage_rllib_solution/trained_model"):
    """
    Train CAGE Challenge 4 agents using PPO
    """
    print("=" * 60)
    print("CAGE Challenge 4 - RLlib Training")
    print("=" * 60)
    print(f"Episodes: {num_iterations}")
    print(f"Episode length: {episode_length}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)
    
    # Register environment
    register_env("CAGE4", create_env)
    
    # Create environment to get spaces
    env = create_env({'episode_length': episode_length})
    
    print(f"Environment agents: {env.agents}")
    print(f"Action spaces: {[env.action_space(agent).n for agent in env.agents]}")
    print(f"Observation spaces: {[env.observation_space(agent).shape for agent in env.agents]}")
    
    # Policy mapping function
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == "blue_agent_4":
            return "policy_large"  # Agent 4 has different spaces
        else:
            return "policy_small"  # Agents 0-3 have same spaces
    
    # PPO configuration
    config = (
        PPOConfig()
        .framework("torch")
        .environment(
            env="CAGE4",
            env_config={'episode_length': episode_length}
        )
        .training(
            lr=3e-4,
            gamma=0.9,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            train_batch_size=2000,
            minibatch_size=64,
            num_epochs=10,
            model={
                "fcnet_hiddens": [128, 128],
                "fcnet_activation": "relu",
            }
        )
        .env_runners(
            num_env_runners=1,
            rollout_fragment_length=200,
        )
        .multi_agent(
            policies={
                "policy_small": PolicySpec(
                    observation_space=env.observation_space("blue_agent_0"),
                    action_space=env.action_space("blue_agent_0"),
                ),
                "policy_large": PolicySpec(
                    observation_space=env.observation_space("blue_agent_4"),
                    action_space=env.action_space("blue_agent_4"),
                )
            },
            policy_mapping_fn=policy_mapping_fn,
        )
        .debugging(log_level="WARN")
        .resources(num_gpus=0)
    )
    
    # Build and train
    algo = config.build()
    
    print("\nStarting training...")
    best_reward = float('-inf')
    
    for i in range(num_iterations):
        try:
            result = algo.train()
            
            # Get reward information from env_runners (where actual rewards are stored)
            env_runners = result.get("env_runners", {})
            episode_reward_mean = env_runners.get("episode_reward_mean", 0)
            episode_reward_min = env_runners.get("episode_reward_min", 0)
            episode_reward_max = env_runners.get("episode_reward_max", 0)
            episodes_this_iter = env_runners.get("episodes_this_iter", 0)
            
            print(f"Iteration {i+1:2d}/{num_iterations}: "
                  f"Mean={episode_reward_mean:7.1f}, "
                  f"Min={episode_reward_min:7.1f}, "
                  f"Max={episode_reward_max:7.1f}, "
                  f"Episodes={episodes_this_iter}")
            
            if episode_reward_mean > best_reward:
                best_reward = episode_reward_mean
                print(f"  → New best: {best_reward:.1f}")
            
        except Exception as e:
            print(f"Training iteration {i+1} failed: {e}")
            continue
    
    # Save model
    checkpoint_path = algo.save(checkpoint_dir)
    print(f"\nTraining completed!")
    print(f"Best reward: {best_reward:.1f}")
    print(f"Model saved to: {checkpoint_path}")
    
    # Cleanup
    algo.stop()
    ray.shutdown()
    
    return checkpoint_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CAGE Challenge 4 agents")
    parser.add_argument("--iterations", type=int, default=20, help="Training iterations")
    parser.add_argument("--episode-length", type=int, default=200, help="Episode length")
    parser.add_argument("--checkpoint-dir", type=str, default="cage_rllib_solution/trained_model", help="Checkpoint directory")
    
    args = parser.parse_args()
    
    checkpoint_path = train_agents(
        num_iterations=args.iterations,
        episode_length=args.episode_length,
        checkpoint_dir=args.checkpoint_dir
    )
    
    print(f"\n✅ Training completed! Model saved at: {checkpoint_path}")