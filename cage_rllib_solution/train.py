"""
Training script for CAGE Challenge 4 using RLlib PPO
"""

import os
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env

from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

import warnings
warnings.filterwarnings("ignore")


def create_env(env_config):
    """Create CAGE Challenge 4 environment"""
    from CybORG.Agents.Wrappers import EnterpriseMAE

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
    print(f"\n🎯 Training: {num_iterations} iterations, episode length: {episode_length}")
    
    # Initialize Ray with increased timeout for Windows
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True, 
            log_to_driver=False,
            num_cpus=2,  # Limit CPUs to avoid resource issues
            object_store_memory=500_000_000,  # 500MB
            _system_config={
                "object_timeout_milliseconds": 200000,
            }
        )
    
    # Register environment
    register_env("CAGE4", create_env)
    
    # Create environment to get spaces
    env = create_env({'episode_length': episode_length})
    
    # Policy mapping function
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == "blue_agent_4":
            return "policy_large"  # Agent 4 has different spaces
        else:
            return "policy_small"  # Agents 0-3 have same spaces
    
    # Improved PPO configuration
    config = (
        PPOConfig()
        .framework("torch")
        .environment(
            env="CAGE4",
            env_config={'episode_length': episode_length}
        )
        .training(
            # Learning rate with schedule for better convergence
            lr=5e-4,
            lr_schedule=[
                [0, 5e-4],
                [500000, 1e-4],
                [1000000, 5e-5],
            ],
            # Discount factor - higher for long-term planning
            gamma=0.99,
            lambda_=0.95,
            # PPO clipping
            clip_param=0.2,
            # Higher entropy for more exploration
            entropy_coeff=0.05,
            entropy_coeff_schedule=[
                [0, 0.05],
                [500000, 0.01],
            ],
            # Value function coefficient
            vf_loss_coeff=0.5,
            # Larger batch sizes for stable learning
            train_batch_size=4000,
            minibatch_size=128,
            num_epochs=10,
            # Gradient clipping
            grad_clip=0.5,
            # KL divergence target
            kl_coeff=0.2,
            kl_target=0.01,
            # Larger network for complex environment
            model={
                "fcnet_hiddens": [256, 256, 128],
                "fcnet_activation": "relu",
                "vf_share_layers": False,  # Separate value function network
            }
        )
        .env_runners(
            # More parallel workers for faster learning
            num_env_runners=2,
            rollout_fragment_length=250,
            # Enable exploration
            explore=True,
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
            # Train all policies
            policies_to_train=["policy_small", "policy_large"],
        )
        .debugging(log_level="WARN")
        .resources(num_gpus=0)
        .reporting(
            min_sample_timesteps_per_iteration=1000,
        )
    )
    
    # Build and train
    algo = config.build()
    
    # Try to restore from checkpoint if it exists
    abs_checkpoint_dir = os.path.abspath(checkpoint_dir)
    if os.path.exists(abs_checkpoint_dir) and os.path.isdir(abs_checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(abs_checkpoint_dir) if f.startswith('algorithm_state')]
        if checkpoint_files:
            try:
                algo.restore(abs_checkpoint_dir)
                print("📂 Resumed from checkpoint")
            except Exception as e:
                print("🆕 Starting fresh")
        else:
            print("🆕 Starting fresh")
    else:
        print("🆕 Starting fresh")
    
    best_reward = float('-inf')
    
    for i in range(num_iterations):
        try:
            result = algo.train()
            
            # Get reward information from env_runners
            env_runners = result.get("env_runners", {})
            episode_reward_mean = env_runners.get("episode_reward_mean", 0)
            
            # Track improvement
            if episode_reward_mean > best_reward:
                improvement = episode_reward_mean - best_reward
                best_reward = episode_reward_mean
                print(f"[{i+1:3d}/{num_iterations}] Reward: {episode_reward_mean:7.1f} ✓ (+{improvement:.1f})")
            else:
                print(f"[{i+1:3d}/{num_iterations}] Reward: {episode_reward_mean:7.1f}")
            
        except Exception as e:
            print(f"[{i+1}] Error: {e}")
            continue
    
    # Save final model as trained_model
    checkpoint_path = algo.save(checkpoint_dir)
    print(f"\n✅ Done! Best reward: {best_reward:.1f}")
    print(f"💾 Saved to: {checkpoint_dir}")
    
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
    
