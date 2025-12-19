"""
Training script for CAGE Challenge 4 using RLlib IMPALA
IMPALA = Importance Weighted Actor-Learner Architecture (distributed RL)
"""

import os
import ray
from ray.rllib.algorithms.impala import IMPALAConfig
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


def train_agents(num_iterations=25, episode_length=500, checkpoint_dir="cage_impala_solution/trained_model"):
    """
    Train CAGE Challenge 4 agents using IMPALA
    """
    print(f"\n🎯 Training IMPALA: {num_iterations} iterations, episode length: {episode_length}")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True, 
            log_to_driver=False,
            num_cpus=4,
            object_store_memory=500_000_000,
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
            return "policy_large"
        else:
            return "policy_small"
    
    # IMPALA configuration
    config = (
        IMPALAConfig()
        .framework("torch")
        .environment(
            env="CAGE4",
            env_config={'episode_length': episode_length}
        )
        .training(
            # Learning rate
            lr=5e-4,
            # Discount factor
            gamma=0.99,
            # Entropy for exploration
            entropy_coeff=0.01,
            # Value function coefficient
            vf_loss_coeff=0.5,
            # Training batch size - larger for better learning
            train_batch_size=2000,
            # Gradient clipping
            grad_clip=40.0,
            # V-trace parameters (importance sampling correction)
            vtrace=True,
            vtrace_clip_rho_threshold=1.0,
            vtrace_clip_pg_rho_threshold=1.0,
            # Network architecture
            model={
                "fcnet_hiddens": [256, 256, 128],
                "fcnet_activation": "relu",
                "vf_share_layers": False,
            }
        )
        .env_runners(
            # More workers for distributed training
            num_env_runners=4,
            rollout_fragment_length=50,
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
    
    # Try to restore from checkpoint
    abs_checkpoint_dir = os.path.abspath(checkpoint_dir)
    if os.path.exists(abs_checkpoint_dir) and os.path.isdir(abs_checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(abs_checkpoint_dir) if f.startswith('algorithm_state')]
        if checkpoint_files:
            try:
                algo.restore(abs_checkpoint_dir)
                print("📂 Resumed from checkpoint")
            except Exception:
                print("🆕 Starting fresh")
        else:
            print("🆕 Starting fresh")
    else:
        print("🆕 Starting fresh")
    
    best_reward = float('-inf')
    
    for i in range(num_iterations):
        try:
            result = algo.train()
            
            # Get reward information
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
    
    # Save final model
    checkpoint_path = algo.save(checkpoint_dir)
    print(f"\n✅ Done! Best reward: {best_reward:.1f}")
    print(f"💾 Saved to: {checkpoint_dir}")
    
    # Cleanup
    algo.stop()
    ray.shutdown()
    
    return checkpoint_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CAGE Challenge 4 agents with IMPALA")
    parser.add_argument("--iterations", type=int, default=25, help="Training iterations")
    parser.add_argument("--episode-length", type=int, default=500, help="Episode length")
    parser.add_argument("--checkpoint-dir", type=str, default="cage_impala_solution/trained_model", help="Checkpoint directory")
    
    args = parser.parse_args()
    
    checkpoint_path = train_agents(
        num_iterations=args.iterations,
        episode_length=args.episode_length,
        checkpoint_dir=args.checkpoint_dir,
    )
