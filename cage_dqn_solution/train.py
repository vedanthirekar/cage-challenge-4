"""
Training script for CAGE Challenge 4 using RLlib DQN
"""

import os
import warnings
warnings.filterwarnings("ignore")

import ray
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env

from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator


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


def train_agents(num_iterations=25, episode_length=500,
                 checkpoint_dir="cage_dqn_solution/trained_model"):
    """
    Train CAGE Challenge 4 agents using DQN
    """
    print(f"\n🎯 DQN Training: {num_iterations} iterations, "
          f"episode length: {episode_length}")

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            log_to_driver=False,
            num_cpus=6,
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
        return "policy_small"

    # DQN configuration - using old API stack for stability
    config = (
        DQNConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .framework("torch")
        .environment(
            env="CAGE4",
            env_config={'episode_length': episode_length}
        )
        .training(
            # Learning rate
            lr=1e-4,
            lr_schedule=[
                [0, 1e-4],
                [500000, 5e-5],
                [1000000, 1e-5],
            ],
            # Discount factor
            gamma=0.99,
            # Training batch
            train_batch_size=256,
            # Target network update frequency
            target_network_update_freq=2000,
            # Replay buffer
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": 100000,
            },
            # Warmup before training
            num_steps_sampled_before_learning_starts=5000,
            # Double DQN
            double_q=True,
            # Dueling DQN
            dueling=True,
            # N-step returns
            n_step=1,
            # Gradient clipping
            grad_clip=10.0,
            # Network architecture
            model={
                "fcnet_hiddens": [256, 256, 128],
                "fcnet_activation": "relu",
            },
        )
        .env_runners(
            num_env_runners=0,  # Single-threaded for checkpoint compatibility
            rollout_fragment_length=200,
        )
        .exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.1,
                "epsilon_timesteps": 300000,
            }
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

    # Build algorithm
    algo = config.build()

    # Try to restore from checkpoint
    abs_checkpoint_dir = os.path.abspath(checkpoint_dir)
    if os.path.exists(abs_checkpoint_dir) and os.path.isdir(abs_checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(abs_checkpoint_dir)
                           if f.startswith('algorithm_state')]
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

            # Get reward from env_runners (new API) or sampler_results (old API)
            env_runners = result.get("env_runners", result.get("sampler_results", {}))
            episode_reward_mean = env_runners.get("episode_reward_mean", 0)

            if episode_reward_mean > best_reward:
                improvement = episode_reward_mean - best_reward
                best_reward = episode_reward_mean
                print(f"[{i+1:3d}/{num_iterations}] Reward: "
                      f"{episode_reward_mean:7.1f} ✓ (+{improvement:.1f})")
            else:
                print(f"[{i+1:3d}/{num_iterations}] Reward: "
                      f"{episode_reward_mean:7.1f}")

        except Exception as e:
            print(f"[{i+1}] Error: {e}")
            continue

    # Save final model
    algo.save(checkpoint_dir)
    print(f"\n✅ Done! Best reward: {best_reward:.1f}")
    print(f"💾 Saved to: {checkpoint_dir}")

    algo.stop()
    ray.shutdown()

    return checkpoint_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train CAGE4 agents with DQN")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Training iterations")
    parser.add_argument("--episode-length", type=int, default=500,
                        help="Episode length (matches evaluation)")
    parser.add_argument("--checkpoint-dir", type=str,
                        default="cage_dqn_solution/trained_model",
                        help="Checkpoint directory")

    args = parser.parse_args()

    train_agents(
        num_iterations=args.iterations,
        episode_length=args.episode_length,
        checkpoint_dir=args.checkpoint_dir
    )
