# CAGE Challenge 4 - A2C Solution

Stable-Baselines3 A2C (Advantage Actor-Critic) solution for CAGE Challenge 4.

## Multi-Agent Handling

This solution handles the multi-agent environment by:
1. **Single Shared Policy**: One A2C model trained on combined observations from all 5 agents
2. **Observation Concatenation**: Combines observations from all agents (578 dims total)
3. **Action Decomposition**: Model outputs 5 actions simultaneously, one per agent
4. **Synchronized Inference**: Collects observations from all agents, then gets combined actions

## Files

- `train.py` - Training script
- `a2c_agent.py` - A2C agent class for inference
- `submission.py` - Official submission format
- `evaluate.py` - Evaluation script

## Training

```bash
# Train for 100k timesteps (default)
python cage_a2c_solution/train.py --timesteps 100000

# Train for more timesteps
python cage_a2c_solution/train.py --timesteps 200000
```

## Evaluation

```bash
# Using official evaluation
python CybORG/Evaluation/evaluation.py --max-eps 10 cage_a2c_solution/ results_a2c
```

## Architecture

- **Algorithm**: A2C (Advantage Actor-Critic)
- **Network**: MLP with default SB3 architecture
- **Learning Rate**: 5e-4
- **Entropy Coefficient**: 0.01 (for exploration)
- **Gamma**: 0.99 (discount factor)
- **N-steps**: 128 (before update)

## Why A2C?

- **Simpler than PPO**: Fewer hyperparameters to tune
- **Stable**: Actor-Critic architecture is proven
- **Efficient**: No replay buffer overhead like DQN
- **Multi-agent friendly**: Can handle combined observations/actions
