# CAGE Challenge 4 - A2C Solution (SB3 + PettingZoo)

A2C (Advantage Actor-Critic) solution using Stable-Baselines3 with PettingZoo integration.

## Key Features

- **Algorithm**: A2C (Advantage Actor-Critic)
- **Library**: Stable-Baselines3 + PettingZoo
- **Architecture**: Turn-based multi-agent using `parallel_to_aec` converter
- **Episode Length**: 500 steps (matches evaluation)

## Differences from Previous A2C Solution

**Old approach (cage_a2c_solution):**
- Custom MultiAgentWrapper
- Concatenated all observations (578 dims)
- Single shared policy for all 5 agents
- Result: -5,265 (poor performance)

**New approach (cage_a2c_sb3_sol):**
- Official PettingZoo integration (GenericPzShim + parallel_to_aec)
- Turn-based agent execution (AEC model)
- Each agent gets individual observations
- Follows official TrainingSB3.py example

## Training

```bash
python cage_a2c_sb3_sol/train.py --timesteps 100000
```

Options:
- `--timesteps`: Total training timesteps (default: 100000)
- `--checkpoint-dir`: Directory to save models (default: cage_a2c_sb3_sol/trained_model)
- `--seed`: Random seed (default: 0)
- `--pad-spaces`: Pad observation/action spaces (default: True)

## Evaluation

```bash
python CybORG/Evaluation/evaluation.py --max-eps 10 cage_a2c_sb3_sol/ results_a2c_sb3
```

## Architecture

1. **GenericPzShim**: Makes CybORG compatible with PettingZoo ParallelEnv
2. **parallel_to_aec**: Converts parallel environment to turn-based (AEC)
3. **SB3Wrapper**: Adapts PettingZoo AEC environment for SB3
4. **A2C**: Standard SB3 A2C algorithm

## Expected Performance

Should perform better than the old MultiAgentWrapper approach (-5,265) since:
- Uses official PettingZoo integration
- Proper turn-based execution
- Individual agent observations (not concatenated)
- Follows recommended best practices

Target: Beat -5,265 baseline
