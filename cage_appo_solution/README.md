# CAGE Challenge 4 - APPO Solution

APPO (Asynchronous Proximal Policy Optimization) solution using RLlib.

## What is APPO?

APPO is a distributed, asynchronous version of PPO that:
- Uses multiple workers collecting experience in parallel
- Applies V-trace for off-policy correction
- More sample-efficient than synchronous PPO
- Better for environments with variable episode lengths

## Key Differences from PPO

**PPO (Synchronous):**
- Workers collect experience synchronously
- All workers must finish before training
- On-policy learning only

**APPO (Asynchronous):**
- Workers collect experience asynchronously
- Training happens while workers are still collecting
- V-trace enables off-policy learning
- Faster training, more sample-efficient

## Architecture

- **2 separate policies**: policy_small (agents 0-3) and policy_large (agent 4)
- **4 parallel workers** for distributed experience collection
- **V-trace** for off-policy correction
- **Network**: [256, 256, 128] fully connected layers

## Training

```bash
python cage_appo_solution/train.py --iterations 25
```

Options:
- `--iterations`: Number of training iterations (default: 25)
- `--episode-length`: Episode length (default: 500)
- `--checkpoint-dir`: Directory to save models

## Evaluation

```bash
python CybORG/Evaluation/evaluation.py --max-eps 10 cage_appo_solution/ results_appo
```

## Expected Performance

APPO should perform similarly to or better than PPO:
- **PPO baseline**: -2,300
- **APPO target**: -2,000 to -2,500

APPO's asynchronous nature and V-trace correction may provide:
- Faster training convergence
- Better sample efficiency
- More stable learning

## Comparison with Other Algorithms

| Algorithm | Type | Multi-Agent | Performance |
|-----------|------|-------------|-------------|
| PPO (RLlib) | On-policy, sync | ✅ Native | -2,300 |
| APPO (RLlib) | Off-policy, async | ✅ Native | TBD |
| A2C (SB3) | On-policy, sync | ❌ Wrapper | -5,265 |
| DQN (RLlib) | Off-policy | ✅ Native | Failed |
