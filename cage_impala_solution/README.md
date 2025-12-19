# CAGE Challenge 4 - IMPALA Solution

IMPALA (Importance Weighted Actor-Learner Architecture) solution using RLlib.

## What is IMPALA?

IMPALA is a distributed RL algorithm designed for scalability:
- **Decoupled acting and learning** - Actors collect experience, learners train
- **V-trace** for off-policy correction (importance sampling)
- **High throughput** - Can handle thousands of actors
- **Better than APPO** for multi-agent coordination

## Key Features

**Architecture:**
- Separate actor and learner processes
- Actors collect experience asynchronously
- Learner trains on batched experience with V-trace correction

**Multi-Agent:**
- 2 separate policies: policy_small (agents 0-3) and policy_large (agent 4)
- 4 parallel workers for distributed experience collection
- V-trace handles off-policy data from multiple actors

**Network:**
- [256, 256, 128] fully connected layers
- Separate value function network

## IMPALA vs PPO vs APPO

| Feature | PPO | APPO | IMPALA |
|---------|-----|------|--------|
| Learning | On-policy | Off-policy | Off-policy |
| Synchronization | Sync | Async | Decoupled |
| Actors/Learners | Combined | Combined | Separate |
| Throughput | Medium | High | Very High |
| Multi-Agent | ✅ Best | ❌ Poor | ✅ Good |

**Why IMPALA might work better than APPO:**
- Decoupled architecture reduces coordination issues
- V-trace is more stable than APPO's approach
- Better suited for multi-agent environments

## Training

```bash
python cage_impala_solution/train.py --iterations 50
```

Options:
- `--iterations`: Number of training iterations (default: 25)
- `--episode-length`: Episode length (default: 500)
- `--checkpoint-dir`: Directory to save models

## Evaluation

```bash
python CybORG/Evaluation/evaluation.py --max-eps 10 cage_impala_solution/ results_impala
```

## Expected Performance

**Target:** -2,500 to -3,500

**Comparison:**
- PPO: -2,300 (best)
- IMPALA: TBD (should be competitive)
- APPO: -5,660 (failed)
- A2C: -5,265 (single policy limitation)

IMPALA should perform better than APPO because:
1. Decoupled actors/learners reduce coordination issues
2. More stable V-trace implementation
3. Better suited for distributed multi-agent learning

## Algorithm Comparison Summary

**On-Policy (Synchronous):**
- ✅ PPO: -2,300 - Best for tight coordination

**Off-Policy (Distributed):**
- ✅ IMPALA: TBD - Decoupled architecture
- ❌ APPO: -5,660 - Async updates hurt coordination

**Single-Agent Adapted:**
- ❌ A2C: -5,265 - Single shared policy limitation
