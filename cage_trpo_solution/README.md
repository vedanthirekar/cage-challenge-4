# CAGE Challenge 4 - TRPO Solution

Trust Region Policy Optimization (TRPO) solution using SB3-Contrib for CAGE Challenge 4.

## Why TRPO?

TRPO is similar to PPO but with stronger theoretical guarantees:
- **Trust region constraint**: Prevents policy from changing too much
- **More conservative**: Safer updates than PPO
- **Monotonic improvement**: Guaranteed to not get worse (in theory)
- **Better for sensitive tasks**: When stability is critical

## Files

- `train.py` - Training script
- `trpo_agent.py` - TRPO agent class for inference
- `submission.py` - Official submission format

## Training

```bash
# Train for 100k timesteps (default)
python cage_trpo_solution/train.py --timesteps 100000

# Train for more timesteps
python cage_trpo_solution/train.py --timesteps 200000
```

## Evaluation

```bash
# Using official evaluation
python CybORG/Evaluation/evaluation.py --max-eps 10 cage_trpo_solution/ results_trpo
```

## Architecture

- **Algorithm**: TRPO (Trust Region Policy Optimization)
- **Network**: MLP with default SB3 architecture
- **Learning Rate**: 1e-3
- **Batch Size**: 2048 (larger than A2C for stability)
- **Target KL**: 0.01 (trust region constraint)
- **Gamma**: 0.99 (discount factor)

## TRPO vs PPO vs A2C

| Feature | TRPO | PPO | A2C |
|---------|------|-----|-----|
| **Updates** | Trust region | Clipped | Standard |
| **Stability** | High | High | Medium |
| **Speed** | Slow | Fast | Fast |
| **Theory** | Strong | Good | Basic |
| **Multi-agent** | Single policy | Multi-policy (RLlib) | Single policy |

## Expected Performance

TRPO should perform similarly to A2C but with:
- More stable training
- Less variance in results
- Slower training speed (larger batches)
