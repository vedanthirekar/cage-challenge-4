# CAGE Challenge 4 - DQN Solution

Multi-agent DQN solution using RLlib for CAGE Challenge 4.

## Files

- `train.py` - Training script
- `dqn_agent.py` - DQN agent class for inference
- `submission.py` - Official submission format
- `evaluate.py` - Evaluation script

## Training

```bash
# Train for 200 iterations (default 500-step episodes)
python cage_dqn_solution/train.py --iterations 200

# Resume training from checkpoint
python cage_dqn_solution/train.py --iterations 100
```

## Evaluation

```bash
# Using custom evaluate script
python cage_dqn_solution/evaluate.py --episodes 10

# Using official evaluation
python CybORG/Evaluation/evaluation.py --max-eps 10 cage_dqn_solution/ results_dqn
```

## Architecture

- **Algorithm**: DQN with Double DQN and Dueling networks
- **Policies**: 2 policies (small for agents 0-3, large for agent 4)
- **Network**: [256, 256, 128] fully connected layers
- **Exploration**: Epsilon-greedy (1.0 → 0.1 over 300k steps)

## Notes

- Episode length defaults to 500 to match evaluation
- Uses old RLlib API stack for stability
- Checkpoints saved to `trained_model/`
