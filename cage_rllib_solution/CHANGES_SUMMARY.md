## Before vs After

| Parameter | Before | After | Why |
|-----------|--------|-------|-----|
| Network size | [128, 128] | [256, 256, 128] | More capacity for complex patterns |
| Learning rate | 3e-4 (fixed) | 5e-4 → 1e-4 → 5e-5 | Adaptive learning |
| Entropy coeff | 0.01 | 0.05 → 0.01 | More exploration early on |
| Gamma | 0.9 | 0.99 | Value long-term prevention |
| Train batch | 2000 | 4000 | More stable gradients |
| Mini batch | 64 | 128 | Better GPU usage |
| Env runners | 1 | 2 | 2x faster learning |
| Rollout length | 200 | 250 | Longer trajectories |
| Value network | Shared | Separate | Better value estimation |
| Grad clipping | None | 0.5 | Prevent instability |
| Checkpointing | End only | Best + periodic | Resume training |
