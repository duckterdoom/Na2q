# NA²Q: Neural Attention Additive Q-Learning for Directional Sensor Networks

Implementation of **NA²Q** (Neural Attention Additive Model for Interpretable Multi-Agent Q-Learning) for Directional Sensor Networks (DSN) target tracking.

> **Paper**: [NA²Q (ICML 2023)](https://proceedings.mlr.press/v202/liu23be/liu23be.pdf) | **Code**: [github.com/zichuan-liu/NA2Q](https://github.com/zichuan-liu/NA2Q)

✅ **Verified**: Environment, architecture, and hyperparameters match paper specification.

## DSN Environment

**Scenario 1** (Small): 3×3 grid, 5 sensors, 6 targets  
**Scenario 2** (Large): 10×10 grid, 50 sensors, 60 targets

- **Sensing range**: 18m | **FoV angle**: 60° | **Actions**: TurnLeft/Stay/TurnRight (±5°)
- **Observation**: oᵢⱼ = (i, j, ρᵢⱼ, αᵢⱼ) | **Reward**: Coverage rate + bonuses

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Basic training
python main.py --mode train --scenario 1 --episodes 10000

# GPU training with parallel environments (recommended)
python main.py --mode train --scenario 1 --episodes 30000 --num-envs 8 --device cuda
```

**Recommended episodes**: Scenario 1: 10,000-30,000 | Scenario 2: 20,000-30,000

### Testing & Visualization

```bash
# Test trained model
python main.py --mode test --scenario 1 --model trainedModel/scenario1_best.pt --render

# Generate video
python main.py --mode video --scenario 1 --model trainedModel/scenario1_best.pt

# Generate charts
python main.py --mode visualize --scenario 1
```

## NA²Q Architecture

- **Agent Q-Network**: FC(obs→64) → GRU(64) → FC(64→actions)
- **Identity Semantics (VAE)**: 32-dim hidden, 16-dim latent, β=0.1
- **GAM Mixer**: Order-1 (individual) + Order-2 (pairwise) shape functions with ABS(weight) constraint
- **Value Decomposition**: `Q_tot = Σₖ (αₖ × fₖ(Q_inputs)) + bias(s)`

## Hyperparameters (from paper)

| Parameter | Value |
|-----------|-------|
| Learning rate | 0.0005 (RMSprop for Q, Adam for VAE) |
| Batch size | 32 |
| Discount γ | 0.99 |
| Epsilon | 1.0 → 0.05 (decays over 50,000 steps) |
| Target update | 200 steps (soft update) |
| VAE β | 0.1 |
| Replay buffer | 5,000 episodes |
| Gradient clip | 0.5 |

**Training config**: 100 steps/episode, saves every 1,000 episodes, LR decays 10% every 10,000 steps

## Features

- **GPU Support**: Auto-detects CUDA, falls back to CPU
- **Parallel Environments**: 5-10x faster with `--num-envs 8` (Scenario 1) or `--num-envs 4` (Scenario 2)
- **Long Training**: Supports 30,000+ episodes with periodic saves and memory cleanup
- **Stability**: Huber loss, Q-value clipping, soft target updates, VAE warmup

## Outputs

- **Charts**: `training_dashboard.png`, `coverage_ratio.png`, `training_losses.png`
- **Videos**: `scenario{1,2}_demo.gif` (15-second demo)
- **Models**: `best_model.pt`, `final_model.pt`, `checkpoint_*.pt`
- **Data**: `training_history.npz`, `metrics.json` (TensorBoard compatible)

## Citation

```bibtex
@inproceedings{liu2023na2q,
  title={NA2Q: Neural Attention Additive Model for Interpretable Multi-Agent Q-Learning},
  author={Liu, Zichuan and Zhu, Yuanyang and Chen, Chunlin},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```

## Verification

Run `python verify_implementation.py` to verify environment, architecture, and hyperparameters match the paper.

## References

- [NA²Q Paper (ICML 2023)](https://proceedings.mlr.press/v202/liu23be/liu23be.pdf)
- [NA²Q GitHub](https://github.com/zichuan-liu/NA2Q)
- [HiT-MAC: Multi-Agent Coordination for DSN (NeurIPS 2020)](https://github.com/XuJing1022/HiT-MAC)
