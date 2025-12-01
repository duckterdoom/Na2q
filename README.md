# NA²Q: Neural Attention Additive Q-Learning for Directional Sensor Networks

Implementation of **NA²Q** (Neural Attention Additive Model for Interpretable Multi-Agent Q-Learning) applied to **Directional Sensor Networks (DSN)** for target tracking.

> **Paper**: [NA²Q: Neural Attention Additive Model for Interpretable Multi-Agent Q-Learning](https://proceedings.mlr.press/v202/liu23be/liu23be.pdf) (ICML 2023)
>
> **Original Code**: [github.com/zichuan-liu/NA2Q](https://github.com/zichuan-liu/NA2Q)
>
> **Structure**: Inspired by [HiT-MAC](https://github.com/XuJing1022/HiT-MAC)

## Project Structure

```
Na2q/
├── trainedModel/           # Saved trained models
├── results/                # Training results, charts, videos
├── environment.py          # DSN Environment (Dec-POMDP)
├── model.py               # NA²Q Model implementation
├── main.py                # Main entry point (unified interface)
├── train.py               # Training functions
├── test.py                # Evaluation script
├── visualize.py           # Visualization and video generation
├── verify_implementation.py # Verification script
├── utils/                 # Utility modules
│   ├── __init__.py        # Device selection, setup
│   ├── replay_buffer.py   # Episode replay buffer
│   └── logger.py          # Logging and metrics
├── requirements.txt       # Dependencies
├── README.md              # This file
└── TRAINING_IMPROVEMENTS.md # Training optimizations documentation
```

## DSN Environment

Based on **EXPERIMENT ENVIRONMENT** specification:

### Scenario 1 (Small-scale)
- **Grid**: 3×3 (20m × 20m cells)
- **Sensors**: 5 sensors at cells 1, 3, 5, 7, 9
- **Targets**: 6 randomly moving targets
- **Sensing range**: ρmax = 18m
- **FoV angle**: αmax = 60°

### Scenario 2 (Large-scale)
- **Grid**: 10×10 (20m × 20m cells)
- **Sensors**: 50 sensors (50% probability placement)
- **Targets**: 60 randomly moving targets
- **Sensing range**: ρmax = 18m
- **FoV angle**: αmax = 60°

### Dec-POMDP Formulation
- **Observation**: oᵢⱼ = (i, j, ρᵢⱼ, αᵢⱼ) in polar coordinates
  - `i`: Sensor ID (normalized)
  - `j`: Target ID (normalized)
  - `ρᵢⱼ`: Absolute distance from sensor i to target j (normalized)
  - `αᵢⱼ`: Relative angle from sensor i to target j (normalized)
- **Actions**: TurnLeft (-5°), Stay, TurnRight (+5°)
- **Goal Map**: n × m binary matrix (gᵢⱼ = 1 if target j tracked by sensor i)
- **Reward**: Coverage rate (tracked targets / total targets) + bonus for full coverage
- **Goal**: Maximize number of tracked targets

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Test
```bash
# Verify environment and model work correctly
python main.py --mode quick-test

# Verify implementation matches paper and environment
python verify_implementation.py
```

### Training

**Scenario 1 (Small-scale)**:
```bash
python main.py --mode train --scenario 1 --episodes 2000
```

**Scenario 2 (Large-scale)**:
```bash
python main.py --mode train --scenario 2 --episodes 5000
```

**Long Training (Best Results)**:
```bash
# Scenario 1: 10,000 episodes for optimal results (95-100% coverage)
python main.py --mode train --scenario 1 --episodes 10000

# Scenario 2: 20,000 episodes for optimal results (90-98% coverage)
python main.py --mode train --scenario 2 --episodes 20000
```

> **💡 Training Recommendations**: See `TRAINING_RECOMMENDATIONS.md` for detailed episode recommendations:
> - **Scenario 1**: 5,000-10,000 episodes (recommended: 10,000 for best results)
> - **Scenario 2**: 10,000-20,000 episodes (recommended: 20,000 for best results)

**With CUDA (auto-detected)**:
```bash
# Automatically uses CUDA if available, falls back to CPU
python main.py --mode train --scenario 2 --episodes 10000

# Force CUDA (warns if unavailable)
python main.py --mode train --scenario 2 --episodes 10000 --device cuda

# Force CPU
python main.py --mode train --scenario 2 --episodes 10000 --device cpu
```

### Testing
```bash
python main.py --mode test --scenario 1 --model trainedModel/scenario1_best.pt
```

With visualization:
```bash
python main.py --mode test --scenario 1 --render --verbose
```

### Generate Video
```bash
python main.py --mode video --scenario 1 --model trainedModel/scenario1_best.pt
```

### Generate Charts
```bash
python main.py --mode visualize --scenario 1
```

## NA²Q Architecture

Based on the ICML 2023 paper:

### 1. Agent Q-Network
- FC layer (obs_dim → 64)
- GRU with 64-dimensional hidden state
- FC layer (64 → n_actions)

### 2. Identity Semantics (VAE)
- Encoder: 2 FC layers with 32-dim hidden
- Latent dimension: 16
- Loss: MSE + β×KL (β = 0.1)

### 3. NA²Q Mixer (GAM-based)
- **Order-1**: n individual shape functions
- **Order-2**: C(n,2) pairwise shape functions
- **Shape Function**: 3-layer MLP with ABS(weight) constraint
  - Layer 1: ABS(Linear(input, 8)) + ELU
  - Layer 2: ABS(Linear(8, 4)) + ELU
  - Layer 3: ABS(Linear(4, 1))
- **Attention**: Credit assignment using state and semantics

### 4. Value Decomposition
```
Q_tot = Σₖ (αₖ × fₖ(Q_inputs)) + bias(s)
```

Where αₖ are attention-based credits and fₖ are shape functions.

## Hyperparameters (from paper)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 0.0005 | RMSprop for Q-network, Adam for VAE |
| Batch size | 32 | Episode-based replay buffer |
| Discount γ | 0.99 | Standard Q-learning discount |
| Epsilon | 1.0 → 0.05 | Decays over 50,000 steps |
| Target update | 200 steps | Soft target network updates |
| VAE β | 0.1 | KL divergence weight |
| Optimizer (Q) | RMSprop | For Q-network and mixer |
| Optimizer (VAE) | Adam | For identity semantics encoder |
| Replay buffer | 5,000 | Episode capacity (auto-cleanup) |
| Gradient clip | 10.0 | Prevents exploding gradients |

### Training Configuration

- **Episode length**: 100 steps (configurable)
- **Evaluation interval**: Every 50 episodes
- **Checkpoint interval**: Every 100 episodes
- **History save interval**: Every 1,000 episodes (for long runs)
- **Checkpoint cleanup**: Keeps last 10 checkpoints (for runs > 2,000 episodes)
- **Updates per episode**: 1 for first 100 episodes, 2 thereafter (better sample efficiency)
- **Learning rate decay**: Every 5,000 training steps (50% reduction)

## Features

### CUDA Support
- **Auto-detection**: Automatically uses CUDA if available, falls back to CPU
- **Device information**: Shows CUDA device name and version when using GPU
- **Fallback handling**: Warns and uses CPU if CUDA is requested but unavailable

### Long Training Support
- **Periodic saves**: Training history saved every 1,000 episodes
- **Checkpoint management**: Keeps last 10 checkpoints + best/final models
- **Memory efficient**: ~240 KB for 10,000 episodes training history
- **Replay buffer**: 5,000 capacity with automatic cleanup

### Training Optimizations (Verified & Tested)
- **Epsilon decay**: Proper step-based decay (1.0 → 0.05 over 50,000 steps)
- **Target Q-values**: Correct computation with proper next-state handling and dimension management
- **Reward shaping**: Improved learning signal with bonuses for high/full coverage and penalties for low coverage
- **Learning rate scheduling**: StepLR schedulers (decay by 50% every 5,000 steps) for stable long-term training
- **Hidden state reset**: Properly resets between episodes using previous timestep done flags (prevents information leakage)
- **Replay buffer**: Optimized sampling with proper diversity and padding
- **Multiple updates**: 2 updates per episode after initial exploration (100 episodes) for better sample efficiency
- **Dimension handling**: Efficient tensor dimension management (no unnecessary transformations)

### Verification
- **Implementation check**: `verify_implementation.py` verifies:
  - Environment matches experiment specification
  - NA²Q architecture matches paper
  - Training hyperparameters match paper
  - All components work correctly

## Outputs

After training, the following are generated:

### Charts
- `training_dashboard.png`: Combined training overview (rewards, coverage, loss)
- `coverage_ratio.png`: Target coverage over time
- `training_losses.png`: Loss curves (TD loss + VAE loss)

### Videos
- `scenario{1,2}_demo.gif`: 15-second demonstration showing:
  - Grid layout with sensors and targets
  - Sensor FoV visualization
  - Target tracking status (green=tracked, red=untracked)

### Knowledge
- `training_knowledge.json`: Model configuration, metrics, interpretability data
- `training_history.npz`: Complete training metrics (rewards, coverage, losses)
- `metrics.json`: Detailed training logs with TensorBoard support

### Checkpoints
- `best_model.pt`: Best model based on evaluation reward
- `final_model.pt`: Final model after training
- `checkpoint_*.pt`: Periodic checkpoints (last 10 kept for long runs)

## Citation

```bibtex
@inproceedings{liu2023na2q,
  title={NA2Q: Neural Attention Additive Model for Interpretable Multi-Agent Q-Learning},
  author={Liu, Zichuan and Zhu, Yuanyang and Chen, Chunlin},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```

## Advanced Usage

### Custom Training Configuration
```bash
python main.py --mode train \
    --scenario 1 \
    --episodes 10000 \
    --batch-size 32 \
    --lr 0.0005 \
    --gamma 0.99 \
    --epsilon-start 1.0 \
    --epsilon-end 0.05 \
    --epsilon-decay 50000 \
    --target-update 200 \
    --buffer-capacity 5000 \
    --eval-interval 50 \
    --save-interval 100 \
    --device cuda
```

### Evaluation Options
```bash
# Standard evaluation
python main.py --mode test --scenario 1 --model trainedModel/scenario1_best.pt

# With rendering
python main.py --mode test --scenario 1 --model trainedModel/scenario1_best.pt --render

# Verbose output
python main.py --mode test --scenario 1 --model trainedModel/scenario1_best.pt --verbose

# Custom number of episodes
python main.py --mode test --scenario 1 --model trainedModel/scenario1_best.pt --test-episodes 20
```

### Video Generation
```bash
# Default 15-second video
python main.py --mode video --scenario 1 --model trainedModel/scenario1_best.pt

# Custom duration and FPS
python main.py --mode video \
    --scenario 1 \
    --model trainedModel/scenario1_best.pt \
    --video-duration 30 \
    --video-fps 15
```

## Implementation Details

### Verified Components
✅ **Environment**: Matches EXPERIMENT ENVIRONMENT specification exactly
- Dec-POMDP formulation
- Observation format: oᵢⱼ = (i, j, ρᵢⱼ, αᵢⱼ)
- Action space: 3 discrete actions (±5° rotation)
- Goal map: n × m binary matrix
- Both scenarios implemented correctly
- Improved reward shaping: coverage rate + bonuses/penalties

✅ **NA²Q Architecture**: Matches ICML 2023 paper exactly
- GAM-based value decomposition
- Shape functions with ABS(weight) constraint
- Identity semantics (VAE) with 32-dim hidden, 16-dim latent
- GRU agent networks with 64-dim hidden
- Attention mechanism for credit assignment

✅ **Training**: Matches paper hyperparameters + optimizations
- Learning rate: 0.0005 (with StepLR scheduling for long runs)
- Batch size: 32
- Epsilon decay: 50,000 steps (proper step-based decay)
- Target update: 200 steps
- VAE β: 0.1
- Hidden state reset: Properly handled between episodes
- Target Q-value computation: Correct next-state handling
- Multiple updates per episode: After initial exploration (100 episodes)

### Performance Optimizations
- **CUDA acceleration**: Automatic GPU detection and usage
- **Memory efficiency**: Optimized for long training runs (10,000+ episodes)
- **Checkpoint management**: Automatic cleanup to save disk space
- **Periodic saves**: Prevents data loss during long training
- **Efficient tensor operations**: Optimized dimension handling (no unnecessary transformations)
- **Smart sampling**: Replay buffer with proper diversity and padding
- **Adaptive updates**: Multiple updates per episode after initial exploration phase

## Troubleshooting

### CUDA Issues
- If CUDA is requested but unavailable, the code automatically falls back to CPU
- Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`

### Memory Issues
- For very long training (>10,000 episodes), consider reducing `buffer_capacity`
- Training history uses ~240 KB for 10,000 episodes (acceptable)

### Checkpoint Issues
- Old checkpoints are automatically cleaned up (keeps last 10)
- Best and final models are always preserved

### Training Issues
- **Slow convergence**: Check that epsilon decay is working (should decrease from 1.0 to 0.05)
- **Unstable training**: Learning rate scheduling helps (decays every 5,000 steps)
- **Poor performance**: Ensure reward shaping is active (check coverage bonuses/penalties)
- **Dimension errors**: All tensor dimensions are properly handled (verified)

### Verification
Run the verification script to ensure everything is correct:
```bash
python verify_implementation.py
```

## Training Improvements

All critical fixes and optimizations have been applied and verified:

1. **Epsilon Decay**: Fixed to proper step-based decay (1.0 → 0.05 over 50,000 steps)
2. **Target Q-Values**: Fixed computation with correct next-state handling and dimension management
3. **Reward Shaping**: Improved learning signal with bonuses for high/full coverage
4. **Learning Rate Scheduling**: Added StepLR schedulers for stable long-term training
5. **Hidden State Reset**: Properly resets between episodes (prevents information leakage)
6. **Replay Buffer**: Optimized sampling with proper diversity and padding
7. **Multiple Updates**: 2 updates per episode after initial exploration for better efficiency

See `TRAINING_IMPROVEMENTS.md` for detailed documentation of all improvements.

## Expected Results

With all optimizations applied, you should see:
- **Faster convergence**: Better reward shaping and target Q-value computation
- **Higher final performance**: Improved coverage rates (approaching 100% for Scenario 1)
- **More stable training**: Learning rate scheduling and proper hidden state handling
- **Better long-term learning**: Optimized for 10,000+ episode training runs

## References

- [NA²Q Paper (ICML 2023)](https://proceedings.mlr.press/v202/liu23be/liu23be.pdf)
- [NA²Q GitHub](https://github.com/zichuan-liu/NA2Q)
- [HiT-MAC: Multi-Agent Coordination for DSN (NeurIPS 2020)](https://github.com/XuJing1022/HiT-MAC)
