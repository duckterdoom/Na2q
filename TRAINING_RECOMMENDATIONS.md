# Training Episode Recommendations for NA²Q on DSN

## Quick Answer

**For Best Results:**
- **Scenario 1 (Small-scale)**: **5,000 - 10,000 episodes**
- **Scenario 2 (Large-scale)**: **10,000 - 20,000 episodes**

## Detailed Recommendations

### Scenario 1: Small-scale (3×3 grid, 5 sensors, 6 targets)

#### Minimum Training (Basic Results)
- **Episodes**: 2,000 - 3,000
- **Expected Coverage**: 70-85%
- **Training Time**: ~2-4 hours (CPU), ~30-60 min (CUDA)
- **Use Case**: Quick testing, proof of concept

#### Recommended Training (Good Results)
- **Episodes**: 5,000 - 8,000
- **Expected Coverage**: 85-95%
- **Training Time**: ~5-8 hours (CPU), ~1-2 hours (CUDA)
- **Use Case**: Standard research, good performance

#### Optimal Training (Best Results)
- **Episodes**: 10,000 - 15,000
- **Expected Coverage**: 95-100%
- **Training Time**: ~10-15 hours (CPU), ~2-3 hours (CUDA)
- **Use Case**: Publication-quality results, maximum performance

### Scenario 2: Large-scale (10×10 grid, 50 sensors, 60 targets)

#### Minimum Training (Basic Results)
- **Episodes**: 5,000 - 7,000
- **Expected Coverage**: 60-75%
- **Training Time**: ~8-12 hours (CPU), ~2-3 hours (CUDA)
- **Use Case**: Initial experiments

#### Recommended Training (Good Results)
- **Episodes**: 10,000 - 15,000
- **Expected Coverage**: 75-90%
- **Training Time**: ~15-25 hours (CPU), ~4-6 hours (CUDA)
- **Use Case**: Standard research, competitive performance

#### Optimal Training (Best Results)
- **Episodes**: 20,000 - 30,000
- **Expected Coverage**: 90-98%
- **Training Time**: ~30-50 hours (CPU), ~8-12 hours (CUDA)
- **Use Case**: Best possible results, complex coordination

## Why These Numbers?

### Epsilon Decay Consideration
- **Epsilon decay**: 50,000 training steps
- **Episode length**: 100 steps
- **Minimum episodes for full decay**: ~500 episodes (if 1 update/episode)
- **With multiple updates**: After 100 episodes, 2 updates/episode
- **Effective training steps**: ~500 + (episodes - 100) × 2 × 100
- **For 5,000 episodes**: ~1,080,000 training steps (well beyond epsilon decay)

### Learning Rate Scheduling
- **LR decay**: Every 5,000 training steps
- **Multiple LR decays**: Beneficial for long training
- **For 10,000 episodes**: ~200+ LR decay cycles (fine-tuning)

### Convergence Patterns

**Typical Training Progress:**

1. **Episodes 0-500**: Exploration phase
   - Coverage: 20-40%
   - High epsilon (1.0 → 0.8)
   - Learning basic coordination

2. **Episodes 500-2,000**: Early learning
   - Coverage: 40-70%
   - Epsilon: 0.8 → 0.3
   - Improving coordination

3. **Episodes 2,000-5,000**: Rapid improvement
   - Coverage: 70-90%
   - Epsilon: 0.3 → 0.1
   - Fine-tuning strategies

4. **Episodes 5,000-10,000**: Convergence
   - Coverage: 90-98%
   - Epsilon: 0.1 → 0.05
   - Optimal performance

5. **Episodes 10,000+**: Fine-tuning
   - Coverage: 98-100%
   - Epsilon: 0.05 (stable)
   - Maximum performance

## Monitoring Training Progress

### Key Metrics to Watch

1. **Coverage Rate**
   - Should steadily increase
   - Target: >90% for Scenario 1, >80% for Scenario 2
   - Monitor: `coverage_ratio.png`

2. **Episode Reward**
   - Should increase and stabilize
   - Target: >0.9 for Scenario 1, >0.8 for Scenario 2
   - Monitor: `training_dashboard.png`

3. **TD Loss**
   - Should decrease and stabilize
   - Target: <0.1 (stable)
   - Monitor: `training_losses.png`

4. **Epsilon**
   - Should decay from 1.0 to 0.05
   - Check: Should reach 0.05 by ~5,000 episodes

5. **Evaluation Reward**
   - Should increase and stabilize
   - Best indicator of actual performance

### When to Stop Training

**Stop if:**
- Coverage rate plateaus for 1,000+ episodes
- Evaluation reward doesn't improve for 500+ episodes
- TD loss is stable and low (<0.1)
- Epsilon has reached 0.05 and performance is good

**Continue if:**
- Coverage still improving (even slowly)
- Evaluation reward still increasing
- Haven't reached target coverage yet

## Recommended Training Commands

### Scenario 1 - Best Results (10,000 episodes)
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

### Scenario 2 - Best Results (20,000 episodes)
```bash
python main.py --mode train \
    --scenario 2 \
    --episodes 20000 \
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

## Time Estimates

### Scenario 1 (100 steps/episode)
- **CPU**: ~1-1.5 hours per 1,000 episodes
- **CUDA**: ~10-15 minutes per 1,000 episodes
- **10,000 episodes**: ~10-15 hours (CPU), ~2-3 hours (CUDA)

### Scenario 2 (100 steps/episode)
- **CPU**: ~1.5-2 hours per 1,000 episodes
- **CUDA**: ~15-20 minutes per 1,000 episodes
- **20,000 episodes**: ~30-40 hours (CPU), ~5-7 hours (CUDA)

## Tips for Best Results

1. **Use CUDA**: 5-10x faster training
2. **Monitor regularly**: Check evaluation metrics every 50 episodes
3. **Save checkpoints**: Best model is automatically saved
4. **Be patient**: MARL needs time to learn coordination
5. **Check convergence**: Use `visualize.py` to plot training curves
6. **Early stopping**: Stop if no improvement for 1,000+ episodes

## Summary Table

| Scenario | Minimum | Recommended | Optimal | Expected Coverage |
|----------|---------|-------------|---------|-------------------|
| **1 (Small)** | 2,000-3,000 | 5,000-8,000 | 10,000-15,000 | 95-100% |
| **2 (Large)** | 5,000-7,000 | 10,000-15,000 | 20,000-30,000 | 90-98% |

**For best results, use the Optimal training episodes!**


