# NA²Q for Directional Sensor Networks

## Quick Start

```bash
pip install -r requirements.txt

# Train (GPU settings applied automatically)
python -m na2q.main --mode train --scenario 1

# Test
python -m na2q.main --mode test --scenario 1

# Resume training
python -m na2q.main --mode train --scenario 1 --resume
```

## Scenarios

| Scenario | Grid | Sensors | Targets |
|----------|------|---------|---------|
| 1 | 3×3 | 5 | 6 |
| 2 | 10×10 | 50 | 60 |

## Outputs

- `Result/scenarioX/checkpoints/` — final models
- `Result/scenarioX/history/` — training logs
- `Result/scenarioX/media/` — charts and videos

## Config

See `train_config.py` for default training settings.
