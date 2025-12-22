# NA²Q for Directional Sensor Networks

## Quick Start

```bash
pip install -r requirements.txt

# Train Scenario 1
python -m na2q.main --mode train --scenario 1

# Train Scenario 2
python -m na2q.main --mode train --scenario 2

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

### Run Scenario 2 (Swarm)
To run the 50-sensor "Swarm" configuration:
```bash
python -m na2q.main --mode train --scenario 2
```

### Run Visualization
After training, generate GIF demos of the trained agents:

```bash
# Generate Scenario 1 GIF (5 sensors, 6 targets)
python -m na2q.main --mode video --scenario 1

# Generate Scenario 2 GIF (50 sensors, 60 targets)
python -m na2q.main --mode video --scenario 2
```

GIFs are saved to `Result/scenarioX/media/scenarioX_demo.gif` and show 1 test episode (100 steps) using the best trained model.

## Outputs

- `Result/scenarioX/checkpoints/` — final models
- `Result/scenarioX/history/` — training logs
- `Result/scenarioX/media/` — charts and videos

## Config

See `train_config.py` for default training settings.


