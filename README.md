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

### Quick Verification (Sanity Check)
To test if the fix for **Scenario 1** works without waiting hours, run the "Turbo Test" config (Scenario 99):
```bash
python3 na2q/main.py --mode train --scenario 99
```
*Runs for 15 minutes (2k episodes) to verify the "Infinite Range Reward" fix.*

### Run Scenario 2 (Swarm)
To run the 50-sensor "Swarm" configuration:
```bash
python3 na2q/main.py --mode train --scenario 2
```

### Run Visualization
After training, visualized the results:

## Outputs

- `Result/scenarioX/checkpoints/` — final models
- `Result/scenarioX/history/` — training logs
- `Result/scenarioX/media/` — charts and videos

## Config

See `train_config.py` for default training settings.

## Documentation

- [Coverage Analysis](docs/coverage_analysis.md) — Mathematical proof of why 80% coverage is impossible.
