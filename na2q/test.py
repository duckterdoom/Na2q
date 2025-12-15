"""
Evaluation and testing script for NA²Q.

Similar to HiT-MAC test.py - loads trained model and evaluates performance.
Usage:
    python test.py --model trainedModel/best_model.pt --scenario 1 --episodes 10
"""

import argparse
import numpy as np
import os

from environments.environment import make_env
from na2q.models import NA2QAgent
from na2q.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Test NA²Q on DSN")
    parser.add_argument("--model", type=str, default="trainedModel/best_model.pt", 
                        help="Path to trained model")
    parser.add_argument("--scenario", type=int, default=1, choices=[1, 2],
                        help="Scenario (1: small-scale, 2: large-scale)")
    parser.add_argument("--episodes", type=int, default=10, 
                        help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Maximum steps per episode")
    parser.add_argument("--render", action="store_true",
                        help="Render environment during evaluation")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cpu/cuda/auto). Default: auto-detect (CUDA if available)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed results")
    return parser.parse_args()


def test(args):
    """Run evaluation on trained model."""
    
    # Auto-detect device if not specified
    device = get_device(args.device)
    args.device = device
    
    # Create environment
    env = make_env(
        scenario=args.scenario,
        max_steps=args.max_steps,
        render_mode="human" if args.render else None,
        seed=args.seed
    )
    
    print(f"=" * 60)
    print(f"Testing NA²Q on Directional Sensor Network")
    print(f"=" * 60)
    print(f"Scenario: {args.scenario}")
    print(f"  Grid: {env.grid_size}×{env.grid_size} ({env.cell_size}m cells)")
    print(f"  Sensors: {env.n_sensors}")
    print(f"  Targets: {env.n_targets}")
    print(f"  Sensing range: {env.sensing_range}m")
    print(f"  FoV angle: {np.degrees(env.fov_angle):.0f}°")
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Device: {device}")
    if device == "cuda":
        import torch
        print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"=" * 60)
    
    # Create and load agent
    agent = NA2QAgent(
        n_agents=env.n_sensors,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        device=device
    )
    
    if os.path.exists(args.model):
        agent.load(args.model)
        print(f"Loaded model from {args.model}")
    else:
        print(f"Warning: Model not found at {args.model}, using random policy")
    
    # Evaluate
    episode_rewards = []
    coverage_rates = []
    tracked_counts = []
    
    for ep in range(args.episodes):
        obs_list, info = env.reset()
        observations = np.stack(obs_list)
        agent.init_hidden(1)
        
        episode_reward = 0.0
        done = False
        truncated = False
        step = 0
        
        prev_actions = np.zeros(env.n_sensors, dtype=np.int64)
        
        while not done and not truncated:
            avail_actions = np.stack(env.get_avail_actions())
            actions = agent.select_actions(observations, prev_actions, avail_actions, evaluate=True)
            
            next_obs_list, reward, done, truncated, info = env.step(actions.tolist())
            observations = np.stack(next_obs_list)
            episode_reward += reward
            step += 1
            prev_actions = actions
            
            if args.render:
                env.render()
        
        episode_rewards.append(episode_reward)
        coverage_rates.append(info.get("coverage_rate", 0))
        tracked_counts.append(info.get("n_tracked", 0))
        
        if args.verbose:
            print(f"Episode {ep+1:3d}: Reward={episode_reward:8.3f}, "
                  f"Coverage={info.get('coverage_rate', 0):6.1%}, "
                  f"Tracked={info.get('n_tracked', 0):2d}/{env.n_targets}")
    
    # Print summary
    print(f"\n" + "=" * 60)
    print(f"Results over {args.episodes} episodes:")
    print(f"=" * 60)
    print(f"  Total Reward:  {np.mean(episode_rewards):8.3f} ± {np.std(episode_rewards):.3f}")
    print(f"  Coverage Rate: {np.mean(coverage_rates)*100:8.2f}% ± {np.std(coverage_rates)*100:.2f}%")
    print(f"  Targets Tracked: {np.mean(tracked_counts):6.2f} ± {np.std(tracked_counts):.2f}")
    print(f"  Best Episode:  {np.max(episode_rewards):.3f} reward, {np.max(coverage_rates)*100:.1f}% coverage")
    print(f"  Worst Episode: {np.min(episode_rewards):.3f} reward, {np.min(coverage_rates)*100:.1f}% coverage")
    print(f"=" * 60)
    
    env.close()
    
    # Generate test results chart
    results = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_coverage": np.mean(coverage_rates),
        "std_coverage": np.std(coverage_rates),
        "episode_rewards": episode_rewards,
        "coverage_rates": coverage_rates
    }
    
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result_dir = os.path.join(script_dir, "Result", f"scenario{args.scenario}")
    os.makedirs(result_dir, exist_ok=True)
    
    # Save test history
    history_path = os.path.join(result_dir, "test_history.npz")
    np.savez(history_path,
             episode_rewards=episode_rewards,
             coverage_rates=coverage_rates)
    print(f"Saved: {history_path}")
    
    # Save chart to Result/media folder
    try:
        from visualize import plot_test_results
        media_dir = os.path.join(result_dir, "media")
        os.makedirs(media_dir, exist_ok=True)
        chart_path = os.path.join(media_dir, "test_results.png")
        plot_test_results(results, chart_path, args.scenario)
    except Exception as e:
        print(f"Warning: Could not generate test chart: {e}")
    
    return results


def run_quick_test():
    """Quick test to verify environment and model work correctly."""
    from na2q.utils import get_device
    
    device = get_device()
    
    print("\n" + "=" * 60)
    print("Running Quick Test")
    print(f"Device: {device}")
    if device == "cuda":
        import torch
        print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    
    # Test Scenario 1
    print("\n--- Testing Scenario 1 (Small-scale) ---")
    env1 = make_env(scenario=1)
    print(f"  Grid: {env1.grid_size}×{env1.grid_size}")
    print(f"  Sensors: {env1.n_sensors}")
    print(f"  Targets: {env1.n_targets}")
    print(f"  Obs dim: {env1.obs_dim}")
    print(f"  State dim: {env1.state_dim}")
    
    obs, info = env1.reset()
    print(f"  Reset: {len(obs)} observations, shape {obs[0].shape}")
    
    actions = [env1.action_space.sample() for _ in range(env1.n_sensors)]
    next_obs, reward, done, truncated, info = env1.step(actions)
    print(f"  Step: reward={reward:.3f}, coverage={info.get('coverage_rate', 0):.1%}")
    env1.close()
    
    # Test Scenario 2
    print("\n--- Testing Scenario 2 (Large-scale) ---")
    env2 = make_env(scenario=2)
    print(f"  Grid: {env2.grid_size}×{env2.grid_size}")
    print(f"  Sensors: {env2.n_sensors}")
    print(f"  Targets: {env2.n_targets}")
    print(f"  Obs dim: {env2.obs_dim}")
    print(f"  State dim: {env2.state_dim}")
    
    obs, info = env2.reset()
    print(f"  Reset: {len(obs)} observations, shape {obs[0].shape}")
    
    actions = [env2.action_space.sample() for _ in range(env2.n_sensors)]
    next_obs, reward, done, truncated, info = env2.step(actions)
    print(f"  Step: reward={reward:.3f}, coverage={info.get('coverage_rate', 0):.1%}")
    env2.close()
    
    # Test NA2Q Model
    print("\n--- Testing NA²Q Model ---")
    env = make_env(scenario=1)
    agent = NA2QAgent(
        n_agents=env.n_sensors,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        device=device
    )
    
    obs_list, _ = env.reset()
    observations = np.stack(obs_list)
    agent.init_hidden(1)
    avail_actions = np.stack(env.get_avail_actions())
    prev_actions = np.zeros(env.n_sensors, dtype=np.int64)
    actions = agent.select_actions(observations, prev_actions, avail_actions)
    print(f"  Actions selected: {actions}")
    
    # Test interpretable contributions
    state = env.get_state()
    contribs = agent.get_interpretable_contributions(observations, state, actions)
    print(f"  Individual contributions shape: {contribs['individual_contribs'].shape}")
    print(f"  Pairwise contributions shape: {contribs['pairwise_contribs'].shape}")
    print(f"  Attention weights shape: {contribs['attention_weights'].shape}")
    
    env.close()
    
    print("\n" + "=" * 60)
    print("Quick Test PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_test()
    else:
        args = parse_args()
        test(args)
