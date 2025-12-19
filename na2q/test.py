"""
NA²Q Evaluation Script - Test trained models.

Usage:
    python -m na2q.test --model Result/scenario1/best_model.pt --scenario 1
"""

import argparse
import numpy as np
import os

from environments.environment import make_env
from na2q.models import NA2QAgent
from na2q.utils import get_device


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Test NA²Q on DSN")
    parser.add_argument("--model", type=str, default="trainedModel/best_model.pt")
    parser.add_argument("--scenario", type=int, default=1, choices=[1, 2])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension (must match trained model)")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


# =============================================================================
# Test Function
# =============================================================================

def test(args):
    """Run evaluation on trained model."""
    device = get_device(args.device)
    args.device = device
    
    # Create environment
    env = make_env(
        scenario=args.scenario,
        max_steps=args.max_steps,
        render_mode="human" if args.render else None,
        seed=args.seed
    )
    
    # Print info
    print("=" * 60)
    print("Testing NA²Q on Directional Sensor Network")
    print("=" * 60)
    print(f"Scenario: {args.scenario}")
    print(f"  Sensors: {env.n_sensors}, Targets: {env.n_targets}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Create and load agent
    agent = NA2QAgent(
        n_agents=env.n_sensors,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        hidden_dim=args.hidden_dim,
        rnn_hidden_dim=args.hidden_dim,
        attention_hidden_dim=args.hidden_dim,
        device=device
    )
    
    if os.path.exists(args.model):
        agent.load(args.model)
        print(f"Loaded model from {args.model}")
    else:
        print(f"Warning: Model not found, using random policy")
    
    # Evaluate
    episode_rewards = []
    coverage_rates = []
    
    for ep in range(args.episodes):
        obs_list, info = env.reset()
        observations = np.stack(obs_list)
        agent.init_hidden(1)
        
        episode_reward = 0.0
        done, truncated = False, False
        prev_actions = np.zeros(env.n_sensors, dtype=np.int64)
        
        while not done and not truncated:
            avail_actions = np.stack(env.get_avail_actions())
            actions = agent.select_actions(observations, prev_actions, avail_actions, evaluate=True)
            
            next_obs_list, reward, done, truncated, info = env.step(actions.tolist())
            observations = np.stack(next_obs_list)
            episode_reward += reward
            prev_actions = actions
            
            if args.render:
                env.render()
        
        episode_rewards.append(episode_reward)
        coverage_rates.append(info.get("coverage_rate", 0))
        
        if args.verbose:
            print(f"Episode {ep+1:3d}: Reward={episode_reward:8.3f}, Coverage={info.get('coverage_rate', 0):6.1%}")
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Results over {args.episodes} episodes:")
    print("=" * 60)
    print(f"  Reward:   {np.mean(episode_rewards):8.3f} ± {np.std(episode_rewards):.3f}")
    print(f"  Coverage: {np.mean(coverage_rates)*100:8.2f}% ± {np.std(coverage_rates)*100:.2f}%")
    print("=" * 60)
    
    env.close()
    
    # Save results
    results = {
        "mean_reward": np.mean(episode_rewards),
        "mean_coverage": np.mean(coverage_rates),
        "episode_rewards": episode_rewards,
        "coverage_rates": coverage_rates
    }
    
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result_dir = os.path.join(script_dir, "Result", f"scenario{args.scenario}")
    os.makedirs(result_dir, exist_ok=True)
    
    history_path = os.path.join(result_dir, "test_history.npz")
    np.savez(history_path, episode_rewards=episode_rewards, coverage_rates=coverage_rates)
    print(f"Saved: {history_path}")
    
    # Generate chart
    try:
        from visualize import plot_test_results
        media_dir = os.path.join(result_dir, "media")
        os.makedirs(media_dir, exist_ok=True)
        chart_path = os.path.join(media_dir, "test_results.png")
        plot_test_results(results, chart_path, args.scenario)
    except Exception as e:
        print(f"Warning: Could not generate chart: {e}")
    
    return results


# =============================================================================
# Quick Test
# =============================================================================

def run_quick_test():
    """Quick test to verify environment and model work."""
    device = get_device()
    
    print("\n" + "=" * 60)
    print("Running Quick Test")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Test Scenario 1
    print("\n--- Scenario 1 ---")
    env1 = make_env(scenario=1)
    print(f"  Grid: {env1.grid_size}×{env1.grid_size}")
    print(f"  Sensors: {env1.n_sensors}, Targets: {env1.n_targets}")
    
    obs, _ = env1.reset()
    actions = [env1.action_space.sample() for _ in range(env1.n_sensors)]
    _, reward, _, _, info = env1.step(actions)
    print(f"  Step: reward={reward:.3f}, coverage={info.get('coverage_rate', 0):.1%}")
    env1.close()
    
    # Test Scenario 2
    print("\n--- Scenario 2 ---")
    env2 = make_env(scenario=2)
    print(f"  Grid: {env2.grid_size}×{env2.grid_size}")
    print(f"  Sensors: {env2.n_sensors}, Targets: {env2.n_targets}")
    
    obs, _ = env2.reset()
    actions = [env2.action_space.sample() for _ in range(env2.n_sensors)]
    _, reward, _, _, info = env2.step(actions)
    print(f"  Step: reward={reward:.3f}, coverage={info.get('coverage_rate', 0):.1%}")
    env2.close()
    
    # Test Model
    print("\n--- Model Test ---")
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
    print(f"  Actions: {actions}")
    env.close()
    
    print("\n" + "=" * 60)
    print("Quick Test PASSED ✓")
    print("=" * 60)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_test()
    else:
        args = parse_args()
        test(args)
