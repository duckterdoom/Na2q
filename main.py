"""
Main entry point for NA²Q on Directional Sensor Network.

Unified interface for:
- Training: python main.py --mode train --scenario 1
- Testing: python main.py --mode test --scenario 1
- Video generation: python main.py --mode video --scenario 1

Structure similar to HiT-MAC (https://github.com/XuJing1022/HiT-MAC)
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="NA²Q: Neural Attention Additive Model for Multi-Agent Q-Learning on DSN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Training Scenario 1 (small-scale, best results with GPU):
    python main.py --mode train --scenario 1 --episodes 10000 --num-envs 8
    
  Training Scenario 2 (large-scale, best results with GPU):
    python main.py --mode train --scenario 2 --episodes 20000 --num-envs 4
    
  Quick training (testing):
    python main.py --mode train --scenario 1 --episodes 2000
    
  Testing with trained model:
    python main.py --mode test --scenario 1 --model trainedModel/scenario1_best.pt
    
  Generate video:
    python main.py --mode video --scenario 1 --model trainedModel/scenario1_best.pt
    
  Quick test:
    python main.py --mode quick-test
    
  Parallel environments for GPU utilization:
    python main.py --mode train --scenario 1 --episodes 10000 --num-envs 8 --device cuda
    
  See TRAINING_RECOMMENDATIONS.md for detailed episode recommendations.
"""
    )
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "test", "video", "visualize", "quick-test"],
                        help="Operation mode")
    
    # Environment settings
    parser.add_argument("--scenario", type=int, default=1, choices=[1, 2],
                        help="Scenario (1: 3×3 grid/5 sensors, 2: 10×10 grid/50 sensors)")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Maximum steps per episode")
    
    # Training settings (from paper Table 3)
    parser.add_argument("--episodes", type=int, default=2000,
                        help="Number of training episodes (recommended: 10000 for Scenario 1, 20000 for Scenario 2)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate (0.0005 from paper)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=1.0,
                        help="Initial epsilon for exploration")
    parser.add_argument("--epsilon-end", type=float, default=0.05,
                        help="Final epsilon")
    parser.add_argument("--epsilon-decay", type=int, default=50000,
                        help="Epsilon decay steps")
    parser.add_argument("--target-update", type=int, default=200,
                        help="Target network update interval")
    parser.add_argument("--buffer-capacity", type=int, default=5000,
                        help="Replay buffer capacity (default: 5000, sufficient for 10k+ episodes)")
    
    # Evaluation settings
    parser.add_argument("--eval-interval", type=int, default=50,
                        help="Evaluation interval (episodes)")
    parser.add_argument("--save-interval", type=int, default=100,
                        help="Checkpoint save interval")
    parser.add_argument("--test-episodes", type=int, default=10,
                        help="Number of test episodes")
    
    # Model paths
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model for test/video modes")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Experiment name for logging")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory for results")
    
    # Video settings
    parser.add_argument("--video-duration", type=int, default=15,
                        help="Video duration in seconds")
    parser.add_argument("--video-fps", type=int, default=10,
                        help="Video frames per second")
    
    # Parallel environment settings
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Number of parallel environments (1=sequential, >1=parallel for GPU)")
    
    # Other settings
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cpu/cuda/auto). Default: auto-detect (CUDA if available)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--render", action="store_true",
                        help="Render environment during test")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    
    return parser.parse_args()


def run_train(args):
    """Run training mode."""
    from train import train
    
    exp_name = args.exp_name or f"scenario{args.scenario}"
    
    # Auto-select num_envs based on device and scenario
    num_envs = args.num_envs
    if num_envs == 1 and args.device == "cuda":
        # Suggest using parallel envs for GPU
        print("💡 Tip: Use --num-envs 4 or --num-envs 8 with CUDA for faster training")
    
    result = train(
        scenario=args.scenario,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        target_update_interval=args.target_update,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        log_dir=args.results_dir,
        exp_name=exp_name,
        device=args.device,
        seed=args.seed,
        num_envs=num_envs
    )
    
    print(f"\nTraining completed!")
    print(f"  Best model: {result['best_model_path']}")
    print(f"  Best eval reward: {result['best_eval_reward']:.3f}")
    
    # Copy best model to trainedModel/
    import shutil
    os.makedirs("trainedModel", exist_ok=True)
    best_model_dest = f"trainedModel/scenario{args.scenario}_best.pt"
    shutil.copy(result['best_model_path'], best_model_dest)
    print(f"  Copied to: {best_model_dest}")
    
    # Generate visualizations
    print("\nGenerating training visualizations...")
    from visualize import plot_training_results
    plot_training_results(result['exp_dir'])
    
    return result


def run_test(args):
    """Run test/evaluation mode."""
    from test import test
    
    class TestArgs:
        def __init__(self, args):
            self.model = args.model or f"trainedModel/scenario{args.scenario}_best.pt"
            self.scenario = args.scenario
            self.episodes = args.test_episodes
            self.max_steps = args.max_steps
            self.render = args.render
            self.device = args.device
            self.seed = args.seed
            self.verbose = args.verbose
    
    return test(TestArgs(args))


def run_video(args):
    """Generate video of trained agent."""
    from visualize import generate_video
    
    model_path = args.model or f"trainedModel/scenario{args.scenario}_best.pt"
    output_path = f"results/scenario{args.scenario}_demo.gif"
    
    generate_video(
        model_path=model_path,
        scenario=args.scenario,
        output_path=output_path,
        duration=args.video_duration,
        fps=args.video_fps,
        device=args.device,
        seed=args.seed
    )
    
    return output_path


def run_visualize(args):
    """Generate visualizations from training results."""
    from visualize import plot_training_results
    
    exp_dir = os.path.join(args.results_dir, args.exp_name or f"scenario{args.scenario}")
    
    if os.path.exists(exp_dir):
        plot_training_results(exp_dir)
    else:
        print(f"Error: Experiment directory not found: {exp_dir}")


def run_quick_test(args):
    """Run quick test to verify everything works."""
    from test import run_quick_test
    run_quick_test()


def main():
    from utils import get_device
    
    args = parse_args()
    
    # Auto-detect device if not specified
    device = get_device(args.device)
    
    print("=" * 60)
    print("NA²Q: Neural Attention Additive Q-Learning")
    print("Applied to Directional Sensor Network")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Scenario: {args.scenario}")
    print(f"Device: {device}")
    if device == "cuda":
        import torch
        print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
    print("=" * 60)
    
    # Update args.device with auto-detected device
    args.device = device
    
    if args.mode == "train":
        run_train(args)
    elif args.mode == "test":
        run_test(args)
    elif args.mode == "video":
        run_video(args)
    elif args.mode == "visualize":
        run_visualize(args)
    elif args.mode == "quick-test":
        run_quick_test(args)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()

