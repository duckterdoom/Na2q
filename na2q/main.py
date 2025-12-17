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
  Training Scenario 1 (GPU settings applied automatically):
    python main.py --mode train --scenario 1
    
  Training Scenario 2:
    python main.py --mode train --scenario 2
     
  Testing with trained model:
    python main.py --mode test --scenario 1 --model trainedModel/scenario1_best.pt
    
  Generate video:
    python main.py --mode video --scenario 1 --model trainedModel/scenario1_best.pt
    
  Quick test:
    python main.py --mode quick-test
    
  Override settings (e.g., fewer episodes):
    python main.py --mode train --scenario 1 --episodes 5000
    
  See train_config.py for default GPU training settings.
"""
    )
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "test", "video", "visualize", "quick-test"],
                        help="Operation mode")
    
    # Environment settings
    parser.add_argument("--scenario", type=int, default=1, choices=[1, 2, 99],
                        help="Scenario (1: 3×3 grid/5 sensors, 2: 10×10 grid/50 sensors, 99: Sanity Check)")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Maximum steps per episode")
    
    # Training settings (from paper Table 3)
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of training episodes (default: from train_config.py - 10000 for Scenario 1, 30000 for Scenario 2)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (default: from train_config.py)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: from train_config.py)")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Discount factor (default: from train_config.py)")
    parser.add_argument("--epsilon-start", type=float, default=None,
                        help="Initial epsilon (default: from train_config.py)")
    parser.add_argument("--epsilon-end", type=float, default=None,
                        help="Final epsilon (default: from train_config.py)")
    parser.add_argument("--epsilon-decay", type=int, default=None,
                        help="Epsilon decay steps (default: from train_config.py)")
    parser.add_argument("--target-update", type=int, default=None,
                        help="Target network update interval (default: from train_config.py)")
    parser.add_argument("--buffer-capacity", type=int, default=None,
                        help="Replay buffer capacity (default: from train_config.py)")
    parser.add_argument("--learning-starts", type=int, default=None,
                        help="Steps before starting learning (default: 5000)")
    
    # Evaluation settings
    parser.add_argument("--eval-interval", type=int, default=None,
                        help="Evaluation interval (default: from train_config.py)")
    parser.add_argument("--save-interval", type=int, default=None,
                        help="Checkpoint save interval (default: from train_config.py)")
    parser.add_argument("--test-episodes", type=int, default=10,
                        help="Number of test episodes")
    
    # Model paths
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model for test/video modes")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Experiment name for logging")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from existing checkpoint")
    parser.add_argument("--results-dir", type=str, default="Result",
                        help="Directory for training results")
    
    # Video settings
    parser.add_argument("--video-duration", type=int, default=15,
                        help="Video duration in seconds")
    parser.add_argument("--video-fps", type=int, default=10,
                        help="Video frames per second")
    
    # Parallel environment settings
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Number of parallel environments (default: from train_config.py)")
    
    # Other settings
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cpu/cuda/auto). Default: auto-detect (CUDA if available)")
    parser.add_argument("--gpu-id", type=int, default=None,
                        help="CUDA device index to use (default: 0). Ignored on CPU.")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision on CUDA (enabled by default for faster GPU training)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--render", action="store_true",
                        help="Render environment during test")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    
    return parser.parse_args()


def run_train(args):
    """Run training mode."""
    from na2q.engine.trainer import Trainer
    from environments.environment import DSNEnv
    from train_config import apply_strong_gpu_defaults
    
    # Always apply strong GPU defaults for training (can be overridden by CLI args)
    args = apply_strong_gpu_defaults(args, override_existing=False)
    
    # Print actual config being used (after applying defaults and CLI overrides)
    print(f"Training config (Scenario {args.scenario}) - from train_config.py:")
    print(f"  episodes        : {args.episodes}")
    print(f"  batch_size      : {args.batch_size}")
    print(f"  lr              : {args.lr}")
    print(f"  gamma           : {args.gamma}")
    print(f"  epsilon_start   : {args.epsilon_start}")
    print(f"  epsilon_end     : {args.epsilon_end}")
    print(f"  epsilon_decay   : {args.epsilon_decay}")
    print(f"  target_update   : {args.target_update}")
    print(f"  buffer_capacity : {args.buffer_capacity}")
    print(f"  num_envs        : {args.num_envs}")
    print(f"  eval_interval   : {args.eval_interval}")
    print(f"  save_interval   : {args.save_interval}")
    
    exp_name = args.exp_name or f"scenario{args.scenario}"
    
    num_envs = args.num_envs
    
    # Create config dict from args
    config = vars(args)
    config.update({
        "n_episodes": args.episodes,
        "log_dir": args.results_dir,
        "exp_name": exp_name,
        "use_amp": not args.no_amp,
        "learning_starts": getattr(args, 'learning_starts', 5000),
        "target_update_interval": args.target_update
    })
    
    # Initialize and run Trainer
    trainer = Trainer(config)
    result = trainer.train()
    
    print(f"\nTraining completed!")
    print(f"  Best model: {result['best_model_path']}")
    if result['best_eval_reward'] != -float('inf'):
        print(f"  Best eval reward: {result['best_eval_reward']:.3f}")
    else:
        print(f"  Best eval reward: (using final model)")
    
    # Copy best model to Result/scenarioX/
    import shutil
    Result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                       "Result", f"scenario{args.scenario}")
    os.makedirs(Result_dir, exist_ok=True)
    best_model_dest = os.path.join(Result_dir, "best_model.pt")
    
    if os.path.exists(result['best_model_path']):
        shutil.copy(result['best_model_path'], best_model_dest)
        print(f"  Saved to: {best_model_dest}")
    
    # Generate visualizations
    print("\nGenerating training visualizations...")
    from visualize import plot_training_results
    plot_training_results(result['exp_dir'])
    
    return result


def run_test(args):
    """Run test/evaluation mode."""
    from na2q.test import test
    
    class TestArgs:
        def __init__(self, args):
            # Look in Result first, fall back to results
            Result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                              "Result", f"scenario{args.scenario}")
            default_model = os.path.join(Result_dir, "best_model.pt")
            if not os.path.exists(default_model):
                exp_dir = os.path.join(args.results_dir, args.exp_name or f"scenario{args.scenario}")
                default_model = os.path.join(exp_dir, "checkpoints", "best_model.pt")
            self.model = args.model or default_model
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
    
    # Model from Result, output video to Result/media
    Result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              "Result", f"scenario{args.scenario}")
    default_model = os.path.join(Result_dir, "best_model.pt")
    model_path = args.model or default_model
    
    media_dir = os.path.join(Result_dir, "media")
    os.makedirs(media_dir, exist_ok=True)
    output_path = os.path.join(media_dir, f"scenario{args.scenario}_demo.gif")
    
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
    
    # Read history from training_result, output training charts there
    training_result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                       "training_result", f"scenario{args.scenario}")
    history_dir = os.path.join(training_result_dir, "checkpoints")  # History in checkpoints
    media_dir = os.path.join(training_result_dir, "media")
    os.makedirs(media_dir, exist_ok=True)
    
    if os.path.exists(os.path.join(history_dir, "training_history.npz")):
        plot_training_results(
            exp_dir=training_result_dir,
            history_dir=history_dir,
            media_dir=media_dir
        )
        print(f"Training charts saved to: {media_dir}")
    else:
        print(f"Error: No training history found at {history_dir}")


def run_quick_test(args):
    """Run quick test to verify everything works."""
    from na2q.test import run_quick_test
    run_quick_test()


def main():
    from na2q.utils import get_device
    
    args = parse_args()
    
    # Note: Strong GPU defaults are now applied automatically in run_train()
    
    # Auto-detect device if not specified
    device = get_device(args.device)
    args.device = device
    
    print("=" * 60)
    print("NA²Q: Neural Attention Additive Q-Learning")
    print("Applied to Directional Sensor Network")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Scenario: {args.scenario}")
    print(f"Device: {device}")
    if device == "cuda":
        import torch
        if args.gpu_id is not None:
            torch.cuda.set_device(args.gpu_id)
        print(f"  CUDA Device [{torch.cuda.current_device()}]: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"  CUDA Version: {torch.version.cuda}")
    print("=" * 60)
    
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
