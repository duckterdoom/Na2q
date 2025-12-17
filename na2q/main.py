"""
NA²Q - Neural Attention Additive Q-Learning for Directional Sensor Networks.

Usage:
    python -m na2q.main --mode train --scenario 1
    python -m na2q.main --mode test --scenario 1
    python -m na2q.main --mode video --scenario 1
"""

import argparse
import os
import sys


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="NA²Q: Multi-Agent Q-Learning on DSN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m na2q.main --mode train --scenario 1
  python -m na2q.main --mode test --scenario 1
  python -m na2q.main --mode video --scenario 1
"""
    )
    
    # Mode
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "test", "video", "visualize", "quick-test"])
    
    # Environment
    parser.add_argument("--scenario", type=int, default=1, choices=[1, 2])
    parser.add_argument("--max-steps", type=int, default=100)
    
    # Training (defaults from train_config.py)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--epsilon-start", type=float, default=None)
    parser.add_argument("--epsilon-end", type=float, default=None)
    parser.add_argument("--epsilon-decay", type=int, default=None)
    parser.add_argument("--target-update", type=int, default=None)
    parser.add_argument("--buffer-capacity", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    
    # Evaluation
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--test-episodes", type=int, default=10)
    
    # Paths
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--results-dir", type=str, default="Result")
    
    # Video
    parser.add_argument("--video-duration", type=int, default=15)
    parser.add_argument("--video-fps", type=int, default=10)
    
    # Hardware
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    # Misc
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    return parser.parse_args()


# =============================================================================
# Train Mode
# =============================================================================

def run_train(args):
    """Run training mode."""
    from na2q.engine.trainer import Trainer
    from train_config import apply_strong_gpu_defaults
    import shutil
    
    # Apply config defaults
    args = apply_strong_gpu_defaults(args, override_existing=False)
    
    # Print config
    print(f"Training config (Scenario {args.scenario}) - from train_config.py:")
    print(f"  episodes        : {args.episodes}")
    print(f"  batch_size      : {args.batch_size}")
    print(f"  lr              : {args.lr}")
    print(f"  gamma           : {args.gamma}")
    print(f"  epsilon_decay   : {args.epsilon_decay}")
    print(f"  num_envs        : {args.num_envs}")
    
    exp_name = args.exp_name or f"scenario{args.scenario}"
    
    config = vars(args)
    config.update({
        "n_episodes": args.episodes,
        "log_dir": args.results_dir,
        "exp_name": exp_name,
        "use_amp": not args.no_amp,
        "learning_starts": getattr(args, 'learning_starts', 5000),
        "target_update_interval": args.target_update
    })
    
    # Train
    trainer = Trainer(config)
    result = trainer.train()
    
    print(f"\nTraining completed!")
    print(f"  Best model: {result['best_model_path']}")
    
    # Copy best model to Result/scenarioX/
    result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              "Result", f"scenario{args.scenario}")
    os.makedirs(result_dir, exist_ok=True)
    best_model_dest = os.path.join(result_dir, "best_model.pt")
    
    if os.path.exists(result['best_model_path']):
        shutil.copy(result['best_model_path'], best_model_dest)
        print(f"  Saved to: {best_model_dest}")
    
    # Generate visualizations
    print("\nGenerating training visualizations...")
    from visualize import plot_training_results
    plot_training_results(result['exp_dir'])
    
    return result


# =============================================================================
# Test Mode
# =============================================================================

def run_test(args):
    """Run test/evaluation mode."""
    from na2q.test import test
    
    class TestArgs:
        def __init__(self, args):
            result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                      "Result", f"scenario{args.scenario}")
            default_model = os.path.join(result_dir, "best_model.pt")
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


# =============================================================================
# Video Mode
# =============================================================================

def run_video(args):
    """Generate video of trained agent."""
    from visualize import generate_video
    
    result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              "Result", f"scenario{args.scenario}")
    model_path = args.model or os.path.join(result_dir, "best_model.pt")
    
    media_dir = os.path.join(result_dir, "media")
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


# =============================================================================
# Visualize Mode
# =============================================================================

def run_visualize(args):
    """Generate visualizations from training results."""
    from visualize import plot_training_results
    
    training_result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                       "training_result", f"scenario{args.scenario}")
    history_dir = os.path.join(training_result_dir, "checkpoints")
    media_dir = os.path.join(training_result_dir, "media")
    os.makedirs(media_dir, exist_ok=True)
    
    if os.path.exists(os.path.join(history_dir, "training_history.npz")):
        plot_training_results(exp_dir=training_result_dir, history_dir=history_dir, media_dir=media_dir)
        print(f"Training charts saved to: {media_dir}")
    else:
        print(f"Error: No training history found at {history_dir}")


# =============================================================================
# Quick Test Mode
# =============================================================================

def run_quick_test(args):
    """Run quick test to verify everything works."""
    from na2q.test import run_quick_test
    run_quick_test()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    from na2q.utils import get_device
    import torch
    
    args = parse_args()
    
    # Auto-detect device
    device = get_device(args.device)
    args.device = device
    
    # Banner
    print("=" * 60)
    print("NA²Q: Neural Attention Additive Q-Learning")
    print("Applied to Directional Sensor Network")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Scenario: {args.scenario}")
    print(f"Device: {device}")
    if device == "cuda":
        if args.gpu_id is not None:
            torch.cuda.set_device(args.gpu_id)
        print(f"  CUDA Device: {torch.cuda.get_device_name()}")
    print("=" * 60)
    
    # Dispatch
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
