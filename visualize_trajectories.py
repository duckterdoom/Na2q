
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from environments.environment import DSNEnv
from na2q.models.agent import NA2QAgent
from train_config import DEFAULT_CONFIG

def visualize_trajectories(model_path, scenario=1, steps=100):
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")
    
    print(f"🎥 Loading model from {model_path}...")
    
    # Load Config (Needed for Model Init)
    config = DEFAULT_CONFIG.copy()
    
    # Init Env & Agent
    env = DSNEnv(scenario=scenario)
    agent = NA2QAgent(env.n_sensors, env.n_targets, env.n_actions, 
                      config["obs_dim"], config["state_dim"], config, device)
    
    # Load Weights
    agent.load(model_path)
    
    # 2. Run Episode
    obs_list, _ = env.reset()
    observations = np.stack(obs_list)
    agent.init_hidden(1)
    prev_actions = np.zeros(env.n_sensors, dtype=np.int64)
    
    # History
    history = {i: {"x": [], "y": []} for i in range(env.n_sensors)}
    target_history = {j: {"x": [], "y": []} for j in range(env.n_targets)}
    
    print(f"🏃 Running simulation for {steps} steps...")
    
    for _ in range(steps):
        # Record Positions
        for i in range(env.n_sensors):
            pos = env.sensor_positions[i]
            history[i]["x"].append(pos[0])
            history[i]["y"].append(pos[1])
            
        for j in range(env.n_targets):
            pos = env.target_positions[j]
            target_history[j]["x"].append(pos[0])
            target_history[j]["y"].append(pos[1])
            
        # Action
        avail_actions = np.stack(env.get_avail_actions())
        actions = agent.select_actions(observations, prev_actions, avail_actions, evaluate=True)
        
        # Step
        next_obs, _, done, truncated, _ = env.step(actions.tolist())
        observations = np.stack(next_obs)
        prev_actions = actions
        
        if done or truncated:
            break

    # 3. Plot
    print("🎨 Generating plot...")
    plt.figure(figsize=(10, 10))
    
    # Plot Field
    plt.xlim(-2, env.field_size + 2)
    plt.ylim(-2, env.field_size + 2)
    plt.gca().set_aspect('equal')
    plt.grid(True, alpha=0.3)
    
    # Plot Targets (Red Dots)
    for j in range(env.n_targets):
        plt.plot(target_history[j]["x"], target_history[j]["y"], 'ro', alpha=0.5, markersize=8, label="Target" if j==0 else "")
    
    # Plot Sensors (Blue Lines)
    colors = plt.cm.jet(np.linspace(0, 1, env.n_sensors))
    for i in range(env.n_sensors):
        plt.plot(history[i]["x"], history[i]["y"], '-', color=colors[i], linewidth=2, alpha=0.7, label=f"Sensor {i}")
        plt.plot(history[i]["x"][-1], history[i]["y"][-1], 'x', color=colors[i], markersize=10) # End point
        
    plt.title(f"Sensor Trajectories (Scenario {scenario})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Consolidate Output Path
    media_dir = os.path.join("Result", f"scenario{scenario}", "media")
    os.makedirs(media_dir, exist_ok=True)
    
    output_path = os.path.join(media_dir, "trajectory_plot.png")
    plt.savefig(output_path, dpi=150)
    print(f"✅ Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--scenario", type=int, default=1)
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
    else:
        visualize_trajectories(args.model, args.scenario)
