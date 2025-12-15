import numpy as np
import os

history_path = 'Result/scenario1/history/training_history.npz'
if os.path.exists(history_path):
    data = np.load(history_path)
    rewards = data['episode_rewards']
    coverage = data['coverage_rates']
    
    chunk_size = 1000
    num_chunks = len(rewards) // chunk_size
    
    print(f"Total episodes: {len(rewards)}")
    print(f"Analyzing in {num_chunks} chunks of {chunk_size} episodes:")
    print(f"{'Chunk':<10} {'Avg Reward':<15} {'Avg Coverage':<15}")
    print("-" * 40)
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        r_chunk = rewards[start:end]
        c_chunk = coverage[start:end]
        print(f"{i:<10} {np.mean(r_chunk):<15.4f} {np.mean(c_chunk):<15.4f}")
        
    # Last chunk if any remainder
    if len(rewards) % chunk_size != 0:
        r_chunk = rewards[num_chunks*chunk_size:]
        c_chunk = coverage[num_chunks*chunk_size:]
        print(f"{'Last':<10} {np.mean(r_chunk):<15.4f} {np.mean(c_chunk):<15.4f}")
else:
    print(f"File not found: {history_path}")
