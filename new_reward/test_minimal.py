"""Minimal test to debug import issues."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Step 1: Importing pandas...")
import pandas as pd
print("✓ Pandas imported")

print("\nStep 2: Loading data...")
data_path = Path(__file__).parent.parent / 'shade_optimization_data_usc_simple_features.csv'
data = pd.read_csv(data_path)
print(f"✓ Loaded {len(data)} rows")

print("\nStep 3: Importing regional filter...")
from new_reward.regional_filters import filter_region
print("✓ Regional filter imported")

print("\nStep 4: Filtering to USC...")
usc_data = filter_region(data, 'USC')
print(f"✓ Filtered to {len(usc_data)} points")

print("\nStep 5: Importing Approach 1...")
from new_reward.approaches.approach1_weighted import EnhancedWeightedSumReward
print("✓ Approach 1 imported")

print("\nStep 6: Initializing reward function...")
reward_func = EnhancedWeightedSumReward(usc_data, region='USC')
print("✓ Reward function initialized")

print("\nStep 7: Testing single reward calculation...")
reward = reward_func.calculate_reward([], 0)
print(f"✓ Reward for index 0: {reward:.4f}")

print("\nStep 8: Testing greedy optimization (k=3)...")
state = []
for i in range(3):
    best_idx = None
    best_reward = -float('inf')

    for idx in range(min(50, len(usc_data))):  # Only check first 50 points
        if idx in state:
            continue
        r = reward_func.calculate_reward(state, idx)
        if r > best_reward:
            best_reward = r
            best_idx = idx

    state.append(best_idx)
    print(f"  Iteration {i+1}: Selected index {best_idx}, reward={best_reward:.4f}")

print("\n✓ All tests passed!")
