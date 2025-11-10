"""
Main Training Script for LA Shade Placement RL Optimization

This script runs the complete training pipeline:
1. Load cleaned data
2. Initialize reward function
3. Train Q-Learning agent
4. Compare with baselines
5. Save results and visualizations

Usage:
    python train_rl.py [--episodes 1000] [--budget 50] [--output results/]
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

from reward_function import ShadeRewardFunction
from rl_methodology import ShadeQLearningAgent, RandomBaseline, GreedyByFeatureBaseline, GreedyOptimizationBaseline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train RL agent for shade placement optimization')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes (default: 1000)')
    parser.add_argument('--budget', type=int, default=50,
                       help='Number of shades to place (default: 50)')
    parser.add_argument('--output', type=str, default='../results',
                       help='Output directory for results (default: ../results)')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Learning rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.95,
                       help='Discount factor (default: 0.95)')
    parser.add_argument('--epsilon', type=float, default=0.3,
                       help='Initial exploration rate (default: 0.3)')
    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    print("="*80)
    print("LA 2028 OLYMPICS SHADE PLACEMENT - RL TRAINING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Episodes: {args.episodes}")
    print(f"  Shade budget: {args.budget}")
    print(f"  Learning rate (α): {args.alpha}")
    print(f"  Discount factor (γ): {args.gamma}")
    print(f"  Initial exploration (ε): {args.epsilon}")
    print(f"  Output directory: {run_dir}")

    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: LOADING CLEANED DATA")
    print("="*80)

    data_path = "../shade_optimization_data_cleaned.csv"
    if not os.path.exists(data_path):
        # Try eda_outputs path
        data_path = "../eda_outputs/data_cleaned.csv"

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "Cleaned data not found. Please run eda_full.py first to generate cleaned data."
        )

    print(f"\n✓ Loading data from: {data_path}")
    data = pd.read_csv(data_path)
    print(f"  Shape: {data.shape}")
    print(f"  Features: {list(data.columns[:10])}... ({len(data.columns)} total)")

    # ========================================================================
    # STEP 2: INITIALIZE REWARD FUNCTION
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: INITIALIZING REWARD FUNCTION")
    print("="*80)

    reward_fn = ShadeRewardFunction(data)

    # Save reward function configuration
    config = {
        'timestamp': timestamp,
        'data_shape': data.shape,
        'reward_weights': reward_fn.weights,
        'optimal_spacing_km': reward_fn.optimal_spacing,
        'hyperparameters': {
            'alpha': args.alpha,
            'gamma': args.gamma,
            'epsilon': args.epsilon,
            'episodes': args.episodes,
            'budget': args.budget
        }
    }

    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # ========================================================================
    # STEP 3: TRAIN Q-LEARNING AGENT
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: TRAINING Q-LEARNING AGENT")
    print("="*80)

    agent = ShadeQLearningAgent(
        data_df=data,
        reward_function=reward_fn,
        n_shades_budget=args.budget,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon
    )

    # Train
    print(f"\nStarting training for {args.episodes} episodes...")
    history = agent.train(n_episodes=args.episodes, verbose=True)

    # Extract optimal policy
    print(f"\nExtracting optimal policy...")
    optimal_policy = agent.get_optimal_policy()
    rl_reward = agent.evaluate_policy(optimal_policy)

    print(f"\n✓ RL Training Complete!")
    print(f"  Final ε: {agent.epsilon:.4f}")
    print(f"  States explored: {len(agent.Q)}")
    print(f"  Optimal policy: {len(optimal_policy)} shades")
    print(f"  Total reward: {rl_reward:.4f}")

    # ========================================================================
    # STEP 4: BASELINE COMPARISONS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: COMPARING WITH BASELINES")
    print("="*80)

    results = {'Q-Learning (RL)': rl_reward}
    policies = {'Q-Learning (RL)': optimal_policy}

    # Baseline 1: Random
    print(f"\n[1/4] Random Baseline...")
    random_baseline = RandomBaseline(data, n_shades=args.budget)
    random_policy = random_baseline.get_policy()
    random_reward = agent.evaluate_policy(random_policy)
    results['Random'] = random_reward
    policies['Random'] = random_policy
    print(f"  Reward: {random_reward:.4f}")

    # Baseline 2: Greedy by env_exposure_index
    print(f"\n[2/4] Greedy-by-Environmental-Exposure...")
    greedy_env = GreedyByFeatureBaseline(data, 'env_exposure_index', n_shades=args.budget)
    greedy_env_policy = greedy_env.get_policy()
    greedy_env_reward = agent.evaluate_policy(greedy_env_policy)
    results['Greedy-by-Env-Exposure'] = greedy_env_reward
    policies['Greedy-by-Env-Exposure'] = greedy_env_policy
    print(f"  Reward: {greedy_env_reward:.4f}")

    # Baseline 3: Greedy by SoVI score
    print(f"\n[3/4] Greedy-by-Social-Vulnerability...")
    greedy_sovi = GreedyByFeatureBaseline(data, 'cva_sovi_score', n_shades=args.budget)
    greedy_sovi_policy = greedy_sovi.get_policy()
    greedy_sovi_reward = agent.evaluate_policy(greedy_sovi_policy)
    results['Greedy-by-SoVI'] = greedy_sovi_reward
    policies['Greedy-by-SoVI'] = greedy_sovi_policy
    print(f"  Reward: {greedy_sovi_reward:.4f}")

    # Baseline 4: Greedy Optimization (MAIN COMPARISON)
    print(f"\n[4/4] Greedy Optimization (using reward function)...")
    greedy_opt = GreedyOptimizationBaseline(data, reward_fn, n_shades=args.budget)
    greedy_opt_policy = greedy_opt.get_policy()
    greedy_opt_reward = agent.evaluate_policy(greedy_opt_policy)
    results['Greedy Optimization'] = greedy_opt_reward
    policies['Greedy Optimization'] = greedy_opt_policy
    print(f"  Reward: {greedy_opt_reward:.4f}")

    # ========================================================================
    # STEP 5: RESULTS ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: RESULTS ANALYSIS")
    print("="*80)

    # Print ranking
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'Rank':<6} {'Method':<30} {'Reward':>10} {'vs Best':>10}")
    print("-"*80)

    best_reward = sorted_results[0][1]
    for rank, (method, reward) in enumerate(sorted_results, 1):
        vs_best = ((reward - best_reward) / best_reward) * 100
        marker = " ★" if rank == 1 else ""
        print(f"{rank:<6} {method:<30} {reward:>10.4f} {vs_best:>9.1f}%{marker}")

    # Key comparisons
    improvement_greedy = ((rl_reward - greedy_opt_reward) / greedy_opt_reward) * 100
    improvement_random = ((rl_reward - random_reward) / random_reward) * 100

    print(f"\nKey Improvements:")
    print(f"  RL vs Greedy Optimization: {improvement_greedy:+.2f}%")
    print(f"  RL vs Random:              {improvement_random:+.2f}%")

    # Save results
    results_df = pd.DataFrame([
        {'method': k, 'total_reward': v}
        for k, v in sorted_results
    ])
    results_df.to_csv(run_dir / 'results_summary.csv', index=False)

    # Save policies
    for method, policy in policies.items():
        policy_df = data.iloc[policy][['latitude', 'longitude', 'env_exposure_index',
                                        'canopy_gap', 'cva_population', 'cva_sovi_score']]
        policy_df.to_csv(run_dir / f'policy_{method.replace(" ", "_").lower()}.csv', index=False)

    # ========================================================================
    # STEP 6: GENERATE VISUALIZATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: GENERATING VISUALIZATIONS")
    print("="*80)

    # 1. Training curves
    print("\n[1/4] Training curves...")
    fig = agent.plot_training_curves(save_path=run_dir / 'training_curves.png')
    plt.close(fig)

    # 2. Results comparison bar chart
    print("[2/4] Results comparison...")
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = [x[0] for x in sorted_results]
    rewards = [x[1] for x in sorted_results]
    colors = ['#2ecc71' if m == 'Q-Learning (RL)' else '#95a5a6' for m in methods]
    ax.barh(methods, rewards, color=colors)
    ax.set_xlabel('Total Cumulative Reward', fontsize=12)
    ax.set_title('Algorithm Comparison: Total Reward', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / 'results_comparison.png', dpi=300)
    plt.close()

    # 3. Spatial distribution map
    print("[3/4] Spatial distribution map...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # RL policy
    rl_locs = data.iloc[optimal_policy]
    axes[0].scatter(data['longitude'], data['latitude'], c='lightgray', s=1, alpha=0.3, label='All grid points')
    axes[0].scatter(rl_locs['longitude'], rl_locs['latitude'], c=rl_locs['env_exposure_index'],
                   cmap='YlOrRd', s=100, edgecolors='black', linewidth=0.5, label='RL placements')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].set_title('Q-Learning Policy', fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(alpha=0.3)

    # Greedy policy
    greedy_locs = data.iloc[greedy_opt_policy]
    axes[1].scatter(data['longitude'], data['latitude'], c='lightgray', s=1, alpha=0.3, label='All grid points')
    axes[1].scatter(greedy_locs['longitude'], greedy_locs['latitude'], c=greedy_locs['env_exposure_index'],
                   cmap='YlOrRd', s=100, edgecolors='black', linewidth=0.5, label='Greedy placements')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].set_title('Greedy Optimization Policy', fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.3)

    plt.suptitle('Spatial Distribution Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(run_dir / 'spatial_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Feature importance heatmap
    print("[4/4] Feature importance analysis...")
    rl_locs = data.iloc[optimal_policy]
    feature_stats = pd.DataFrame({
        'env_exposure_index': [rl_locs['env_exposure_index'].mean(), data['env_exposure_index'].mean()],
        'canopy_gap': [rl_locs['canopy_gap'].mean(), data['canopy_gap'].mean()],
        'cva_sovi_score': [rl_locs['cva_sovi_score'].mean(), data['cva_sovi_score'].mean()],
        'cva_poverty': [rl_locs['cva_poverty'].mean(), data['cva_poverty'].mean()],
        'lashade_pctpoc': [rl_locs['lashade_pctpoc'].mean(), data['lashade_pctpoc'].mean()],
    }, index=['RL Selected', 'All Locations'])

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(feature_stats.T, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'Mean Value'})
    ax.set_title('Feature Comparison: RL Selected vs All Locations', fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    plt.savefig(run_dir / 'feature_importance.png', dpi=300)
    plt.close()

    print(f"\n✓ All visualizations saved to: {run_dir}")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)

    print(f"\n📁 Results saved to: {run_dir}")
    print(f"\n📊 Generated files:")
    print(f"  - config.json                    (configuration)")
    print(f"  - results_summary.csv            (comparison table)")
    print(f"  - policy_*.csv                   (placement coordinates)")
    print(f"  - training_curves.png            (RL learning progress)")
    print(f"  - results_comparison.png         (algorithm comparison)")
    print(f"  - spatial_distribution.png       (map visualization)")
    print(f"  - feature_importance.png         (feature analysis)")

    print(f"\n🎯 Performance Summary:")
    print(f"  Best Method: {sorted_results[0][0]}")
    print(f"  Best Reward: {sorted_results[0][1]:.4f}")
    print(f"  RL vs Greedy: {improvement_greedy:+.2f}% improvement")

    if improvement_greedy > 5:
        print(f"\n✓ Strong performance! RL significantly outperforms greedy optimization.")
    elif improvement_greedy > 0:
        print(f"\n→ Modest improvement. Consider tuning hyperparameters or increasing episodes.")
    else:
        print(f"\n⚠ RL underperforms greedy. Review hyperparameters and increase training time.")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
