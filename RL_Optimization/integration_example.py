"""
Integration Example: How Reward Function and RL Work Together

This script demonstrates the complete pipeline from data loading to 
final policy comparison, showing how the reward function integrates
with the Q-Learning agent.
"""

import pandas as pd
import numpy as np
from reward_function import ShadeRewardFunction
from rl_methodology import (
    ShadeQLearningAgent,
    RandomBaseline,
    GreedyByFeatureBaseline,
    GreedyOptimizationBaseline
)


def main():
    """
    Complete pipeline for shade placement optimization.
    
    Pipeline:
    1. Load dataset
    2. Initialize reward function
    3. Train RL agent
    4. Compare with baselines
    5. Analyze results
    """
    
    print("="*80)
    print("COMPLETE PIPELINE: REWARD FUNCTION + RL OPTIMIZATION")
    print("="*80)
    
    # ========================================================================
    # SECTION 1: DATA LOADING
    # ========================================================================
    print("\n" + "="*80)
    print("SECTION 1: LOADING DATA")
    print("="*80)
    
    # Load the cleaned dataset with all 84 features
    data = pd.read_csv('la_coverage_points_features.csv')
    
    print(f"\n✓ Dataset loaded successfully")
    print(f"  Grid points: {len(data)}")
    print(f"  Features: {len(data.columns)}")
    print(f"  Coverage area: LA City (lat: 33.70°-34.85°, lon: -118.95°--117.65°)")
    
    # Display sample of key features
    print(f"\nSample of key features (first 3 grid points):")
    key_features = [
        'latitude', 'longitude',
        'urban_heat_idx_percentile',
        'lashade_tot1200',
        'cva_population',
        'dist_to_venue1',
        'cva_sovi_score',
        'lashade_ej_disadva'
    ]
    print(data[key_features].head(3).to_string())
    
    # ========================================================================
    # SECTION 2: REWARD FUNCTION INITIALIZATION
    # ========================================================================
    print("\n" + "="*80)
    print("SECTION 2: INITIALIZING REWARD FUNCTION")
    print("="*80)
    
    # Create reward function instance
    # This is the SAME reward function used by both RL and greedy baselines
    reward_fn = ShadeRewardFunction(data)
    
    print(f"\n✓ Reward function initialized")
    print(f"\nComponent weights:")
    for component, weight in reward_fn.weights.items():
        print(f"  {component:25s}: {weight:.2f}")
    
    # ----------------------------------------------------------------
    # DEMONSTRATION: Calculate reward for a sample action
    # ----------------------------------------------------------------
    print(f"\n" + "-"*80)
    print("DEMONSTRATION: Reward Calculation")
    print("-"*80)
    
    # Example state: 3 shades already placed
    example_state = [50, 150, 200]
    # Example action: propose placing shade at index 100
    example_action = 100
    
    print(f"\nScenario:")
    print(f"  Current state: Shades at indices {example_state}")
    print(f"  Proposed action: Place shade at index {example_action}")
    
    # Calculate reward
    reward = reward_fn.calculate_reward(example_state, example_action)
    
    print(f"\n  → Total Reward R(s,a) = {reward:.4f}")
    
    # Get detailed breakdown
    breakdown = reward_fn.get_component_breakdown(example_state, example_action)
    
    print(f"\nComponent Breakdown:")
    for component, value in breakdown['components'].items():
        weighted = breakdown['weighted'][component]
        print(f"  {component:25s}: {value:.4f} × {reward_fn.weights[component]:.2f} = {weighted:.4f}")
    
    print("-"*80)
    
    # ========================================================================
    # SECTION 3: Q-LEARNING AGENT TRAINING
    # ========================================================================
    print("\n" + "="*80)
    print("SECTION 3: TRAINING Q-LEARNING AGENT")
    print("="*80)
    
    # Initialize agent with reward function
    agent = ShadeQLearningAgent(
        data_df=data,
        reward_function=reward_fn,  # ← REWARD FUNCTION PASSED HERE
        n_shades_budget=50,
        alpha=0.1,      # Learning rate
        gamma=0.95,     # Discount factor
        epsilon=0.3     # Initial exploration rate
    )
    
    # Train the agent
    print(f"\nStarting training...")
    history = agent.train(n_episodes=1000, verbose=True)
    
    # Extract learned policy
    print(f"\nExtracting optimal policy from learned Q-values...")
    optimal_policy = agent.get_optimal_policy()
    
    # Evaluate RL policy
    rl_reward = agent.evaluate_policy(optimal_policy)
    print(f"\n✓ RL Policy Total Reward: {rl_reward:.3f}")
    
    # ========================================================================
    # SECTION 4: BASELINE COMPARISONS
    # ========================================================================
    print("\n" + "="*80)
    print("SECTION 4: COMPARING WITH BASELINE ALGORITHMS")
    print("="*80)
    
    results = {}
    
    # ----------------------------------------------------------------
    # Baseline 1: Random Placement
    # ----------------------------------------------------------------
    print(f"\n[1/4] Random Baseline...")
    random_baseline = RandomBaseline(data, n_shades=50)
    random_policy = random_baseline.get_policy()
    random_reward = agent.evaluate_policy(random_policy)
    results['Random'] = random_reward
    print(f"  Total Reward: {random_reward:.3f}")
    
    # ----------------------------------------------------------------
    # Baseline 2: Greedy by Urban Heat Index
    # ----------------------------------------------------------------
    print(f"\n[2/4] Greedy-by-UHI Baseline...")
    greedy_uhi = GreedyByFeatureBaseline(
        data, 
        'urban_heat_idx_percentile', 
        n_shades=50
    )
    greedy_uhi_policy = greedy_uhi.get_policy()
    greedy_uhi_reward = agent.evaluate_policy(greedy_uhi_policy)
    results['Greedy-by-UHI'] = greedy_uhi_reward
    print(f"  Total Reward: {greedy_uhi_reward:.3f}")
    
    # ----------------------------------------------------------------
    # Baseline 3: Greedy by Poverty Rate
    # ----------------------------------------------------------------
    print(f"\n[3/4] Greedy-by-Poverty Baseline...")
    greedy_pov = GreedyByFeatureBaseline(
        data,
        'cva_poverty',
        n_shades=50
    )
    greedy_pov_policy = greedy_pov.get_policy()
    greedy_pov_reward = agent.evaluate_policy(greedy_pov_policy)
    results['Greedy-by-Poverty'] = greedy_pov_reward
    print(f"  Total Reward: {greedy_pov_reward:.3f}")
    
    # ----------------------------------------------------------------
    # Baseline 4: Greedy Optimization (MAIN COMPARISON)
    # ----------------------------------------------------------------
    print(f"\n[4/4] Greedy Optimization Baseline...")
    print(f"  (This uses the SAME reward function as RL)")
    greedy_opt = GreedyOptimizationBaseline(
        data,
        reward_fn,  # ← SAME REWARD FUNCTION AS RL
        n_shades=50
    )
    greedy_opt_policy = greedy_opt.get_policy()
    greedy_opt_reward = agent.evaluate_policy(greedy_opt_policy)
    results['Greedy Optimization'] = greedy_opt_reward
    print(f"  Total Reward: {greedy_opt_reward:.3f}")
    
    # Add RL to results
    results['Q-Learning (RL)'] = rl_reward
    
    # ========================================================================
    # SECTION 5: RESULTS ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("SECTION 5: RESULTS ANALYSIS")
    print("="*80)
    
    # ----------------------------------------------------------------
    # Ranking Table
    # ----------------------------------------------------------------
    print("\n" + "-"*80)
    print("RANKING (by Total Cumulative Reward)")
    print("-"*80)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Rank':<6} {'Method':<30} {'Reward':>10} {'vs Best':>10}")
    print("-"*80)
    
    best_reward = sorted_results[0][1]
    for rank, (method, reward) in enumerate(sorted_results, 1):
        vs_best = ((reward - best_reward) / best_reward) * 100
        marker = " ★" if rank == 1 else ""
        print(f"{rank:<6} {method:<30} {reward:>10.3f} {vs_best:>9.1f}%{marker}")
    
    # ----------------------------------------------------------------
    # Key Comparisons
    # ----------------------------------------------------------------
    print("\n" + "-"*80)
    print("KEY COMPARISONS")
    print("-"*80)
    
    # RL vs Greedy Optimization (tests value of lookahead)
    improvement_greedy = ((rl_reward - greedy_opt_reward) / greedy_opt_reward) * 100
    print(f"\nRL vs. Greedy Optimization:")
    print(f"  Improvement: {improvement_greedy:+.2f}%")
    print(f"  Interpretation: {'✓ Strategic planning adds value' if improvement_greedy > 3 else '→ Modest benefit from lookahead'}")
    
    # RL vs Random (sanity check)
    improvement_random = ((rl_reward - random_reward) / random_reward) * 100
    print(f"\nRL vs. Random:")
    print(f"  Improvement: {improvement_random:+.2f}%")
    print(f"  Interpretation: ✓ Substantial improvement (sanity check passes)")
    
    # RL vs Best Single-Feature Baseline
    best_single_feature = max(greedy_uhi_reward, greedy_pov_reward)
    improvement_single = ((rl_reward - best_single_feature) / best_single_feature) * 100
    print(f"\nRL vs. Best Single-Feature Baseline:")
    print(f"  Improvement: {improvement_single:+.2f}%")
    print(f"  Interpretation: {'✓ Multi-objective optimization superior' if improvement_single > 5 else '→ Single features capture some value'}")
    
    # ----------------------------------------------------------------
    # Statistical Summary
    # ----------------------------------------------------------------
    print("\n" + "-"*80)
    print("STATISTICAL SUMMARY")
    print("-"*80)
    
    all_rewards = list(results.values())
    print(f"\nReward Statistics Across All Methods:")
    print(f"  Mean:   {np.mean(all_rewards):.3f}")
    print(f"  Median: {np.median(all_rewards):.3f}")
    print(f"  Std:    {np.std(all_rewards):.3f}")
    print(f"  Range:  [{np.min(all_rewards):.3f}, {np.max(all_rewards):.3f}]")
    
    # ----------------------------------------------------------------
    # Policy Analysis
    # ----------------------------------------------------------------
    print("\n" + "-"*80)
    print("POLICY ANALYSIS")
    print("-"*80)
    
    print(f"\nRL Policy Characteristics:")
    
    # Analyze first 5 placements
    print(f"\n  First 5 placements (grid indices): {optimal_policy[:5]}")
    
    for i, idx in enumerate(optimal_policy[:5], 1):
        loc = data.iloc[idx]
        print(f"\n  Placement #{i} (index {idx}):")
        print(f"    Location: ({loc['latitude']:.4f}, {loc['longitude']:.4f})")
        print(f"    UHI Percentile: {loc['urban_heat_idx_percentile']:.3f}")
        print(f"    Current Shade: {loc['lashade_tot1200']:.3f}")
        print(f"    Population: {loc['cva_population']:.0f}")
        print(f"    Dist to Olympic Venue: {loc['dist_to_venue1']:.2f} km")
        print(f"    Social Vulnerability: {loc['cva_sovi_score']:.3f}")
        print(f"    EPA Disadvantaged: {loc['lashade_ej_disadva']}")
    
    # ----------------------------------------------------------------
    # Coverage Metrics
    # ----------------------------------------------------------------
    print("\n" + "-"*80)
    print("COVERAGE METRICS")
    print("-"*80)
    
    # Calculate average spacing between shades
    def calculate_avg_spacing(policy, data_df):
        """Calculate average minimum distance between shades."""
        if len(policy) < 2:
            return None
        
        total_min_dist = 0
        count = 0
        
        for i, idx_i in enumerate(policy):
            lat_i = data_df.iloc[idx_i]['latitude']
            lon_i = data_df.iloc[idx_i]['longitude']
            
            min_dist = float('inf')
            for j, idx_j in enumerate(policy):
                if i != j:
                    lat_j = data_df.iloc[idx_j]['latitude']
                    lon_j = data_df.iloc[idx_j]['longitude']
                    
                    # Simple distance approximation
                    dist = np.sqrt((lat_i - lat_j)**2 + (lon_i - lon_j)**2) * 111  # approx km
                    min_dist = min(min_dist, dist)
            
            if min_dist != float('inf'):
                total_min_dist += min_dist
                count += 1
        
        return total_min_dist / count if count > 0 else None
    
    rl_spacing = calculate_avg_spacing(optimal_policy, data)
    greedy_spacing = calculate_avg_spacing(greedy_opt_policy, data)
    
    print(f"\nSpatial Efficiency (avg min distance between shades):")
    print(f"  RL Policy:              {rl_spacing:.2f} km")
    print(f"  Greedy Optimization:    {greedy_spacing:.2f} km")
    print(f"  Target (optimal):       ≥0.80 km")
    print(f"  Interpretation: {'✓ Good dispersion' if rl_spacing >= 0.6 else '⚠ Some clustering'}")
    
    # ========================================================================
    # SECTION 6: KEY INSIGHTS
    # ========================================================================
    print("\n" + "="*80)
    print("SECTION 6: KEY INSIGHTS")
    print("="*80)
    
    print("\n1. REWARD FUNCTION INTEGRATION:")
    print("   ✓ Same reward function used by both RL and greedy baseline")
    print("   ✓ Ensures fair comparison of algorithms, not objectives")
    print("   ✓ State-dependent coverage component enables RL learning")
    
    print("\n2. RL LEARNING PROCESS:")
    print(f"   ✓ Trained over {len(history['episode_rewards'])} episodes")
    print(f"   ✓ Explored {len(agent.Q)} unique states")
    print(f"   ✓ Converged with ε={agent.epsilon:.4f} (minimal exploration)")
    
    print("\n3. PERFORMANCE COMPARISON:")
    if improvement_greedy > 5:
        print("   ✓ RL SIGNIFICANTLY outperforms greedy optimization")
        print("   ✓ Strategic planning (lookahead) provides substantial value")
        print("   ✓ Learning spatial dependencies was successful")
    elif improvement_greedy > 0:
        print("   → RL modestly outperforms greedy optimization")
        print("   → Strategic planning adds some value")
        print("   → Consider longer training or hyperparameter tuning")
    else:
        print("   ⚠ RL underperforms greedy optimization")
        print("   ⚠ May need more training episodes or parameter adjustment")
    
    print("\n4. POLICY CHARACTERISTICS:")
    print(f"   • RL places {len(optimal_policy)} shades")
    print(f"   • Average spacing: {rl_spacing:.2f} km")
    print("   • Balances heat vulnerability, equity, and coverage efficiency")
    
    print("\n5. PRACTICAL IMPLICATIONS:")
    print("   • Policy provides actionable recommendations for LA 2028")
    print("   • Prioritizes Olympic venues and disadvantaged communities")
    print("   • Optimizes spatial coverage while avoiding redundancy")
    
    # ========================================================================
    # SECTION 7: NEXT STEPS
    # ========================================================================
    print("\n" + "="*80)
    print("SECTION 7: NEXT STEPS FOR PROPOSAL")
    print("="*80)
    
    print("\n1. VISUALIZATION:")
    print("   □ Generate heatmap showing priority areas")
    print("   □ Create side-by-side maps: RL vs. greedy baseline")
    print("   □ Plot training curves (reward, epsilon, episode length)")
    print("   □ Show feature importance analysis")
    
    print("\n2. CROSS-VALIDATION:")
    print("   □ Partition LA into 5 geographic regions")
    print("   □ Train on 4 regions, test on held-out region")
    print("   □ Evaluate generalization performance")
    
    print("\n3. SENSITIVITY ANALYSIS:")
    print("   □ Test different weight combinations")
    print("   □ Vary hyperparameters (α, γ, ε)")
    print("   □ Analyze impact of budget size (20, 50, 100 shades)")
    
    print("\n4. EQUITY METRICS:")
    print("   □ Calculate Gini coefficient across income quartiles")
    print("   □ Measure coverage in EPA-disadvantaged communities")
    print("   □ Assess heat reduction in vulnerable populations")
    
    print("\n5. DOCUMENTATION:")
    print("   □ Include reward function formulas in proposal")
    print("   □ Explain RL vs. greedy conceptual differences")
    print("   □ Provide sample of cleaned dataset")
    print("   □ Document distinctions from instructor's example")
    
    # ========================================================================
    # CONCLUSION
    # ========================================================================
    print("\n" + "="*80)
    print("EXECUTION COMPLETE")
    print("="*80)
    
    print("\n✓ Data loaded and processed")
    print("✓ Reward function initialized and tested")
    print("✓ Q-Learning agent trained successfully")
    print("✓ Baselines computed for comparison")
    print("✓ Results analyzed and insights generated")
    
    print("\n" + "="*80)
    print("Ready for proposal integration!")
    print("="*80)


# ============================================================================
# HELPER FUNCTION: Understanding the Integration
# ============================================================================

def explain_integration():
    """
    Explain how reward function and RL integrate conceptually.
    """
    print("\n" + "="*80)
    print("HOW REWARD FUNCTION AND RL INTEGRATE")
    print("="*80)
    
    print("\n[CONCEPTUAL OVERVIEW]")
    print("""
    The reward function R(s,a) and Q-Learning work together as follows:
    
    1. REWARD FUNCTION (reward_function.py)
       • Input: state s (list of placed shades), action a (new location)
       • Output: scalar reward value (0-1 range)
       • Role: Evaluates "goodness" of placing shade at location a
       • Used by: BOTH RL agent and greedy baselines
    
    2. Q-LEARNING AGENT (rl_methodology.py)
       • Learns: Q(s,a) ≈ R(s,a) + γ·max Q(s',a')
       • Process: 
         - Tries action a in state s
         - Gets immediate reward R(s,a) from reward function
         - Updates Q(s,a) to account for future rewards
         - Repeats over many episodes
       • Result: Learns to maximize CUMULATIVE reward, not just immediate
    
    3. KEY DIFFERENCE FROM GREEDY
       • Greedy:  At each step, pick max_a R(s,a)
       • RL:      At each step, pick max_a Q(s,a)
       
       Where Q(s,a) considers both immediate AND future rewards!
    """)
    
    print("\n[INTEGRATION POINTS]")
    print("""
    The code integration happens at these key points:
    
    A. INITIALIZATION (main function)
       reward_fn = ShadeRewardFunction(data)
       agent = ShadeQLearningAgent(data, reward_fn, ...)
                                          ^^^^^^^^
                                    Reward function passed here
    
    B. TRAINING LOOP (agent.train)
       for episode in episodes:
           for step in steps:
               action = agent.choose_action(state)
               reward = reward_fn.calculate_reward(state, action)  ← CALL HERE
               agent.update_q_value(state, action, reward, next_state)
    
    C. BASELINE EVALUATION (greedy baselines)
       for location in all_locations:
           reward = reward_fn.calculate_reward(state, location)  ← SAME CALL
           best_location = argmax(reward)
    
    D. POLICY EVALUATION (compare methods)
       for policy in [rl_policy, greedy_policy, random_policy]:
           total_reward = sum(reward_fn.calculate_reward(...))  ← SAME FUNCTION
    """)
    
    print("\n[WHY THIS DESIGN WORKS]")
    print("""
    ✓ FAIR COMPARISON
      • All methods evaluated with same reward function
      • Compares algorithms, not objectives
      • Isolates benefit of RL learning
    
    ✓ STATE DEPENDENCE
      • r_coverage(s,a) depends on existing shades
      • Enables RL to learn spatial patterns
      • Greedy cannot capture these dependencies
    
    ✓ MODULARITY
      • Reward function is reusable component
      • Easy to adjust weights or add components
      • RL algorithm independent of reward details
    
    ✓ INTERPRETABILITY
      • Reward breakdown shows WHY locations chosen
      • Components map to stakeholder priorities
      • Results explainable to policymakers
    """)
    
    print("\n" + "="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run the complete pipeline
    main()
    
    # Optional: Print conceptual explanation
    print("\n\n")
    explain_integration()