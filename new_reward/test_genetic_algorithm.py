"""
Quick test script to run Genetic Algorithm and compare with Greedy.

This will show if GA can beat Greedy on your actual data.
"""

import json
import time
from approaches.approach1_weighted import EnhancedWeightedSumReward
from approaches.approach2_hierarchical import MultiplicativeHierarchicalReward
from methods.greedy import greedy_optimization
from methods.genetic_algorithm import genetic_algorithm_optimization
from evaluation.metrics import evaluate_solution
import pandas as pd


def load_data():
    """Load grid data (adjust path as needed)."""
    # Try to find the data file
    import os
    possible_paths = [
        'data/grid_data.csv',
        '../data/grid_data.csv',
        '../../data/grid_data.csv',
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading data from: {path}")
            return pd.read_csv(path)

    raise FileNotFoundError("Could not find grid_data.csv. Please specify path.")


def run_comparison(approach_name, reward_function, k=10):
    """Run Greedy vs GA comparison."""

    print(f"\n{'='*80}")
    print(f"Testing: {approach_name}")
    print(f"{'='*80}\n")

    # 1. Run Greedy (baseline)
    print("1. Running Greedy (baseline)...")
    start = time.time()
    greedy_solution = greedy_optimization(reward_function, k, verbose=True)
    greedy_time = time.time() - start

    greedy_metrics = evaluate_solution(greedy_solution, reward_function.data)

    print(f"\n✓ Greedy completed in {greedy_time:.2f}s")
    print(f"  Heat: {greedy_metrics['heat_sum']:.2f}")
    print(f"  Socio-Vuln: {greedy_metrics['socio_sum']:.2f}")
    print(f"  Population: {greedy_metrics['population_served']:,.0f}")
    print(f"  Olympic: {greedy_metrics['olympic_coverage']:.1f}%")
    print(f"  Equity (Gini): {greedy_metrics['equity_gini']:.3f}")

    # 2. Run Genetic Algorithm
    print("\n2. Running Genetic Algorithm...")
    start = time.time()
    ga_solution = genetic_algorithm_optimization(
        reward_function,
        k=k,
        population_size=200,
        generations=300,
        mutation_rate=0.15,
        crossover_rate=0.8,
        verbose=True
    )
    ga_time = time.time() - start

    ga_metrics = evaluate_solution(ga_solution, reward_function.data)

    print(f"\n✓ GA completed in {ga_time:.2f}s")
    print(f"  Heat: {ga_metrics['heat_sum']:.2f}")
    print(f"  Socio-Vuln: {ga_metrics['socio_sum']:.2f}")
    print(f"  Population: {ga_metrics['population_served']:,.0f}")
    print(f"  Olympic: {ga_metrics['olympic_coverage']:.1f}%")
    print(f"  Equity (Gini): {ga_metrics['equity_gini']:.3f}")

    # 3. Comparison
    print(f"\n{'='*80}")
    print("COMPARISON: GA vs Greedy")
    print(f"{'='*80}\n")

    def percent_change(ga_val, greedy_val, lower_is_better=False):
        if greedy_val == 0:
            return "N/A"
        diff = ((ga_val - greedy_val) / greedy_val) * 100
        if lower_is_better:
            diff = -diff  # Flip sign for metrics where lower is better
        symbol = "✅" if diff > 0 else "❌" if diff < -1 else "➖"
        return f"{symbol} {diff:+.1f}%"

    print(f"Heat:       GA={ga_metrics['heat_sum']:.1f}  vs  Greedy={greedy_metrics['heat_sum']:.1f}")
    print(f"            {percent_change(ga_metrics['heat_sum'], greedy_metrics['heat_sum'])}")

    print(f"\nSocio-Vuln: GA={ga_metrics['socio_sum']:.1f}  vs  Greedy={greedy_metrics['socio_sum']:.1f}")
    print(f"            {percent_change(ga_metrics['socio_sum'], greedy_metrics['socio_sum'])}")

    print(f"\nPopulation: GA={ga_metrics['population_served']:,.0f}  vs  Greedy={greedy_metrics['population_served']:,.0f}")
    print(f"            {percent_change(ga_metrics['population_served'], greedy_metrics['population_served'])}")

    print(f"\nOlympic:    GA={ga_metrics['olympic_coverage']:.1f}%  vs  Greedy={greedy_metrics['olympic_coverage']:.1f}%")
    print(f"            {percent_change(ga_metrics['olympic_coverage'], greedy_metrics['olympic_coverage'])}")

    print(f"\nEquity:     GA={ga_metrics['equity_gini']:.3f}  vs  Greedy={greedy_metrics['equity_gini']:.3f}")
    print(f"            {percent_change(ga_metrics['equity_gini'], greedy_metrics['equity_gini'], lower_is_better=True)}")

    print(f"\nRuntime:    GA={ga_time:.1f}s  vs  Greedy={greedy_time:.1f}s")
    print(f"            (GA is {ga_time/greedy_time:.1f}x slower)")

    # 4. Save results
    results = {
        'approach': approach_name,
        'k': k,
        'greedy': {
            'solution': greedy_solution,
            'metrics': greedy_metrics,
            'time': greedy_time
        },
        'ga': {
            'solution': ga_solution,
            'metrics': ga_metrics,
            'time': ga_time
        }
    }

    filename = f"results/ga_comparison_{approach_name.replace(' ', '_')}_k{k}.json"
    with open(filename, 'w') as f:
        # Convert numpy types to native Python for JSON serialization
        results_serializable = {
            'approach': results['approach'],
            'k': int(results['k']),
            'greedy': {
                'solution': [int(x) for x in results['greedy']['solution']],
                'metrics': {k: float(v) for k, v in results['greedy']['metrics'].items()},
                'time': float(results['greedy']['time'])
            },
            'ga': {
                'solution': [int(x) for x in results['ga']['solution']],
                'metrics': {k: float(v) for k, v in results['ga']['metrics'].items()},
                'time': float(results['ga']['time'])
            }
        }
        json.dump(results_serializable, f, indent=2)

    print(f"\n✓ Results saved to: {filename}")

    return results


def main():
    """Main execution."""

    print("="*80)
    print("GENETIC ALGORITHM vs GREEDY COMPARISON")
    print("="*80)

    # Load data
    data = load_data()
    print(f"\nLoaded {len(data)} grid locations")

    k = 10
    region = 'All'

    # Test Approach 1
    print("\n\n" + "="*80)
    print("TEST 1: Approach 1 (Enhanced Weighted Sum)")
    print("="*80)

    reward_fn_1 = EnhancedWeightedSumReward(data, region=region)
    results_1 = run_comparison("Approach1", reward_fn_1, k=k)

    # Test Approach 2
    print("\n\n" + "="*80)
    print("TEST 2: Approach 2 (Multiplicative/Hierarchical)")
    print("="*80)

    reward_fn_2 = MultiplicativeHierarchicalReward(data, region=region)
    results_2 = run_comparison("Approach2", reward_fn_2, k=k)

    # Final summary
    print("\n\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    print("\nDid GA beat Greedy?")

    # Approach 1
    pop_diff_1 = results_1['ga']['metrics']['population_served'] - results_1['greedy']['metrics']['population_served']
    heat_diff_1 = results_1['ga']['metrics']['heat_sum'] - results_1['greedy']['metrics']['heat_sum']

    print(f"\nApproach 1:")
    print(f"  Population: {'+' if pop_diff_1 > 0 else ''}{pop_diff_1:,.0f} people")
    print(f"  Heat: {'+' if heat_diff_1 > 0 else ''}{heat_diff_1:.2f}")
    print(f"  Verdict: {'✅ GA WINS!' if pop_diff_1 > 0 else '❌ Greedy still better'}")

    # Approach 2
    pop_diff_2 = results_2['ga']['metrics']['population_served'] - results_2['greedy']['metrics']['population_served']
    heat_diff_2 = results_2['ga']['metrics']['heat_sum'] - results_2['greedy']['metrics']['heat_sum']

    print(f"\nApproach 2:")
    print(f"  Population: {'+' if pop_diff_2 > 0 else ''}{pop_diff_2:,.0f} people")
    print(f"  Heat: {'+' if heat_diff_2 > 0 else ''}{heat_diff_2:.2f}")
    print(f"  Verdict: {'✅ GA WINS!' if pop_diff_2 > 0 else '❌ Greedy still better'}")

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    if pop_diff_1 > 0 or pop_diff_2 > 0:
        print("\n✅ Genetic Algorithm shows improvement over Greedy!")
        print("   Recommendation: Use GA for final k=100/200 runs")
    else:
        print("\n❌ Greedy still outperforms GA in this test.")
        print("   Possible reasons:")
        print("   - GA needs more generations (try 500-1000)")
        print("   - Need better hyperparameter tuning")
        print("   - Greedy is truly near-optimal for this problem")
        print("\n   Recommendation: Try NSGA-II for multi-objective optimization instead")


if __name__ == '__main__':
    main()
