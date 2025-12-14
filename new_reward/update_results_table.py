#!/usr/bin/env python3
"""
Script to update the results table with new greedy and NSGA-II experiments
"""
import json
import os
from pathlib import Path

def load_json_results(file_path):
    """Load results from a JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def format_number(value, is_int=False, is_percent=False):
    """Format numbers for LaTeX table"""
    if is_percent:
        return f"{value:.2f}"
    elif is_int:
        return f"{int(value):,}"
    else:
        return f"{value:.2f}"

def main():
    # Base directories
    greedy_dir = Path("/home/fhliang/projects/libero_shade/new_reward/results/greedy_experiments")
    nsga2_dir = Path("/home/fhliang/projects/libero_shade/new_reward/results/nsga2_experiments")

    # Collect all results
    all_results = []

    # Load greedy experiments (approach2 and approach3)
    for approach in ['approach2', 'approach3']:
        for k in [10, 20, 50, 100, 200]:
            greedy_file = greedy_dir / approach / "All" / f"{approach}_k{k}.json"
            if greedy_file.exists():
                data = load_json_results(greedy_file)
                all_results.append({
                    'approach': approach.replace('approach', 'Approach '),
                    'method': 'Greedy',
                    'k': k,
                    'metrics': data['metrics']
                })

    # Load NSGA-II experiments (approach2 and approach3)
    for approach in ['approach2', 'approach3']:
        for k in [10, 20, 50, 100, 200]:
            nsga2_file = nsga2_dir / approach / "All" / f"{approach}_nsga2_k{k}.json"
            if nsga2_file.exists():
                data = load_json_results(nsga2_file)
                all_results.append({
                    'approach': approach.replace('approach', 'Approach '),
                    'method': 'NSGA-II',
                    'k': k,
                    'metrics': data['metrics']
                })

    # Print results for k=10
    print("\n=== Results for k=10 ===")
    print(f"{'Approach':<15} {'Method':<15} {'Heat Sum':>10} {'Socio-Vuln':>12} {'Gini':>8} {'Population':>15} {'Olympic %':>10}")
    print("-" * 100)

    k10_results = [r for r in all_results if r['k'] == 10]
    for result in sorted(k10_results, key=lambda x: (x['approach'], x['method'])):
        m = result['metrics']
        print(f"{result['approach']:<15} {result['method']:<15} "
              f"{m['heat_sum']:>10.2f} {m['socio_sum']:>12.2f} "
              f"{m['equity_gini']:>8.4f} {int(m['population_served']):>15,} "
              f"{m['olympic_coverage']:>10.2f}")

    # Print results for k=20
    print("\n=== Results for k=20 ===")
    print(f"{'Approach':<15} {'Method':<15} {'Heat Sum':>10} {'Socio-Vuln':>12} {'Gini':>8} {'Population':>15} {'Olympic %':>10}")
    print("-" * 100)

    k20_results = [r for r in all_results if r['k'] == 20]
    for result in sorted(k20_results, key=lambda x: (x['approach'], x['method'])):
        m = result['metrics']
        print(f"{result['approach']:<15} {result['method']:<15} "
              f"{m['heat_sum']:>10.2f} {m['socio_sum']:>12.2f} "
              f"{m['equity_gini']:>8.4f} {int(m['population_served']):>15,} "
              f"{m['olympic_coverage']:>10.2f}")

    # Print results for higher k values
    for k_val in [50, 100, 200]:
        kx_results = [r for r in all_results if r['k'] == k_val]
        if kx_results:
            print(f"\n=== Results for k={k_val} ===")
            print(f"{'Approach':<15} {'Method':<15} {'Heat Sum':>10} {'Socio-Vuln':>12} {'Gini':>8} {'Population':>15} {'Olympic %':>10}")
            print("-" * 100)
            for result in sorted(kx_results, key=lambda x: (x['approach'], x['method'])):
                m = result['metrics']
                print(f"{result['approach']:<15} {result['method']:<15} "
                      f"{m['heat_sum']:>10.2f} {m['socio_sum']:>12.2f} "
                      f"{m['equity_gini']:>8.4f} {int(m['population_served']):>15,} "
                      f"{m['olympic_coverage']:>10.2f}")

    # Find best values for k=10 (for underlining in LaTeX)
    best_k10 = {
        'heat_sum': max(k10_results, key=lambda x: x['metrics']['heat_sum']),
        'socio_sum': max(k10_results, key=lambda x: x['metrics']['socio_sum']),
        'equity_gini': min(k10_results, key=lambda x: x['metrics']['equity_gini']),
        'population_served': max(k10_results, key=lambda x: x['metrics']['population_served']),
        'olympic_coverage': max(k10_results, key=lambda x: x['metrics']['olympic_coverage'])
    }

    print("\n=== Best values for k=10 ===")
    for metric, result in best_k10.items():
        m = result['metrics']
        print(f"{metric}: {result['approach']} {result['method']} = {m[metric]:.2f}")

    # Save results to JSON for later use
    output_file = Path("/home/fhliang/projects/libero_shade/new_reward/processed_results.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")

    return all_results

if __name__ == "__main__":
    main()
