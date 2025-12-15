#!/usr/bin/env python3
"""
Merge old results with new greedy_experiments and nsga2_experiments.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Old results manually extracted from the previous table
OLD_RESULTS = [
    # Enhanced Weighted Sum (k=10) - Approach 1
    {'approach': 'Approach1', 'method': 'Greedy', 'k': 10, 'heat_sum': 498.32, 'socio_sum': 22.23, 'gini': 0.86, 'population': 1175747, 'olympic_pct': 11.11},
    {'approach': 'Approach1', 'method': 'KMeans', 'k': 10, 'heat_sum': 481.92, 'socio_sum': 15.07, 'gini': 0.86, 'population': 1085786, 'olympic_pct': 22.22},
    {'approach': 'Approach1', 'method': 'Random', 'k': 10, 'heat_sum': 485.36, 'socio_sum': 13.50, 'gini': 0.87, 'population': 1067766, 'olympic_pct': 55.56},
    {'approach': 'Approach1', 'method': 'GA', 'k': 10, 'heat_sum': 495.58, 'socio_sum': 23.71, 'gini': 0.88, 'population': 1051678, 'olympic_pct': 11.11},
    {'approach': 'Approach1', 'method': 'RL', 'k': 10, 'heat_sum': 486.49, 'socio_sum': 19.27, 'gini': 0.86, 'population': 1021351, 'olympic_pct': 22.22},
    {'approach': 'Approach1', 'method': 'ExpertHeuristic', 'k': 10, 'heat_sum': 497.01, 'socio_sum': 27.18, 'gini': 0.93, 'population': 751964, 'olympic_pct': 0.00},
    {'approach': 'Approach1', 'method': 'GreedyByTemp', 'k': 10, 'heat_sum': 517.17, 'socio_sum': 23.70, 'gini': 0.96, 'population': 357455, 'olympic_pct': 0.00},

    # Approach 2 other methods (k=10)
    {'approach': 'Approach2', 'method': 'GA', 'k': 10, 'heat_sum': 497.08, 'socio_sum': 26.58, 'gini': 0.85, 'population': 1160601, 'olympic_pct': 0.00},
    {'approach': 'Approach2', 'method': 'KMeans', 'k': 10, 'heat_sum': 481.92, 'socio_sum': 15.07, 'gini': 0.86, 'population': 1085786, 'olympic_pct': 22.22},
    {'approach': 'Approach2', 'method': 'Random', 'k': 10, 'heat_sum': 485.36, 'socio_sum': 13.50, 'gini': 0.87, 'population': 1067766, 'olympic_pct': 55.56},
    {'approach': 'Approach2', 'method': 'RL', 'k': 10, 'heat_sum': 496.06, 'socio_sum': 36.18, 'gini': 0.87, 'population': 1001749, 'olympic_pct': 0.00},
    {'approach': 'Approach2', 'method': 'ExpertHeuristic', 'k': 10, 'heat_sum': 497.01, 'socio_sum': 27.18, 'gini': 0.93, 'population': 751964, 'olympic_pct': 0.00},
    {'approach': 'Approach2', 'method': 'GreedyByTemp', 'k': 10, 'heat_sum': 517.17, 'socio_sum': 23.70, 'gini': 0.96, 'population': 357455, 'olympic_pct': 0.00},

    # k=20 Results - Approach 1
    {'approach': 'Approach1', 'method': 'Greedy', 'k': 20, 'heat_sum': 985.60, 'socio_sum': 40.22, 'gini': 0.73, 'population': 2302162, 'olympic_pct': 22.22},
    {'approach': 'Approach1', 'method': 'RL', 'k': 20, 'heat_sum': 976.12, 'socio_sum': 38.32, 'gini': 0.77, 'population': 2136590, 'olympic_pct': 0.00},
]

def load_json_result(filepath: str) -> Dict:
    """Load a JSON result file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_metrics(data: Dict) -> Dict:
    """Extract relevant metrics from JSON data."""
    metrics = data['metrics']
    metadata = data['metadata']

    # Capitalize method name for consistency
    method = metadata['method']
    if method == 'greedy':
        method = 'Greedy'

    return {
        'approach': metadata['approach'],
        'method': method,
        'k': metadata['k'],
        'heat_sum': metrics['heat_sum'],
        'socio_sum': metrics['socio_sum'],
        'gini': metrics['equity_gini'],
        'population': metrics['population_served'],
        'olympic_pct': metrics['olympic_coverage']
    }

def format_number(num, is_percentage=False):
    """Format numbers for LaTeX table."""
    if is_percentage:
        return f"{num:.2f}"
    elif isinstance(num, float):
        if num > 10000:
            return f"{int(num):,}"
        else:
            return f"{num:.2f}"
    else:
        return f"{int(num):,}"

def collect_new_results(base_path: str) -> List[Dict]:
    """Collect all results from greedy and nsga2 experiments."""
    results = []

    # Collect greedy experiments
    greedy_path = Path(base_path) / "greedy_experiments"
    for approach_dir in greedy_path.glob("approach*"):
        for json_file in approach_dir.rglob("*.json"):
            data = load_json_result(json_file)
            result = extract_metrics(data)
            results.append(result)
            print(f"Loaded NEW: {json_file.name} - Approach {result['approach']}, k={result['k']}, method={result['method']}")

    # Collect NSGA2 experiments
    nsga2_path = Path(base_path) / "nsga2_experiments"
    for approach_dir in nsga2_path.glob("approach*"):
        for json_file in approach_dir.rglob("*.json"):
            data = load_json_result(json_file)
            result = extract_metrics(data)
            results.append(result)
            print(f"Loaded NEW: {json_file.name} - Approach {result['approach']}, k={result['k']}, method={result['method']}")

    return results

def merge_results(old_results: List[Dict], new_results: List[Dict]) -> List[Dict]:
    """Merge old and new results, preferring new results for Greedy/NSGA-II on Approach 2/3."""
    all_results = []

    # Add all old results
    for old in old_results:
        all_results.append(old)
        print(f"Added OLD: Approach {old['approach']}, k={old['k']}, method={old['method']}")

    # Add new results
    for new in new_results:
        all_results.append(new)
        print(f"Added NEW: Approach {new['approach']}, k={new['k']}, method={new['method']}")

    return all_results

def find_best_in_group(results: List[Dict], k: int) -> Dict[str, Tuple[float, str]]:
    """Find best values for each metric within a k group."""
    k_results = [r for r in results if r['k'] == k]
    if not k_results:
        return {}

    best = {
        'heat_sum': (max(k_results, key=lambda x: x['heat_sum'])['heat_sum'], 'max'),
        'socio_sum': (max(k_results, key=lambda x: x['socio_sum'])['socio_sum'], 'max'),
        'gini': (min(k_results, key=lambda x: x['gini'])['gini'], 'min'),
        'population': (max(k_results, key=lambda x: x['population'])['population'], 'max'),
        'olympic_pct': (max(k_results, key=lambda x: x['olympic_pct'])['olympic_pct'], 'max')
    }
    return best

def format_value_with_best(value, metric_name, best_values, is_percentage=False):
    """Format value and add bold+underline if it's the best."""
    formatted = format_number(value, is_percentage)

    if metric_name in best_values:
        best_val, _ = best_values[metric_name]
        if abs(value - best_val) < 0.01:
            return f"\\textbf{{\\underline{{{formatted}}}}}"

    return formatted

def method_sort_key(method_name: str) -> int:
    """Sort key for methods to maintain consistent ordering."""
    order = {
        'Greedy': 0,
        'KMeans': 1,
        'Random': 2,
        'GA': 3,
        'RL': 4,
        'ExpertHeuristic': 5,
        'GreedyByTemp': 6,
        'NSGA-II': 7
    }
    return order.get(method_name, 99)

def generate_latex_table(results: List[Dict]) -> str:
    """Generate the complete LaTeX table."""
    k_values = sorted(set(r['k'] for r in results))

    lines = []
    lines.append("\\documentclass{article}")
    lines.append("\\usepackage{booktabs}")
    lines.append("\\usepackage{graphicx}")
    lines.append("\\usepackage[normalem]{ulem}")
    lines.append("")
    lines.append("\\begin{document}")
    lines.append("")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Performance comparison of reward function approaches and optimization methods for cooling shade placement in Los Angeles. Best values in each section are underlined.}")
    lines.append("\\vspace{0.1cm}")
    lines.append("\\label{tab:results}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{@{}llrrrrr@{}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Reward Function} & \\textbf{Method} & \\textbf{Heat Sum} & \\textbf{Socio-Vuln} & \\textbf{Gini} $\\downarrow$ & \\textbf{Population} & \\textbf{Olympic \\%} \\\\")
    lines.append("\\midrule")

    for k_idx, k in enumerate(k_values):
        k_results = [r for r in results if r['k'] == k]
        best_values = find_best_in_group(results, k)

        # Group by approach for k=10, otherwise just show k=X Results
        if k == 10:
            # Enhanced Weighted Sum - Approach 1
            approach1_results = [r for r in k_results if r['approach'] == 'Approach1']
            if approach1_results:
                lines.append("\\multicolumn{7}{l}{\\textit{Enhanced Weighted Sum (k=10)}} \\\\")
                for result in sorted(approach1_results, key=lambda x: method_sort_key(x['method'])):
                    method_name = result['method']
                    line = f"\\quad Approach 1 & {method_name} & "
                    line += f"{format_value_with_best(result['heat_sum'], 'heat_sum', best_values)} & "
                    line += f"{format_value_with_best(result['socio_sum'], 'socio_sum', best_values)} & "
                    line += f"{format_value_with_best(result['gini'], 'gini', best_values)} & "
                    line += f"{format_value_with_best(result['population'], 'population', best_values)} & "
                    line += f"{format_value_with_best(result['olympic_pct'], 'olympic_pct', best_values, is_percentage=True)} \\\\"
                    lines.append(line)
                lines.append("\\midrule")

            # Hierarchical/Multiplicative - Approach 2
            approach2_results = [r for r in k_results if r['approach'] == 'Approach2']
            if approach2_results:
                lines.append("\\multicolumn{7}{l}{\\textit{Hierarchical/Multiplicative (k=10)}} \\\\")
                for result in sorted(approach2_results, key=lambda x: method_sort_key(x['method'])):
                    method_name = result['method']
                    line = f"\\quad Approach 2 & {method_name} & "
                    line += f"{format_value_with_best(result['heat_sum'], 'heat_sum', best_values)} & "
                    line += f"{format_value_with_best(result['socio_sum'], 'socio_sum', best_values)} & "
                    line += f"{format_value_with_best(result['gini'], 'gini', best_values)} & "
                    line += f"{format_value_with_best(result['population'], 'population', best_values)} & "
                    line += f"{format_value_with_best(result['olympic_pct'], 'olympic_pct', best_values, is_percentage=True)} \\\\"
                    lines.append(line)
                lines.append("\\midrule")

            # Multi-Objective - Approach 3
            approach3_results = [r for r in k_results if r['approach'] == 'Approach3']
            if approach3_results:
                lines.append("\\multicolumn{7}{l}{\\textit{Multi-Objective NSGA-II Approach (k=10)}} \\\\")
                for result in sorted(approach3_results, key=lambda x: method_sort_key(x['method'])):
                    method_name = result['method']
                    line = f"\\quad Approach 3 & {method_name} & "
                    line += f"{format_value_with_best(result['heat_sum'], 'heat_sum', best_values)} & "
                    line += f"{format_value_with_best(result['socio_sum'], 'socio_sum', best_values)} & "
                    line += f"{format_value_with_best(result['gini'], 'gini', best_values)} & "
                    line += f"{format_value_with_best(result['population'], 'population', best_values)} & "
                    line += f"{format_value_with_best(result['olympic_pct'], 'olympic_pct', best_values, is_percentage=True)} \\\\"
                    lines.append(line)
        else:
            # For k > 10, just show "k=X Results"
            lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{k={k} Results}}}} \\\\")
            for result in sorted(k_results, key=lambda x: (x['approach'], method_sort_key(x['method']))):
                approach_num = result['approach'].replace('Approach', '')
                method_name = result['method']
                line = f"\\quad Approach {approach_num} & {method_name} & "
                line += f"{format_value_with_best(result['heat_sum'], 'heat_sum', best_values)} & "
                line += f"{format_value_with_best(result['socio_sum'], 'socio_sum', best_values)} & "
                line += f"{format_value_with_best(result['gini'], 'gini', best_values)} & "
                line += f"{format_value_with_best(result['population'], 'population', best_values)} & "
                line += f"{format_value_with_best(result['olympic_pct'], 'olympic_pct', best_values, is_percentage=True)} \\\\"
                lines.append(line)

        # Add midrule between k values (except after the last one)
        if k_idx < len(k_values) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\vspace{0.1cm}")
    lines.append("")
    lines.append("\\smallskip")
    lines.append("\\small")
    lines.append("\\textit{Note:} Heat Sum and Socio-Vuln (socio-vulnerability) are cumulative scores (higher is better). Gini coefficient measures inequality in benefit distribution (lower is better, marked with $\\downarrow$). Population indicates total population served within 500m. Olympic \\% represents percentage of Olympic venues covered. Results grouped by k value (number of shade placements). Approach 1 uses enhanced weighted sum. Approach 2 uses hierarchical/multiplicative reward. Approach 3 uses multi-objective optimization with NSGA-II to explore the Pareto frontier.")
    lines.append("\\end{table}")
    lines.append("")
    lines.append("\\vspace{0.3cm}")
    lines.append("")
    lines.append("\\noindent\\textbf{Key Observations:}")
    lines.append("")
    lines.append("\\noindent\\textit{Greedy vs. NSGA-II for Approach 3:} The greedy algorithm can perform poorly with Approach 3's multi-objective reward function because it makes locally optimal decisions based on aggregated metrics at each step, which may fail to explore the Pareto frontier effectively. NSGA-II, in contrast, maintains a diverse population of solutions across multiple objectives, enabling it to discover superior trade-offs through evolutionary search.")
    lines.append("")
    lines.append("\\noindent\\textit{NSGA-II Scalability:} NSGA-II may face computational and constraint-satisfaction challenges at larger k values. As k increases with strict planting constraints, the feasible solution space becomes increasingly sparse, requiring more evaluations to find satisfactory solutions.")
    lines.append("")
    lines.append("\\end{document}")

    return "\n".join(lines)

def main():
    base_path = "/home/fhliang/projects/libero_shade/new_reward/results"
    output_path = "/home/fhliang/projects/libero_shade/new_reward/results_table.tex"

    print("=" * 60)
    print("Collecting new experimental results...")
    print("=" * 60)
    new_results = collect_new_results(base_path)

    print("\n" + "=" * 60)
    print("Merging with old results...")
    print("=" * 60)
    all_results = merge_results(OLD_RESULTS, new_results)

    print(f"\n" + "=" * 60)
    print(f"Total results: {len(all_results)}")
    print(f"K values: {sorted(set(r['k'] for r in all_results))}")
    print(f"Approaches: {sorted(set(r['approach'] for r in all_results))}")
    print(f"Methods: {sorted(set(r['method'] for r in all_results))}")
    print("=" * 60)

    print("\nGenerating LaTeX table...")
    latex_content = generate_latex_table(all_results)

    print(f"Writing to {output_path}...")
    with open(output_path, 'w') as f:
        f.write(latex_content)

    print("\n✓ Done! Results table updated successfully with merged results.")

if __name__ == "__main__":
    main()
