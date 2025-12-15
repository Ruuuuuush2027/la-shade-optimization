#!/usr/bin/env python3
"""
Collect results from greedy_experiments and nsga2_experiments and update the LaTeX table.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

def load_json_result(filepath: str) -> Dict:
    """Load a JSON result file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_metrics(data: Dict) -> Dict:
    """Extract relevant metrics from JSON data."""
    metrics = data['metrics']
    metadata = data['metadata']

    return {
        'approach': metadata['approach'],
        'method': metadata['method'],
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
            # Format with comma separator for large numbers
            return f"{int(num):,}"
        else:
            return f"{num:.2f}"
    else:
        return f"{int(num):,}"

def collect_all_results(base_path: str) -> List[Dict]:
    """Collect all results from greedy and nsga2 experiments."""
    results = []

    # Collect greedy experiments
    greedy_path = Path(base_path) / "greedy_experiments"
    for approach_dir in greedy_path.glob("approach*"):
        for json_file in approach_dir.rglob("*.json"):
            data = load_json_result(json_file)
            result = extract_metrics(data)
            results.append(result)
            print(f"Loaded: {json_file.name} - Approach {result['approach']}, k={result['k']}, method={result['method']}")

    # Collect NSGA2 experiments
    nsga2_path = Path(base_path) / "nsga2_experiments"
    for approach_dir in nsga2_path.glob("approach*"):
        for json_file in approach_dir.rglob("*.json"):
            data = load_json_result(json_file)
            result = extract_metrics(data)
            results.append(result)
            print(f"Loaded: {json_file.name} - Approach {result['approach']}, k={result['k']}, method={result['method']}")

    return results

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
        if abs(value - best_val) < 0.01:  # Account for floating point precision
            return f"\\textbf{{\\underline{{{formatted}}}}}"

    return formatted

def generate_latex_table(results: List[Dict]) -> str:
    """Generate the complete LaTeX table."""

    # Group results by k value
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
    lines.append("\\label{tab:results}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{@{}llrrrrr@{}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Reward Function} & \\textbf{Method} & \\textbf{Heat Sum} & \\textbf{Socio-Vuln} & \\textbf{Gini} $\\downarrow$ & \\textbf{Population} & \\textbf{Olympic \\%} \\\\")
    lines.append("\\midrule")

    # Process each k value
    for k in k_values:
        k_results = [r for r in results if r['k'] == k]
        best_values = find_best_in_group(results, k)

        # Determine section header
        if k <= 10:
            # For k=10, we might have both approach 2 and approach 3
            has_approach2 = any(r['approach'] == 'Approach2' for r in k_results)
            has_approach3 = any(r['approach'] == 'Approach3' for r in k_results)

            if has_approach2:
                lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{Hierarchical/Multiplicative (k={k})}}}} \\\\")
                approach2_results = [r for r in k_results if r['approach'] == 'Approach2']
                for result in sorted(approach2_results, key=lambda x: (x['method'])):
                    method_name = result['method'].title() if result['method'] != 'NSGA-II' else 'NSGA-II'
                    line = f"\\quad Approach 2 & {method_name} & "
                    line += f"{format_value_with_best(result['heat_sum'], 'heat_sum', best_values)} & "
                    line += f"{format_value_with_best(result['socio_sum'], 'socio_sum', best_values)} & "
                    line += f"{format_value_with_best(result['gini'], 'gini', best_values)} & "
                    line += f"{format_value_with_best(result['population'], 'population', best_values)} & "
                    line += f"{format_value_with_best(result['olympic_pct'], 'olympic_pct', best_values, is_percentage=True)} \\\\"
                    lines.append(line)

                if has_approach3:
                    lines.append("\\midrule")

            if has_approach3:
                lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{Multi-Objective NSGA-II Approach (k={k})}}}} \\\\")
                approach3_results = [r for r in k_results if r['approach'] == 'Approach3']
                for result in sorted(approach3_results, key=lambda x: (x['method'])):
                    method_name = result['method'].title() if result['method'] != 'NSGA-II' else 'NSGA-II'
                    line = f"\\quad Approach 3 & {method_name} & "
                    line += f"{format_value_with_best(result['heat_sum'], 'heat_sum', best_values)} & "
                    line += f"{format_value_with_best(result['socio_sum'], 'socio_sum', best_values)} & "
                    line += f"{format_value_with_best(result['gini'], 'gini', best_values)} & "
                    line += f"{format_value_with_best(result['population'], 'population', best_values)} & "
                    line += f"{format_value_with_best(result['olympic_pct'], 'olympic_pct', best_values, is_percentage=True)} \\\\"
                    lines.append(line)
        else:
            # For k>10, group all approaches together
            lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{k={k} Results}}}} \\\\")

            # Sort by approach then method
            for result in sorted(k_results, key=lambda x: (x['approach'], x['method'])):
                approach_num = result['approach'].replace('Approach', '')
                method_name = result['method'].title() if result['method'] != 'NSGA-II' else 'NSGA-II'

                line = f"\\quad Approach {approach_num} & {method_name} & "
                line += f"{format_value_with_best(result['heat_sum'], 'heat_sum', best_values)} & "
                line += f"{format_value_with_best(result['socio_sum'], 'socio_sum', best_values)} & "
                line += f"{format_value_with_best(result['gini'], 'gini', best_values)} & "
                line += f"{format_value_with_best(result['population'], 'population', best_values)} & "
                line += f"{format_value_with_best(result['olympic_pct'], 'olympic_pct', best_values, is_percentage=True)} \\\\"
                lines.append(line)

        # Add midrule between k values (except after the last one)
        if k != k_values[-1]:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\vspace{0.1cm}")
    lines.append("")
    lines.append("\\smallskip")
    lines.append("\\small")
    lines.append("\\textit{Note:} Heat Sum and Socio-Vuln (socio-vulnerability) are cumulative scores (higher is better). Gini coefficient measures inequality in benefit distribution (lower is better, marked with $\\downarrow$). Population indicates total population served within 500m. Olympic \\% represents percentage of Olympic venues covered. Results grouped by k value (number of shade placements). Approach 2 uses hierarchical/multiplicative reward. Approach 3 uses multi-objective optimization with NSGA-II to explore the Pareto frontier.")
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

    print("Collecting all results...")
    results = collect_all_results(base_path)

    print(f"\nTotal results collected: {len(results)}")
    print(f"K values: {sorted(set(r['k'] for r in results))}")
    print(f"Approaches: {sorted(set(r['approach'] for r in results))}")
    print(f"Methods: {sorted(set(r['method'] for r in results))}")

    print("\nGenerating LaTeX table...")
    latex_content = generate_latex_table(results)

    print(f"Writing to {output_path}...")
    with open(output_path, 'w') as f:
        f.write(latex_content)

    print("Done! Results table updated successfully.")

if __name__ == "__main__":
    main()
