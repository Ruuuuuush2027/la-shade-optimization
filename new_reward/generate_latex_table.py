#!/usr/bin/env python3
"""
Generate updated LaTeX table with all results
"""
import json
import os
from pathlib import Path
from collections import defaultdict

def load_json_results(file_path):
    """Load results from a JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def format_value(value, best_value=None, is_lower_better=False):
    """Format value for LaTeX, with optional underline for best values"""
    formatted = f"{value:.2f}"
    if best_value is not None:
        if is_lower_better:
            if abs(value - best_value) < 0.001:  # Close enough to best
                return f"\\textbf{{\\underline{{{formatted}}}}}"
        else:
            if abs(value - best_value) < 0.001:
                return f"\\textbf{{\\underline{{{formatted}}}}}"
    return formatted

def format_population(value, best_value=None):
    """Format population with thousands separator"""
    formatted = f"{int(value):,}"
    if best_value is not None and abs(value - best_value) < 1:
        return f"\\textbf{{\\underline{{{formatted}}}}}"
    return formatted

def main():
    # Load old results (Approach 1 and Approach 2)
    old_results_dir = Path("/home/fhliang/projects/libero_shade/new_reward/results/region_specific/All")

    # Load new greedy and NSGA-II experiments
    greedy_dir = Path("/home/fhliang/projects/libero_shade/new_reward/results/greedy_experiments")
    nsga2_dir = Path("/home/fhliang/projects/libero_shade/new_reward/results/nsga2_experiments")

    # Organize results by approach and k value
    results_k10 = []
    results_k20 = []
    results_k50 = []
    results_k100 = []
    results_k200 = []

    # Load old Approach 1 k=10 results
    for method in ['Greedy', 'KMeans', 'Random', 'GA', 'RL', 'ExpertHeuristic', 'GreedyByTemp']:
        file_path = old_results_dir / f"Approach1_{method}_k10.json"
        if file_path.exists():
            data = load_json_results(file_path)
            results_k10.append({
                'approach': 'Approach 1',
                'method': method,
                'metrics': data['metrics']
            })

    # Load old Approach 2 k=10 results
    for method in ['Greedy', 'GA', 'KMeans', 'Random', 'RL', 'ExpertHeuristic', 'GreedyByTemp']:
        file_path = old_results_dir / f"Approach2_{method}_k10.json"
        if file_path.exists():
            data = load_json_results(file_path)
            # Skip if we're going to load the new greedy results for approach 2
            if method != 'Greedy':  # We'll use new greedy results
                results_k10.append({
                    'approach': 'Approach 2',
                    'method': method,
                    'metrics': data['metrics']
                })

    # Load new Approach 2 and Approach 3 results (k=10)
    for approach in ['approach2', 'approach3']:
        # Greedy
        greedy_file = greedy_dir / approach / "All" / f"{approach}_k10.json"
        if greedy_file.exists():
            data = load_json_results(greedy_file)
            results_k10.append({
                'approach': approach.replace('approach', 'Approach '),
                'method': 'Greedy',
                'metrics': data['metrics']
            })

        # NSGA-II
        nsga2_file = nsga2_dir / approach / "All" / f"{approach}_nsga2_k10.json"
        if nsga2_file.exists():
            data = load_json_results(nsga2_file)
            results_k10.append({
                'approach': approach.replace('approach', 'Approach '),
                'method': 'NSGA-II',
                'metrics': data['metrics']
            })

    # Load k=20 results
    # Old Approach 1 k=20
    for method in ['Greedy', 'RL']:
        file_path = old_results_dir / f"Approach1_{method}_k20.json"
        if file_path.exists():
            data = load_json_results(file_path)
            results_k20.append({
                'approach': 'Approach 1',
                'method': method,
                'metrics': data['metrics']
            })

    # New Approach 2 and Approach 3 k=20
    for approach in ['approach2', 'approach3']:
        # Greedy
        greedy_file = greedy_dir / approach / "All" / f"{approach}_k20.json"
        if greedy_file.exists():
            data = load_json_results(greedy_file)
            results_k20.append({
                'approach': approach.replace('approach', 'Approach '),
                'method': 'Greedy',
                'metrics': data['metrics']
            })

        # NSGA-II
        nsga2_file = nsga2_dir / approach / "All" / f"{approach}_nsga2_k20.json"
        if nsga2_file.exists():
            data = load_json_results(nsga2_file)
            results_k20.append({
                'approach': approach.replace('approach', 'Approach '),
                'method': 'NSGA-II',
                'metrics': data['metrics']
            })

    # Load k=50 results
    for approach in ['approach2', 'approach3']:
        # Greedy
        greedy_file = greedy_dir / approach / "All" / f"{approach}_k50.json"
        if greedy_file.exists():
            data = load_json_results(greedy_file)
            results_k50.append({
                'approach': approach.replace('approach', 'Approach '),
                'method': 'Greedy',
                'metrics': data['metrics']
            })

        # NSGA-II
        nsga2_file = nsga2_dir / approach / "All" / f"{approach}_nsga2_k50.json"
        if nsga2_file.exists():
            data = load_json_results(nsga2_file)
            results_k50.append({
                'approach': approach.replace('approach', 'Approach '),
                'method': 'NSGA-II',
                'metrics': data['metrics']
            })

    # Load k=100 results
    for approach in ['approach2', 'approach3']:
        # Greedy
        greedy_file = greedy_dir / approach / "All" / f"{approach}_k100.json"
        if greedy_file.exists():
            data = load_json_results(greedy_file)
            results_k100.append({
                'approach': approach.replace('approach', 'Approach '),
                'method': 'Greedy',
                'metrics': data['metrics']
            })

        # NSGA-II
        nsga2_file = nsga2_dir / approach / "All" / f"{approach}_nsga2_k100.json"
        if nsga2_file.exists():
            data = load_json_results(nsga2_file)
            results_k100.append({
                'approach': approach.replace('approach', 'Approach '),
                'method': 'NSGA-II',
                'metrics': data['metrics']
            })

    # Load k=200 results
    for approach in ['approach2', 'approach3']:
        # Greedy
        greedy_file = greedy_dir / approach / "All" / f"{approach}_k200.json"
        if greedy_file.exists():
            data = load_json_results(greedy_file)
            results_k200.append({
                'approach': approach.replace('approach', 'Approach '),
                'method': 'Greedy',
                'metrics': data['metrics']
            })

        # NSGA-II
        nsga2_file = nsga2_dir / approach / "All" / f"{approach}_nsga2_k200.json"
        if nsga2_file.exists():
            data = load_json_results(nsga2_file)
            results_k200.append({
                'approach': approach.replace('approach', 'Approach '),
                'method': 'NSGA-II',
                'metrics': data['metrics']
            })

    # Find best values for k=10
    best_k10 = {
        'heat_sum': max([r['metrics']['heat_sum'] for r in results_k10]),
        'socio_sum': max([r['metrics']['socio_sum'] for r in results_k10]),
        'equity_gini': min([r['metrics']['equity_gini'] for r in results_k10]),
        'population_served': max([r['metrics']['population_served'] for r in results_k10]),
        'olympic_coverage': max([r['metrics']['olympic_coverage'] for r in results_k10])
    }

    # Find best values for k=20
    best_k20 = {
        'heat_sum': max([r['metrics']['heat_sum'] for r in results_k20]),
        'socio_sum': max([r['metrics']['socio_sum'] for r in results_k20]),
        'equity_gini': min([r['metrics']['equity_gini'] for r in results_k20]),
        'population_served': max([r['metrics']['population_served'] for r in results_k20]),
        'olympic_coverage': max([r['metrics']['olympic_coverage'] for r in results_k20])
    }

    # Generate LaTeX table
    latex_lines = []
    latex_lines.append("\\documentclass{article}")
    latex_lines.append("\\usepackage{booktabs}")
    latex_lines.append("\\usepackage{graphicx}")
    latex_lines.append("\\usepackage[normalem]{ulem}")
    latex_lines.append("")
    latex_lines.append("\\begin{document}")
    latex_lines.append("")
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Performance comparison of reward function approaches and optimization methods for cooling shade placement in Los Angeles. Best values in each section are underlined.}")
    latex_lines.append("\\label{tab:results}")
    latex_lines.append("\\resizebox{\\textwidth}{!}{%")
    latex_lines.append("\\begin{tabular}{@{}llrrrrr@{}}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\textbf{Reward Function} & \\textbf{Method} & \\textbf{Heat Sum} & \\textbf{Socio-Vuln} & \\textbf{Gini} $\\downarrow$ & \\textbf{Population} & \\textbf{Olympic \\%} \\\\")
    latex_lines.append("\\midrule")

    # k=10 results - group by approach
    approaches_k10 = {}
    for result in results_k10:
        app = result['approach']
        if app not in approaches_k10:
            approaches_k10[app] = []
        approaches_k10[app].append(result)

    # Sort approaches: Approach 1, Approach 2, Approach 3
    for approach in ['Approach 1', 'Approach 2', 'Approach 3']:
        if approach not in approaches_k10:
            continue

        # Section header
        if approach == 'Approach 1':
            latex_lines.append("\\multicolumn{7}{l}{\\textit{Enhanced Weighted Sum (k=10)}} \\\\")
        elif approach == 'Approach 2':
            latex_lines.append("\\multicolumn{7}{l}{\\textit{Hierarchical/Multiplicative (k=10)}} \\\\")
        elif approach == 'Approach 3':
            latex_lines.append("\\multicolumn{7}{l}{\\textit{Multi-Objective NSGA-II Approach (k=10)}} \\\\")

        # Sort by population served (descending) for better readability
        sorted_results = sorted(approaches_k10[approach],
                              key=lambda x: x['metrics']['population_served'],
                              reverse=True)

        for result in sorted_results:
            m = result['metrics']
            line = f"\\quad {approach} & {result['method']} & "
            line += format_value(m['heat_sum'], best_k10['heat_sum']) + " & "
            line += format_value(m['socio_sum'], best_k10['socio_sum']) + " & "
            line += format_value(m['equity_gini'], best_k10['equity_gini'], is_lower_better=True) + " & "
            line += format_population(m['population_served'], best_k10['population_served']) + " & "
            line += format_value(m['olympic_coverage'], best_k10['olympic_coverage']) + " \\\\"
            latex_lines.append(line)

        latex_lines.append("\\midrule")

    # k=20 results
    latex_lines.append("\\multicolumn{7}{l}{\\textit{k=20 Results}} \\\\")

    # Group by approach
    approaches_k20 = {}
    for result in results_k20:
        app = result['approach']
        if app not in approaches_k20:
            approaches_k20[app] = []
        approaches_k20[app].append(result)

    # Add all k=20 results
    all_k20_sorted = sorted(results_k20,
                           key=lambda x: (x['approach'], -x['metrics']['population_served']))

    for result in all_k20_sorted:
        m = result['metrics']
        line = f"\\quad {result['approach']} & {result['method']} & "
        line += format_value(m['heat_sum'], best_k20['heat_sum']) + " & "
        line += format_value(m['socio_sum'], best_k20['socio_sum']) + " & "
        line += format_value(m['equity_gini'], best_k20['equity_gini'], is_lower_better=True) + " & "
        line += format_population(m['population_served'], best_k20['population_served']) + " & "
        line += format_value(m['olympic_coverage'], best_k20['olympic_coverage']) + " \\\\"
        latex_lines.append(line)

    # Add k=50 results if available
    if results_k50:
        latex_lines.append("\\midrule")
        latex_lines.append("\\multicolumn{7}{l}{\\textit{k=50 Results}} \\\\")

        best_k50 = {
            'heat_sum': max([r['metrics']['heat_sum'] for r in results_k50]),
            'socio_sum': max([r['metrics']['socio_sum'] for r in results_k50]),
            'equity_gini': min([r['metrics']['equity_gini'] for r in results_k50]),
            'population_served': max([r['metrics']['population_served'] for r in results_k50]),
            'olympic_coverage': max([r['metrics']['olympic_coverage'] for r in results_k50])
        }

        all_k50_sorted = sorted(results_k50,
                               key=lambda x: (x['approach'], -x['metrics']['population_served']))

        for result in all_k50_sorted:
            m = result['metrics']
            line = f"\\quad {result['approach']} & {result['method']} & "
            line += format_value(m['heat_sum'], best_k50['heat_sum']) + " & "
            line += format_value(m['socio_sum'], best_k50['socio_sum']) + " & "
            line += format_value(m['equity_gini'], best_k50['equity_gini'], is_lower_better=True) + " & "
            line += format_population(m['population_served'], best_k50['population_served']) + " & "
            line += format_value(m['olympic_coverage'], best_k50['olympic_coverage']) + " \\\\"
            latex_lines.append(line)

    # Add k=100 results if available
    if results_k100:
        latex_lines.append("\\midrule")
        latex_lines.append("\\multicolumn{7}{l}{\\textit{k=100 Results}} \\\\")

        best_k100 = {
            'heat_sum': max([r['metrics']['heat_sum'] for r in results_k100]),
            'socio_sum': max([r['metrics']['socio_sum'] for r in results_k100]),
            'equity_gini': min([r['metrics']['equity_gini'] for r in results_k100]),
            'population_served': max([r['metrics']['population_served'] for r in results_k100]),
            'olympic_coverage': max([r['metrics']['olympic_coverage'] for r in results_k100])
        }

        all_k100_sorted = sorted(results_k100,
                               key=lambda x: (x['approach'], -x['metrics']['population_served']))

        for result in all_k100_sorted:
            m = result['metrics']
            line = f"\\quad {result['approach']} & {result['method']} & "
            line += format_value(m['heat_sum'], best_k100['heat_sum']) + " & "
            line += format_value(m['socio_sum'], best_k100['socio_sum']) + " & "
            line += format_value(m['equity_gini'], best_k100['equity_gini'], is_lower_better=True) + " & "
            line += format_population(m['population_served'], best_k100['population_served']) + " & "
            line += format_value(m['olympic_coverage'], best_k100['olympic_coverage']) + " \\\\"
            latex_lines.append(line)

    # Add k=200 results if available
    if results_k200:
        latex_lines.append("\\midrule")
        latex_lines.append("\\multicolumn{7}{l}{\\textit{k=200 Results}} \\\\")

        best_k200 = {
            'heat_sum': max([r['metrics']['heat_sum'] for r in results_k200]),
            'socio_sum': max([r['metrics']['socio_sum'] for r in results_k200]),
            'equity_gini': min([r['metrics']['equity_gini'] for r in results_k200]),
            'population_served': max([r['metrics']['population_served'] for r in results_k200]),
            'olympic_coverage': max([r['metrics']['olympic_coverage'] for r in results_k200])
        }

        all_k200_sorted = sorted(results_k200,
                               key=lambda x: (x['approach'], -x['metrics']['population_served']))

        for result in all_k200_sorted:
            m = result['metrics']
            line = f"\\quad {result['approach']} & {result['method']} & "
            line += format_value(m['heat_sum'], best_k200['heat_sum']) + " & "
            line += format_value(m['socio_sum'], best_k200['socio_sum']) + " & "
            line += format_value(m['equity_gini'], best_k200['equity_gini'], is_lower_better=True) + " & "
            line += format_population(m['population_served'], best_k200['population_served']) + " & "
            line += format_value(m['olympic_coverage'], best_k200['olympic_coverage']) + " \\\\"
            latex_lines.append(line)

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}%")
    latex_lines.append("}")
    latex_lines.append("\\vspace{0.1cm}")
    latex_lines.append("")
    latex_lines.append("\\smallskip")
    latex_lines.append("\\small")
    latex_lines.append("\\textit{Note:} Heat Sum and Socio-Vuln (socio-vulnerability) are cumulative scores (higher is better). Gini coefficient measures inequality in benefit distribution (lower is better, marked with $\\downarrow$). Population indicates total population served within 500m. Olympic \\% represents percentage of Olympic venues covered. Results grouped by k value (number of shade placements). Approach 3 uses multi-objective optimization with NSGA-II to explore the Pareto frontier.")
    latex_lines.append("\\end{table}")
    latex_lines.append("")
    latex_lines.append("\\end{document}")

    # Write to file
    output_file = Path("/home/fhliang/projects/libero_shade/new_reward/results_table_updated.tex")
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_lines))

    print(f"✓ LaTeX table written to {output_file}")
    print(f"\nTable includes:")
    print(f"  - k=10: {len(results_k10)} results")
    print(f"  - k=20: {len(results_k20)} results")
    print(f"  - k=50: {len(results_k50)} results")
    print(f"  - k=100: {len(results_k100)} results")
    print(f"  - k=200: {len(results_k200)} results")

    return latex_lines

if __name__ == "__main__":
    main()
