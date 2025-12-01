"""
Visualize Engineered Features for Reward Function
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("VISUALIZING ENGINEERED FEATURES")
print("=" * 80)

# Load feature-engineered data
df = pd.read_csv('shade_optimization_data_usc_features.csv')
print(f"\nLoaded: {df.shape[0]} rows × {df.shape[1]} columns")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

# ============================================================================
# VISUALIZATION 1: Five Component Indices
# ============================================================================
print("\n1. Creating component indices visualization...")

fig = plt.figure(figsize=(18, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# Five main indices
indices = [
    ('HEAT_VULNERABILITY_INDEX', 0.30, 'Reds'),
    ('POPULATION_IMPACT_INDEX', 0.25, 'Blues'),
    ('ACCESSIBILITY_INDEX', 0.20, 'Greens'),
    ('EQUITY_INDEX', 0.15, 'Purples'),
    ('COVERAGE_EFFICIENCY_INDEX', 0.10, 'Oranges')
]

for idx, (index_name, weight, cmap) in enumerate(indices):
    if index_name in df.columns:
        # Distribution
        ax1 = fig.add_subplot(gs[idx // 2, (idx % 2) * 1])
        ax1.hist(df[index_name], bins=40, color=plt.cm.get_cmap(cmap)(0.6),
                edgecolor='black', alpha=0.7)
        ax1.axvline(df[index_name].mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {df[index_name].mean():.3f}')
        ax1.set_xlabel(index_name.replace('_', ' ').title(), fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title(f'{index_name.replace("_", " ")}\n(Weight: {weight:.0%})',
                     fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

# Reward Potential Score
ax_reward = fig.add_subplot(gs[2, :])
ax_reward.hist(df['REWARD_POTENTIAL_SCORE'], bins=50,
              color='darkgreen', edgecolor='black', alpha=0.7)
ax_reward.axvline(df['REWARD_POTENTIAL_SCORE'].mean(), color='red',
                 linestyle='--', linewidth=2.5,
                 label=f'Mean: {df["REWARD_POTENTIAL_SCORE"].mean():.3f}')
ax_reward.axvline(df['REWARD_POTENTIAL_SCORE'].median(), color='orange',
                 linestyle='--', linewidth=2.5,
                 label=f'Median: {df["REWARD_POTENTIAL_SCORE"].median():.3f}')
ax_reward.set_xlabel('Reward Potential Score', fontsize=13)
ax_reward.set_ylabel('Frequency', fontsize=13)
ax_reward.set_title('OVERALL REWARD POTENTIAL SCORE (Weighted Composite)',
                   fontsize=14, fontweight='bold')
ax_reward.legend(fontsize=11)
ax_reward.grid(True, alpha=0.3)

plt.savefig('eda_outputs_usc/component_indices_distribution.png',
           dpi=300, bbox_inches='tight')
print("✓ Saved: eda_outputs_usc/component_indices_distribution.png")

# ============================================================================
# VISUALIZATION 2: Geographic Heatmaps of Indices
# ============================================================================
print("\n2. Creating geographic heatmaps...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

geo_features = [
    ('HEAT_VULNERABILITY_INDEX', 'Heat Vulnerability', 'RdYlBu_r'),
    ('POPULATION_IMPACT_INDEX', 'Population Impact', 'YlGnBu'),
    ('ACCESSIBILITY_INDEX', 'Accessibility Gap', 'YlOrRd'),
    ('EQUITY_INDEX', 'Equity Need', 'PuRd'),
    ('COVERAGE_EFFICIENCY_INDEX', 'Coverage Priority', 'OrRd'),
    ('REWARD_POTENTIAL_SCORE', 'Overall Reward Potential', 'viridis')
]

for idx, (feature, title, cmap) in enumerate(geo_features):
    if feature in df.columns:
        scatter = axes[idx].scatter(df['longitude'], df['latitude'],
                                   c=df[feature], cmap=cmap,
                                   s=30, alpha=0.7, edgecolors='black',
                                   linewidth=0.3)
        cbar = plt.colorbar(scatter, ax=axes[idx])
        cbar.set_label(f'{title} Score', fontsize=10)
        axes[idx].set_xlabel('Longitude', fontsize=11)
        axes[idx].set_ylabel('Latitude', fontsize=11)
        axes[idx].set_title(f'{title}', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_outputs_usc/geographic_component_heatmaps.png',
           dpi=300, bbox_inches='tight')
print("✓ Saved: eda_outputs_usc/geographic_component_heatmaps.png")

# ============================================================================
# VISUALIZATION 3: Top 50 Priority Locations
# ============================================================================
print("\n3. Creating top 50 priority visualization...")

top_50 = df.nlargest(50, 'REWARD_POTENTIAL_SCORE')
rest = df[~df.index.isin(top_50.index)]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Map view
scatter1 = axes[0, 0].scatter(rest['longitude'], rest['latitude'],
                             c='lightgray', s=20, alpha=0.4,
                             label='Other locations')
scatter2 = axes[0, 0].scatter(top_50['longitude'], top_50['latitude'],
                             c=top_50['REWARD_POTENTIAL_SCORE'],
                             cmap='Reds', s=100, alpha=0.8,
                             edgecolors='black', linewidth=1,
                             label='Top 50 priority')
plt.colorbar(scatter2, ax=axes[0, 0], label='Reward Potential Score')
axes[0, 0].set_xlabel('Longitude', fontsize=12)
axes[0, 0].set_ylabel('Latitude', fontsize=12)
axes[0, 0].set_title('Top 50 Priority Locations for Shade Placement',
                    fontsize=13, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Component comparison (top 50 vs rest)
component_comparison = pd.DataFrame({
    'Heat\nVuln': [top_50['HEAT_VULNERABILITY_INDEX'].mean(),
                   rest['HEAT_VULNERABILITY_INDEX'].mean()],
    'Population\nImpact': [top_50['POPULATION_IMPACT_INDEX'].mean(),
                          rest['POPULATION_IMPACT_INDEX'].mean()],
    'Access\nGap': [top_50['ACCESSIBILITY_INDEX'].mean(),
                   rest['ACCESSIBILITY_INDEX'].mean()],
    'Equity\nNeed': [top_50['EQUITY_INDEX'].mean(),
                    rest['EQUITY_INDEX'].mean()],
    'Coverage\nPriority': [top_50['COVERAGE_EFFICIENCY_INDEX'].mean(),
                          rest['COVERAGE_EFFICIENCY_INDEX'].mean()]
}, index=['Top 50', 'Rest'])

x = np.arange(len(component_comparison.columns))
width = 0.35
axes[0, 1].bar(x - width/2, component_comparison.loc['Top 50'],
              width, label='Top 50', color='darkred', alpha=0.8)
axes[0, 1].bar(x + width/2, component_comparison.loc['Rest'],
              width, label='Rest', color='lightblue', alpha=0.8)
axes[0, 1].set_ylabel('Mean Score', fontsize=12)
axes[0, 1].set_title('Component Index Comparison: Top 50 vs Rest',
                    fontsize=13, fontweight='bold')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(component_comparison.columns, fontsize=10)
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Reward score distribution
axes[1, 0].hist(rest['REWARD_POTENTIAL_SCORE'], bins=40,
               color='lightblue', alpha=0.6, label='Rest', edgecolor='black')
axes[1, 0].hist(top_50['REWARD_POTENTIAL_SCORE'], bins=20,
               color='darkred', alpha=0.8, label='Top 50', edgecolor='black')
axes[1, 0].axvline(top_50['REWARD_POTENTIAL_SCORE'].min(), color='red',
                  linestyle='--', linewidth=2,
                  label=f'Top 50 threshold: {top_50["REWARD_POTENTIAL_SCORE"].min():.3f}')
axes[1, 0].set_xlabel('Reward Potential Score', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Reward Score Distribution', fontsize=13, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# Priority tiers
tier_counts = df['priority_tier'].value_counts().sort_index()
colors_tier = ['#d7191c', '#fdae61', '#ffffbf', '#abd9e9', '#2c7bb6']
axes[1, 1].barh(range(len(tier_counts)), tier_counts.values,
               color=colors_tier[::-1], alpha=0.8, edgecolor='black')
axes[1, 1].set_yticks(range(len(tier_counts)))
axes[1, 1].set_yticklabels(tier_counts.index, fontsize=11)
axes[1, 1].set_xlabel('Number of Locations', fontsize=12)
axes[1, 1].set_title('Priority Tier Distribution', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='x')

# Add count labels
for i, (tier, count) in enumerate(tier_counts.items()):
    pct = count / len(df) * 100
    axes[1, 1].text(count + 5, i, f'{count} ({pct:.1f}%)',
                   va='center', fontsize=10)

plt.tight_layout()
plt.savefig('eda_outputs_usc/top50_priority_analysis.png',
           dpi=300, bbox_inches='tight')
print("✓ Saved: eda_outputs_usc/top50_priority_analysis.png")

# ============================================================================
# VISUALIZATION 4: Correlation Between Components
# ============================================================================
print("\n4. Creating component correlation matrix...")

composite_cols = [
    'HEAT_VULNERABILITY_INDEX',
    'POPULATION_IMPACT_INDEX',
    'ACCESSIBILITY_INDEX',
    'EQUITY_INDEX',
    'COVERAGE_EFFICIENCY_INDEX',
    'REWARD_POTENTIAL_SCORE'
]

available_composites = [col for col in composite_cols if col in df.columns]
corr_matrix = df[available_composites].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f',
           cmap='coolwarm', center=0, square=True,
           linewidths=1, cbar_kws={'label': 'Correlation'},
           ax=ax, vmin=-1, vmax=1)

# Simplify labels
labels = [col.replace('_INDEX', '').replace('_', '\n').title()
         for col in available_composites]
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
ax.set_yticklabels(labels, rotation=0, fontsize=11)
ax.set_title('Correlation Matrix: Reward Function Components',
            fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('eda_outputs_usc/component_correlation_matrix.png',
           dpi=300, bbox_inches='tight')
print("✓ Saved: eda_outputs_usc/component_correlation_matrix.png")

# ============================================================================
# VISUALIZATION 5: Interaction Features
# ============================================================================
print("\n5. Creating interaction features visualization...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

interactions = [
    ('interact_heat_population', 'Heat × Population', 'Reds'),
    ('interact_heat_equity', 'Heat × Equity (EJ)', 'Purples'),
    ('interact_population_access', 'Population × Access Gap', 'Blues'),
    ('interact_equity_access', 'Equity × Access Gap', 'Greens')
]

for idx, (feature, title, cmap) in enumerate(interactions):
    if feature in df.columns:
        row, col = idx // 2, idx % 2

        # Histogram
        axes[row, col].hist(df[feature], bins=40,
                           color=plt.cm.get_cmap(cmap)(0.6),
                           edgecolor='black', alpha=0.7)
        axes[row, col].axvline(df[feature].mean(), color='red',
                              linestyle='--', linewidth=2,
                              label=f'Mean: {df[feature].mean():.3f}')
        axes[row, col].set_xlabel(title, fontsize=12)
        axes[row, col].set_ylabel('Frequency', fontsize=12)
        axes[row, col].set_title(f'{title}\n(Interaction Feature)',
                                fontsize=12, fontweight='bold')
        axes[row, col].legend(fontsize=10)
        axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_outputs_usc/interaction_features.png',
           dpi=300, bbox_inches='tight')
print("✓ Saved: eda_outputs_usc/interaction_features.png")

# ============================================================================
# VISUALIZATION 6: Feature Contribution to Reward Score
# ============================================================================
print("\n6. Creating feature contribution analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Scatter plots showing contribution of each index to reward score
scatter_configs = [
    ('HEAT_VULNERABILITY_INDEX', 'Heat Vulnerability', 0.30, 'Reds'),
    ('POPULATION_IMPACT_INDEX', 'Population Impact', 0.25, 'Blues'),
    ('ACCESSIBILITY_INDEX', 'Accessibility Gap', 0.20, 'Greens'),
    ('EQUITY_INDEX', 'Equity Need', 0.15, 'Purples')
]

for idx, (feature, title, weight, cmap) in enumerate(scatter_configs):
    if feature in df.columns:
        row, col = idx // 2, idx % 2

        scatter = axes[row, col].scatter(df[feature],
                                        df['REWARD_POTENTIAL_SCORE'],
                                        c=df[feature], cmap=cmap,
                                        s=20, alpha=0.6,
                                        edgecolors='black', linewidth=0.3)

        # Add trend line
        z = np.polyfit(df[feature], df['REWARD_POTENTIAL_SCORE'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df[feature].min(), df[feature].max(), 100)
        axes[row, col].plot(x_trend, p(x_trend), 'r--', linewidth=2,
                           label=f'Trend (weight={weight:.0%})')

        # Calculate correlation
        corr = df[feature].corr(df['REWARD_POTENTIAL_SCORE'])

        axes[row, col].set_xlabel(f'{title} Score', fontsize=11)
        axes[row, col].set_ylabel('Reward Potential Score', fontsize=11)
        axes[row, col].set_title(f'{title} vs Reward Score\n(r={corr:.3f})',
                                fontsize=12, fontweight='bold')
        axes[row, col].legend(fontsize=9)
        axes[row, col].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[row, col], label=f'{title} Score')

plt.tight_layout()
plt.savefig('eda_outputs_usc/feature_contribution_to_reward.png',
           dpi=300, bbox_inches='tight')
print("✓ Saved: eda_outputs_usc/feature_contribution_to_reward.png")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print("\nComponent Index Statistics:")
print("-" * 80)
for index_name, weight, _ in indices:
    if index_name in df.columns:
        stats = df[index_name].describe()
        print(f"\n{index_name} (Weight: {weight:.0%}):")
        print(f"  Mean:   {stats['mean']:.3f}")
        print(f"  Std:    {stats['std']:.3f}")
        print(f"  Min:    {stats['min']:.3f}")
        print(f"  25%:    {stats['25%']:.3f}")
        print(f"  Median: {stats['50%']:.3f}")
        print(f"  75%:    {stats['75%']:.3f}")
        print(f"  Max:    {stats['max']:.3f}")

print("\n" + "=" * 80)
print("REWARD POTENTIAL SCORE:")
print("-" * 80)
stats = df['REWARD_POTENTIAL_SCORE'].describe()
print(f"  Mean:   {stats['mean']:.3f}")
print(f"  Std:    {stats['std']:.3f}")
print(f"  Min:    {stats['min']:.3f}")
print(f"  25%:    {stats['25%']:.3f}")
print(f"  Median: {stats['50%']:.3f}")
print(f"  75%:    {stats['75%']:.3f}")
print(f"  Max:    {stats['max']:.3f}")

print("\n" + "=" * 80)
print("TOP 50 PRIORITY LOCATIONS:")
print("-" * 80)
print(f"  Min Reward Score: {top_50['REWARD_POTENTIAL_SCORE'].min():.3f}")
print(f"  Max Reward Score: {top_50['REWARD_POTENTIAL_SCORE'].max():.3f}")
print(f"  Mean Reward Score: {top_50['REWARD_POTENTIAL_SCORE'].mean():.3f}")
print(f"  50th Location Score: {df.nlargest(50, 'REWARD_POTENTIAL_SCORE').iloc[-1]['REWARD_POTENTIAL_SCORE']:.3f}")

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE!")
print("=" * 80)
print("\nGenerated Visualizations:")
print("  1. component_indices_distribution.png - Distribution of 5 component indices")
print("  2. geographic_component_heatmaps.png - Geographic heatmaps of all indices")
print("  3. top50_priority_analysis.png - Top 50 priority locations analysis")
print("  4. component_correlation_matrix.png - Correlation between components")
print("  5. interaction_features.png - Interaction feature distributions")
print("  6. feature_contribution_to_reward.png - How each index affects reward score")
print("=" * 80)
