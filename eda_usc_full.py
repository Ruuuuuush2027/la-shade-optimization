"""
Comprehensive EDA for USC Shade Optimization Data
Focus: land_surface_temp_c imputation and thorough analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import BallTree
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load data
print("=" * 80)
print("LOADING USC SHADE OPTIMIZATION DATA")
print("=" * 80)
df = pd.read_csv('shade_optimization_data_usc.csv')

print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================================
# SECTION 1: BASIC STRUCTURE
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 1: DATASET STRUCTURE")
print("=" * 80)

print("\nColumn Data Types:")
print(df.dtypes.value_counts())

print("\nFirst 5 rows:")
print(df.head())

print("\nBasic Statistics:")
print(df.describe())

# ============================================================================
# SECTION 2: MISSING VALUE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: MISSING VALUE ANALYSIS")
print("=" * 80)

missing_df = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percent': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percent', ascending=False)

print(f"\nColumns with Missing Values: {len(missing_df)}/{df.shape[1]}")
print("\nTop Missing Features:")
print(missing_df.to_string(index=False))

# Visualize missing values
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Missing value heatmap (sample if too many columns)
sample_cols = df.columns[df.isnull().any()].tolist()[:30]  # Top 30 with missing
if len(sample_cols) > 0:
    sns.heatmap(df[sample_cols].isnull(), cbar=True, yticklabels=False, ax=axes[0], cmap='viridis')
    axes[0].set_title('Missing Value Pattern (Top 30 Features)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Features')
else:
    axes[0].text(0.5, 0.5, 'No Missing Values!', ha='center', va='center', fontsize=16)
    axes[0].set_title('Missing Value Pattern')

# Missing percentage bar plot
if len(missing_df) > 0:
    top_missing = missing_df.head(20)
    axes[1].barh(range(len(top_missing)), top_missing['Missing_Percent'], color='coral')
    axes[1].set_yticks(range(len(top_missing)))
    axes[1].set_yticklabels(top_missing['Column'], fontsize=10)
    axes[1].set_xlabel('Missing Percentage (%)', fontsize=12)
    axes[1].set_title('Top 20 Features by Missing %', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
else:
    axes[1].text(0.5, 0.5, 'No Missing Values!', ha='center', va='center', fontsize=16)

plt.tight_layout()
plt.savefig('eda_outputs_usc/missing_values_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: eda_outputs_usc/missing_values_analysis.png")

# ============================================================================
# SECTION 3: LAND_SURFACE_TEMP_C DEEP DIVE
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: LAND_SURFACE_TEMP_C ANALYSIS (CRITICAL FEATURE)")
print("=" * 80)

lst_missing = df['land_surface_temp_c'].isnull().sum()
lst_missing_pct = (lst_missing / len(df) * 100)

print(f"\nMissing Values: {lst_missing} ({lst_missing_pct:.2f}%)")
print(f"Available Values: {len(df) - lst_missing} ({100 - lst_missing_pct:.2f}%)")

if lst_missing < len(df):
    lst_stats = df['land_surface_temp_c'].describe()
    print(f"\nDescriptive Statistics (Non-Missing):")
    print(lst_stats)

    # Visualize distribution
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Histogram
    axes[0, 0].hist(df['land_surface_temp_c'].dropna(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(df['land_surface_temp_c'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df["land_surface_temp_c"].mean():.2f}°C', linewidth=2)
    axes[0, 0].axvline(df['land_surface_temp_c'].median(), color='orange', linestyle='--',
                       label=f'Median: {df["land_surface_temp_c"].median():.2f}°C', linewidth=2)
    axes[0, 0].set_xlabel('Temperature (°C)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Land Surface Temperature Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Box plot
    axes[0, 1].boxplot(df['land_surface_temp_c'].dropna(), vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[0, 1].set_ylabel('Temperature (°C)', fontsize=12)
    axes[0, 1].set_title('Land Surface Temperature Boxplot', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Spatial distribution (lat/lon scatter)
    scatter = axes[1, 0].scatter(df['longitude'], df['latitude'],
                                 c=df['land_surface_temp_c'],
                                 cmap='RdYlBu_r', s=20, alpha=0.6)
    plt.colorbar(scatter, ax=axes[1, 0], label='Temperature (°C)')
    axes[1, 0].set_xlabel('Longitude', fontsize=12)
    axes[1, 0].set_ylabel('Latitude', fontsize=12)
    axes[1, 0].set_title('Spatial Distribution of Land Surface Temperature', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # QQ plot for normality
    from scipy import stats
    stats.probplot(df['land_surface_temp_c'].dropna(), dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('eda_outputs_usc/land_surface_temp_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: eda_outputs_usc/land_surface_temp_distribution.png")

# ============================================================================
# SECTION 4: SPATIAL IMPUTATION FOR LAND_SURFACE_TEMP_C
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: IMPUTING LAND_SURFACE_TEMP_C")
print("=" * 80)

df_imputed = df.copy()

if lst_missing > 0:
    print("\nStrategy: Using multiple imputation methods and comparing results")
    print("-" * 80)

    # Method 1: Simple mean imputation (baseline)
    mean_value = df['land_surface_temp_c'].mean()
    print(f"\n1. MEAN IMPUTATION")
    print(f"   Fill value: {mean_value:.2f}°C")

    # Method 2: KNN Spatial Imputation (using lat/lon + correlated features)
    print(f"\n2. KNN SPATIAL IMPUTATION")

    # Find features correlated with land_surface_temp_c
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_with_lst = df[numeric_cols].corr()['land_surface_temp_c'].abs().sort_values(ascending=False)

    # Select top correlated features (excluding target)
    top_features = corr_with_lst[1:11].index.tolist()  # Top 10 correlated features
    print(f"   Top correlated features: {top_features[:5]}...")

    # Prepare features for KNN
    features_for_knn = ['latitude', 'longitude'] + top_features
    features_for_knn = [f for f in features_for_knn if f in df.columns and f != 'land_surface_temp_c']

    print(f"   Features used: latitude, longitude + {len(top_features)} correlated features")
    print(f"   K neighbors: 5")

    # KNN Imputation
    knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
    impute_data = df[features_for_knn + ['land_surface_temp_c']].copy()
    imputed_values = knn_imputer.fit_transform(impute_data)
    df_knn = pd.DataFrame(imputed_values, columns=features_for_knn + ['land_surface_temp_c'])

    # Method 3: BallTree Spatial (haversine distance - great circle)
    print(f"\n3. BALLTREE SPATIAL IMPUTATION (Haversine)")
    print(f"   Using geographic distance (great-circle)")
    print(f"   K neighbors: 3")

    # Separate data with and without LST
    has_lst = df[df['land_surface_temp_c'].notna()].copy()
    missing_lst = df[df['land_surface_temp_c'].isna()].copy()

    if len(has_lst) > 0 and len(missing_lst) > 0:
        # Convert to radians for haversine
        coords_with_data = np.radians(has_lst[['latitude', 'longitude']].values)
        coords_missing = np.radians(missing_lst[['latitude', 'longitude']].values)

        # Build BallTree
        tree = BallTree(coords_with_data, metric='haversine')

        # Find nearest neighbors
        distances, indices = tree.query(coords_missing, k=3)

        # Weighted average based on inverse distance
        # Convert haversine distance to km (Earth radius = 6371 km)
        distances_km = distances * 6371

        # Avoid division by zero
        weights = 1 / (distances_km + 1e-10)
        weights = weights / weights.sum(axis=1, keepdims=True)

        # Impute using weighted average
        imputed_temps = (has_lst.iloc[indices.flatten()]['land_surface_temp_c'].values.reshape(indices.shape) * weights).sum(axis=1)

        # Fill in the missing values
        df_balltree = df.copy()
        df_balltree.loc[df_balltree['land_surface_temp_c'].isna(), 'land_surface_temp_c'] = imputed_temps

        print(f"   Average distance to nearest neighbor: {distances_km[:, 0].mean():.2f} km")
        print(f"   Imputed temperature range: [{imputed_temps.min():.2f}, {imputed_temps.max():.2f}]°C")
    else:
        df_balltree = df.copy()

    # Method 4: Random Forest Regression
    print(f"\n4. RANDOM FOREST REGRESSION")

    # Use all available numeric features
    feature_cols = [col for col in numeric_cols if col != 'land_surface_temp_c' and df[col].notna().sum() > len(df) * 0.5]
    print(f"   Features used: {len(feature_cols)} numeric features")

    # Prepare training data
    train_data = df[df['land_surface_temp_c'].notna()][feature_cols].copy()
    train_target = df[df['land_surface_temp_c'].notna()]['land_surface_temp_c'].copy()
    test_data = df[df['land_surface_temp_c'].isna()][feature_cols].copy()

    # Fill NaN in features with median
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    train_data_imputed = imputer.fit_transform(train_data)
    test_data_imputed = imputer.transform(test_data)

    if len(test_data) > 0:
        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        rf.fit(train_data_imputed, train_target)

        # Predict missing values
        rf_predictions = rf.predict(test_data_imputed)

        df_rf = df.copy()
        df_rf.loc[df_rf['land_surface_temp_c'].isna(), 'land_surface_temp_c'] = rf_predictions

        print(f"   RF Score: {rf.score(train_data_imputed, train_target):.4f}")
        print(f"   Imputed temperature range: [{rf_predictions.min():.2f}, {rf_predictions.max():.2f}]°C")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n   Top 5 Important Features:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"      {row['feature']}: {row['importance']:.4f}")
    else:
        df_rf = df.copy()

    # Compare methods
    print(f"\n" + "=" * 80)
    print("IMPUTATION COMPARISON")
    print("=" * 80)

    comparison = pd.DataFrame({
        'Method': ['Mean', 'KNN Spatial', 'BallTree', 'Random Forest'],
        'Mean_Temp': [
            mean_value,
            df_knn['land_surface_temp_c'].mean() if 'df_knn' in locals() else mean_value,
            df_balltree['land_surface_temp_c'].mean() if 'df_balltree' in locals() else mean_value,
            df_rf['land_surface_temp_c'].mean() if 'df_rf' in locals() else mean_value
        ],
        'Std_Temp': [
            df['land_surface_temp_c'].std(),
            df_knn['land_surface_temp_c'].std() if 'df_knn' in locals() else df['land_surface_temp_c'].std(),
            df_balltree['land_surface_temp_c'].std() if 'df_balltree' in locals() else df['land_surface_temp_c'].std(),
            df_rf['land_surface_temp_c'].std() if 'df_rf' in locals() else df['land_surface_temp_c'].std()
        ]
    })

    print("\n")
    print(comparison.to_string(index=False))

    # Use BallTree method (best for spatial data)
    print(f"\n✓ SELECTED METHOD: BallTree Spatial (Haversine) - Best for geographic data")
    df_imputed = df_balltree.copy()

    # Visualize imputation results
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Original vs Imputed
    axes[0, 0].hist(df['land_surface_temp_c'].dropna(), bins=30, alpha=0.6,
                    label='Original', color='blue', edgecolor='black')
    axes[0, 0].hist(df_imputed.loc[df['land_surface_temp_c'].isna(), 'land_surface_temp_c'],
                    bins=30, alpha=0.6, label='Imputed', color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Temperature (°C)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Original vs Imputed Values', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Spatial map - Imputed locations
    imputed_mask = df['land_surface_temp_c'].isna()
    axes[0, 1].scatter(df.loc[~imputed_mask, 'longitude'], df.loc[~imputed_mask, 'latitude'],
                      c='blue', s=10, alpha=0.3, label='Original')
    axes[0, 1].scatter(df.loc[imputed_mask, 'longitude'], df.loc[imputed_mask, 'latitude'],
                      c='red', s=30, alpha=0.8, marker='x', label='Imputed')
    axes[0, 1].set_xlabel('Longitude', fontsize=12)
    axes[0, 1].set_ylabel('Latitude', fontsize=12)
    axes[0, 1].set_title('Spatial Distribution of Imputed Points', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Complete spatial heatmap
    scatter = axes[1, 0].scatter(df_imputed['longitude'], df_imputed['latitude'],
                                c=df_imputed['land_surface_temp_c'],
                                cmap='RdYlBu_r', s=20, alpha=0.6)
    plt.colorbar(scatter, ax=axes[1, 0], label='Temperature (°C)')
    axes[1, 0].set_xlabel('Longitude', fontsize=12)
    axes[1, 0].set_ylabel('Latitude', fontsize=12)
    axes[1, 0].set_title('Complete Dataset (After Imputation)', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Box plot comparison
    box_data = [df['land_surface_temp_c'].dropna(),
                df_imputed.loc[imputed_mask, 'land_surface_temp_c']]
    axes[1, 1].boxplot(box_data, labels=['Original', 'Imputed'], patch_artist=True,
                      boxprops=dict(alpha=0.7))
    axes[1, 1].set_ylabel('Temperature (°C)', fontsize=12)
    axes[1, 1].set_title('Distribution Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('eda_outputs_usc/land_surface_temp_imputation.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: eda_outputs_usc/land_surface_temp_imputation.png")

else:
    print("\n✓ No missing values in land_surface_temp_c - no imputation needed!")

# ============================================================================
# SECTION 5: FEATURE CORRELATIONS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5: CORRELATION ANALYSIS")
print("=" * 80)

# Compute correlation matrix
numeric_df = df_imputed.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

# Top correlations with land_surface_temp_c
lst_correlations = corr_matrix['land_surface_temp_c'].abs().sort_values(ascending=False)[1:21]
print("\nTop 20 Features Correlated with land_surface_temp_c:")
for idx, (feature, corr) in enumerate(lst_correlations.items(), 1):
    actual_corr = corr_matrix.loc[feature, 'land_surface_temp_c']
    print(f"{idx:2d}. {feature:40s} {actual_corr:7.4f}")

# Visualize top correlations
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Bar plot of top correlations
top_corr = corr_matrix['land_surface_temp_c'].abs().sort_values(ascending=False)[1:21]
colors = ['green' if corr_matrix.loc[f, 'land_surface_temp_c'] > 0 else 'red' for f in top_corr.index]
axes[0].barh(range(len(top_corr)), [corr_matrix.loc[f, 'land_surface_temp_c'] for f in top_corr.index],
            color=colors, alpha=0.7)
axes[0].set_yticks(range(len(top_corr)))
axes[0].set_yticklabels(top_corr.index, fontsize=10)
axes[0].set_xlabel('Correlation Coefficient', fontsize=12)
axes[0].set_title('Top 20 Correlations with land_surface_temp_c', fontsize=14, fontweight='bold')
axes[0].axvline(0, color='black', linewidth=0.8)
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3)

# Correlation heatmap (top features)
top_features_for_heatmap = ['land_surface_temp_c'] + lst_correlations.head(15).index.tolist()
sns.heatmap(corr_matrix.loc[top_features_for_heatmap, top_features_for_heatmap],
            annot=False, cmap='coolwarm', center=0, ax=axes[1],
            cbar_kws={'label': 'Correlation'})
axes[1].set_title('Correlation Heatmap (Top Features)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('eda_outputs_usc/correlation_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: eda_outputs_usc/correlation_analysis.png")

# ============================================================================
# SECTION 6: FEATURE DISTRIBUTIONS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 6: KEY FEATURE DISTRIBUTIONS")
print("=" * 80)

# Select important features for visualization
important_features = [
    'land_surface_temp_c', 'urban_heat_idx', 'tree_percent_w',
    'pm25', 'cva_poverty', 'cva_sovi_score',
    'lashade_treecanopy', 'lashade_tc_gap', 'lashade_pctpov'
]

available_features = [f for f in important_features if f in df_imputed.columns]

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()

for idx, feature in enumerate(available_features):
    if idx < 9:
        data = df_imputed[feature].dropna()
        axes[idx].hist(data, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
        axes[idx].axvline(data.mean(), color='red', linestyle='--',
                         label=f'Mean: {data.mean():.2f}', linewidth=2)
        axes[idx].set_xlabel(feature, fontsize=10)
        axes[idx].set_ylabel('Frequency', fontsize=10)
        axes[idx].set_title(f'{feature}', fontsize=11, fontweight='bold')
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_outputs_usc/feature_distributions.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: eda_outputs_usc/feature_distributions.png")

# ============================================================================
# SECTION 7: OUTLIER DETECTION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 7: OUTLIER DETECTION (land_surface_temp_c)")
print("=" * 80)

# IQR method
Q1 = df_imputed['land_surface_temp_c'].quantile(0.25)
Q3 = df_imputed['land_surface_temp_c'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_imputed[(df_imputed['land_surface_temp_c'] < lower_bound) |
                      (df_imputed['land_surface_temp_c'] > upper_bound)]

print(f"\nIQR Method:")
print(f"  Q1: {Q1:.2f}°C")
print(f"  Q3: {Q3:.2f}°C")
print(f"  IQR: {IQR:.2f}°C")
print(f"  Lower Bound: {lower_bound:.2f}°C")
print(f"  Upper Bound: {upper_bound:.2f}°C")
print(f"  Number of Outliers: {len(outliers)} ({len(outliers)/len(df_imputed)*100:.2f}%)")

if len(outliers) > 0:
    print(f"\nOutlier Statistics:")
    print(f"  Min Outlier: {outliers['land_surface_temp_c'].min():.2f}°C")
    print(f"  Max Outlier: {outliers['land_surface_temp_c'].max():.2f}°C")

# ============================================================================
# SECTION 8: GEOGRAPHIC ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 8: GEOGRAPHIC COVERAGE")
print("=" * 80)

print(f"\nLatitude Range: [{df_imputed['latitude'].min():.4f}, {df_imputed['latitude'].max():.4f}]")
print(f"Longitude Range: [{df_imputed['longitude'].min():.4f}, {df_imputed['longitude'].max():.4f}]")

# Create geographic heatmaps
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Temperature
scatter1 = axes[0, 0].scatter(df_imputed['longitude'], df_imputed['latitude'],
                             c=df_imputed['land_surface_temp_c'],
                             cmap='RdYlBu_r', s=15, alpha=0.7)
plt.colorbar(scatter1, ax=axes[0, 0], label='Temperature (°C)')
axes[0, 0].set_title('Land Surface Temperature', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Longitude')
axes[0, 0].set_ylabel('Latitude')

# Urban Heat Index
if 'urban_heat_idx' in df_imputed.columns:
    scatter2 = axes[0, 1].scatter(df_imputed['longitude'], df_imputed['latitude'],
                                 c=df_imputed['urban_heat_idx'],
                                 cmap='YlOrRd', s=15, alpha=0.7)
    plt.colorbar(scatter2, ax=axes[0, 1], label='UHI')
    axes[0, 1].set_title('Urban Heat Index', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Longitude')
    axes[0, 1].set_ylabel('Latitude')

# Tree Canopy
if 'tree_percent_w' in df_imputed.columns:
    scatter3 = axes[1, 0].scatter(df_imputed['longitude'], df_imputed['latitude'],
                                 c=df_imputed['tree_percent_w'],
                                 cmap='Greens', s=15, alpha=0.7)
    plt.colorbar(scatter3, ax=axes[1, 0], label='Tree %')
    axes[1, 0].set_title('Tree Canopy Coverage', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Longitude')
    axes[1, 0].set_ylabel('Latitude')

# Social Vulnerability
if 'cva_sovi_score' in df_imputed.columns:
    scatter4 = axes[1, 1].scatter(df_imputed['longitude'], df_imputed['latitude'],
                                 c=df_imputed['cva_sovi_score'],
                                 cmap='plasma', s=15, alpha=0.7)
    plt.colorbar(scatter4, ax=axes[1, 1], label='SOVI Score')
    axes[1, 1].set_title('Social Vulnerability Index', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Longitude')
    axes[1, 1].set_ylabel('Latitude')

plt.tight_layout()
plt.savefig('eda_outputs_usc/geographic_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: eda_outputs_usc/geographic_analysis.png")

# ============================================================================
# SECTION 9: SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 9: SAVING RESULTS")
print("=" * 80)

# Save imputed dataset
output_file = 'shade_optimization_data_usc_cleaned.csv'
df_imputed.to_csv(output_file, index=False)
print(f"\n✓ Saved cleaned dataset: {output_file}")

# Save summary report
with open('eda_outputs_usc/eda_summary_report.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("USC SHADE OPTIMIZATION DATA - EDA SUMMARY REPORT\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    f.write(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n")

    f.write("LAND_SURFACE_TEMP_C ANALYSIS:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Missing Values: {lst_missing} ({lst_missing_pct:.2f}%)\n")
    f.write(f"Imputation Method: BallTree Spatial (Haversine)\n")
    f.write(f"After Imputation - Mean: {df_imputed['land_surface_temp_c'].mean():.2f}°C\n")
    f.write(f"After Imputation - Std: {df_imputed['land_surface_temp_c'].std():.2f}°C\n")
    f.write(f"After Imputation - Range: [{df_imputed['land_surface_temp_c'].min():.2f}, {df_imputed['land_surface_temp_c'].max():.2f}]°C\n\n")

    f.write("TOP 10 FEATURES CORRELATED WITH LAND_SURFACE_TEMP_C:\n")
    f.write("-" * 80 + "\n")
    for idx, (feature, _) in enumerate(lst_correlations.head(10).items(), 1):
        actual_corr = corr_matrix.loc[feature, 'land_surface_temp_c']
        f.write(f"{idx:2d}. {feature:40s} {actual_corr:7.4f}\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("All visualizations saved in eda_outputs_usc/\n")

print("✓ Saved: eda_outputs_usc/eda_summary_report.txt")

print("\n" + "=" * 80)
print("EDA COMPLETE!")
print("=" * 80)
print("\nGenerated Files:")
print("  1. shade_optimization_data_usc_cleaned.csv (imputed dataset)")
print("  2. eda_outputs_usc/missing_values_analysis.png")
print("  3. eda_outputs_usc/land_surface_temp_distribution.png")
print("  4. eda_outputs_usc/land_surface_temp_imputation.png")
print("  5. eda_outputs_usc/correlation_analysis.png")
print("  6. eda_outputs_usc/feature_distributions.png")
print("  7. eda_outputs_usc/geographic_analysis.png")
print("  8. eda_outputs_usc/eda_summary_report.txt")
print("=" * 80)
