"""
Feature Engineering for USC Shade Optimization
Aligned with 5-component Reward Function:
1. Heat Vulnerability (30%)
2. Population Impact (25%)
3. Accessibility (20%)
4. Equity (15%)
5. Coverage Efficiency (10%)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import BallTree
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FEATURE ENGINEERING FOR SHADE OPTIMIZATION REWARD FUNCTION")
print("=" * 80)

# Load cleaned data
df = pd.read_csv('shade_optimization_data_usc_cleaned.csv')
print(f"\nLoaded Dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# Create copy for feature engineering
df_eng = df.copy()

# ============================================================================
# COMPONENT 1: HEAT VULNERABILITY FEATURES (30% of reward)
# ============================================================================
print("\n" + "=" * 80)
print("COMPONENT 1: HEAT VULNERABILITY FEATURES")
print("=" * 80)

# 1.1 Temperature Severity Score (normalized land surface temp)
scaler_temp = MinMaxScaler()
df_eng['heat_temp_severity'] = scaler_temp.fit_transform(df_eng[['land_surface_temp_c']])
print("\n✓ heat_temp_severity: Normalized land surface temperature [0-1]")

# 1.2 Urban Heat Island Intensity (if available)
if 'urban_heat_idx' in df_eng.columns:
    scaler_uhi = MinMaxScaler()
    df_eng['heat_uhi_intensity'] = scaler_uhi.fit_transform(df_eng[['urban_heat_idx']])
    print("✓ heat_uhi_intensity: Normalized urban heat island index [0-1]")

# 1.3 Canopy Deficit (gap between goal and actual)
if 'lashade_tc_gap' in df_eng.columns:
    # Normalize canopy gap
    df_eng['heat_canopy_deficit'] = df_eng['lashade_tc_gap'].fillna(df_eng['lashade_tc_gap'].median())
    scaler_canopy = MinMaxScaler()
    df_eng['heat_canopy_deficit'] = scaler_canopy.fit_transform(df_eng[['heat_canopy_deficit']])
    print("✓ heat_canopy_deficit: Normalized tree canopy gap [0-1]")
elif 'lashade_treecanopy' in df_eng.columns and 'lashade_tc_goal' in df_eng.columns:
    # Calculate gap if not present
    df_eng['heat_canopy_deficit'] = (df_eng['lashade_tc_goal'] - df_eng['lashade_treecanopy']).clip(lower=0)
    scaler_canopy = MinMaxScaler()
    df_eng['heat_canopy_deficit'] = scaler_canopy.fit_transform(df_eng[['heat_canopy_deficit']])
    print("✓ heat_canopy_deficit: Calculated and normalized tree canopy gap [0-1]")

# 1.4 Air Quality Impact (PM2.5)
if 'pm25' in df_eng.columns:
    df_eng['heat_air_quality_impact'] = df_eng['pm25'].fillna(df_eng['pm25'].median())
    scaler_pm25 = MinMaxScaler()
    df_eng['heat_air_quality_impact'] = scaler_pm25.fit_transform(df_eng[['heat_air_quality_impact']])
    print("✓ heat_air_quality_impact: Normalized PM2.5 levels [0-1]")

# 1.5 Vegetation Deficit (inverse of vegetation coverage)
if 'lashade_veg1500' in df_eng.columns:
    df_eng['heat_vegetation_deficit'] = 1 - MinMaxScaler().fit_transform(
        df_eng[['lashade_veg1500']].fillna(df_eng['lashade_veg1500'].median())
    )
    print("✓ heat_vegetation_deficit: Inverse normalized vegetation within 1500m [0-1]")

# 1.6 COMPOSITE: Heat Vulnerability Index
heat_components = [col for col in df_eng.columns if col.startswith('heat_')]
if len(heat_components) >= 3:
    df_eng['HEAT_VULNERABILITY_INDEX'] = df_eng[heat_components].mean(axis=1)
    print(f"\n✓✓ HEAT_VULNERABILITY_INDEX: Composite of {len(heat_components)} heat features")
    print(f"   Components: {heat_components}")
    print(f"   Range: [{df_eng['HEAT_VULNERABILITY_INDEX'].min():.3f}, {df_eng['HEAT_VULNERABILITY_INDEX'].max():.3f}]")

# ============================================================================
# COMPONENT 2: POPULATION IMPACT FEATURES (25% of reward)
# ============================================================================
print("\n" + "=" * 80)
print("COMPONENT 2: POPULATION IMPACT FEATURES")
print("=" * 80)

# 2.1 Population Density Score
pop_cols = ['cva_population', 'lashade_cbg_pop', 'lashade_acs_pop']
available_pop = [col for col in pop_cols if col in df_eng.columns]

if available_pop:
    # Use CVA population as primary, fallback to others
    df_eng['pop_density_score'] = df_eng['cva_population'].fillna(
        df_eng['lashade_cbg_pop'].fillna(df_eng['lashade_acs_pop'].fillna(0))
    )
    scaler_pop = MinMaxScaler()
    df_eng['pop_density_score'] = scaler_pop.fit_transform(df_eng[['pop_density_score']])
    print(f"✓ pop_density_score: Normalized population density [0-1]")

# 2.2 Vulnerable Population Score (children + elderly)
if 'cva_children' in df_eng.columns and 'cva_older_adults' in df_eng.columns:
    df_eng['pop_vulnerable_score'] = (
        df_eng['cva_children'].fillna(0) + df_eng['cva_older_adults'].fillna(0)
    )
    scaler_vuln = MinMaxScaler()
    df_eng['pop_vulnerable_score'] = scaler_vuln.fit_transform(df_eng[['pop_vulnerable_score']])
    print("✓ pop_vulnerable_score: Normalized vulnerable population (children + elderly) [0-1]")
elif 'lashade_child_perc' in df_eng.columns and 'lashade_seniorperc' in df_eng.columns:
    df_eng['pop_vulnerable_score'] = (
        df_eng['lashade_child_perc'].fillna(0) + df_eng['lashade_seniorperc'].fillna(0)
    ) / 2
    scaler_vuln = MinMaxScaler()
    df_eng['pop_vulnerable_score'] = scaler_vuln.fit_transform(df_eng[['pop_vulnerable_score']])
    print("✓ pop_vulnerable_score: Normalized vulnerable population % (children + seniors) [0-1]")

# 2.3 Olympic Venue Proximity Impact
if 'dist_to_venue1' in df_eng.columns:
    # Inverse distance (closer = higher score)
    df_eng['pop_olympic_proximity'] = 1 / (df_eng['dist_to_venue1'] + 0.1)  # Add small constant to avoid div by 0
    scaler_olympic = MinMaxScaler()
    df_eng['pop_olympic_proximity'] = scaler_olympic.fit_transform(df_eng[['pop_olympic_proximity']])
    print("✓ pop_olympic_proximity: Inverse distance to nearest Olympic venue [0-1]")

# 2.4 Transit Access Score (high traffic areas)
transit_dist_cols = ['dist_to_busstop_1', 'dist_to_metrostop_1']
available_transit = [col for col in transit_dist_cols if col in df_eng.columns]

if available_transit:
    # Average inverse distance to transit
    transit_scores = []
    for col in available_transit:
        transit_scores.append(1 / (df_eng[col] + 0.01))

    df_eng['pop_transit_access'] = np.mean(transit_scores, axis=0)
    scaler_transit = MinMaxScaler()
    df_eng['pop_transit_access'] = scaler_transit.fit_transform(df_eng[['pop_transit_access']])
    print(f"✓ pop_transit_access: Normalized transit accessibility (bus + metro) [0-1]")

# 2.5 COMPOSITE: Population Impact Index
pop_components = [col for col in df_eng.columns if col.startswith('pop_')]
if len(pop_components) >= 2:
    df_eng['POPULATION_IMPACT_INDEX'] = df_eng[pop_components].mean(axis=1)
    print(f"\n✓✓ POPULATION_IMPACT_INDEX: Composite of {len(pop_components)} population features")
    print(f"   Components: {pop_components}")
    print(f"   Range: [{df_eng['POPULATION_IMPACT_INDEX'].min():.3f}, {df_eng['POPULATION_IMPACT_INDEX'].max():.3f}]")

# ============================================================================
# COMPONENT 3: ACCESSIBILITY FEATURES (20% of reward)
# ============================================================================
print("\n" + "=" * 80)
print("COMPONENT 3: ACCESSIBILITY FEATURES")
print("=" * 80)

# 3.1 Cooling Infrastructure Gap (distance to AC/cooling centers)
if 'dist_to_ac_1' in df_eng.columns:
    df_eng['access_cooling_gap'] = df_eng['dist_to_ac_1'].fillna(df_eng['dist_to_ac_1'].median())
    scaler_cooling = MinMaxScaler()
    df_eng['access_cooling_gap'] = scaler_cooling.fit_transform(df_eng[['access_cooling_gap']])
    print("✓ access_cooling_gap: Normalized distance to nearest cooling center [0-1]")

# 3.2 Hydration Access Gap
if 'dist_to_hydro_1' in df_eng.columns:
    df_eng['access_hydration_gap'] = df_eng['dist_to_hydro_1'].fillna(df_eng['dist_to_hydro_1'].median())
    scaler_hydro = MinMaxScaler()
    df_eng['access_hydration_gap'] = scaler_hydro.fit_transform(df_eng[['access_hydration_gap']])
    print("✓ access_hydration_gap: Normalized distance to nearest hydration station [0-1]")

# 3.3 Tree Planting Opportunity Score (vacant sites)
vacant_cols = ['dist_to_vacant_park_1', 'dist_to_vacant_street_1']
available_vacant = [col for col in vacant_cols if col in df_eng.columns]

if available_vacant:
    # Inverse distance to vacant sites (closer = more opportunity)
    vacant_scores = []
    for col in available_vacant:
        vacant_scores.append(1 / (df_eng[col] + 0.01))

    df_eng['access_planting_opportunity'] = np.mean(vacant_scores, axis=0)
    scaler_vacant = MinMaxScaler()
    df_eng['access_planting_opportunity'] = scaler_vacant.fit_transform(df_eng[['access_planting_opportunity']])
    print("✓ access_planting_opportunity: Proximity to vacant planting sites [0-1]")

# 3.4 Multi-Modal Transit Access
if 'cva_transit_access' in df_eng.columns:
    df_eng['access_multimodal_transit'] = df_eng['cva_transit_access'].fillna(df_eng['cva_transit_access'].median())
    scaler_transit = MinMaxScaler()
    df_eng['access_multimodal_transit'] = scaler_transit.fit_transform(df_eng[['access_multimodal_transit']])
    print("✓ access_multimodal_transit: CVA transit access score [0-1]")

# 3.5 Combined Infrastructure Access Gap
infra_gap_cols = ['dist_to_ac_1', 'dist_to_hydro_1', 'dist_to_busstop_1']
available_infra = [col for col in infra_gap_cols if col in df_eng.columns]

if len(available_infra) >= 2:
    df_eng['access_infrastructure_gap'] = df_eng[available_infra].mean(axis=1)
    scaler_infra = MinMaxScaler()
    df_eng['access_infrastructure_gap'] = scaler_infra.fit_transform(df_eng[['access_infrastructure_gap']])
    print(f"✓ access_infrastructure_gap: Average distance to {len(available_infra)} infrastructure types [0-1]")

# 3.6 COMPOSITE: Accessibility Index
access_components = [col for col in df_eng.columns if col.startswith('access_')]
if len(access_components) >= 2:
    df_eng['ACCESSIBILITY_INDEX'] = df_eng[access_components].mean(axis=1)
    print(f"\n✓✓ ACCESSIBILITY_INDEX: Composite of {len(access_components)} accessibility features")
    print(f"   Components: {access_components}")
    print(f"   Range: [{df_eng['ACCESSIBILITY_INDEX'].min():.3f}, {df_eng['ACCESSIBILITY_INDEX'].max():.3f}]")

# ============================================================================
# COMPONENT 4: EQUITY FEATURES (15% of reward)
# ============================================================================
print("\n" + "=" * 80)
print("COMPONENT 4: EQUITY FEATURES")
print("=" * 80)

# 4.1 Social Vulnerability Index
if 'cva_sovi_score' in df_eng.columns:
    df_eng['equity_social_vulnerability'] = df_eng['cva_sovi_score'].fillna(df_eng['cva_sovi_score'].median())
    scaler_sovi = MinMaxScaler()
    df_eng['equity_social_vulnerability'] = scaler_sovi.fit_transform(df_eng[['equity_social_vulnerability']])
    print("✓ equity_social_vulnerability: CVA SOVI score [0-1]")

# 4.2 Economic Disadvantage Score
poverty_cols = ['cva_poverty', 'lashade_pctpov']
available_poverty = [col for col in poverty_cols if col in df_eng.columns]

if available_poverty:
    df_eng['equity_economic_disadvantage'] = df_eng[available_poverty[0]].fillna(
        df_eng[available_poverty[0]].median()
    )
    scaler_poverty = MinMaxScaler()
    df_eng['equity_economic_disadvantage'] = scaler_poverty.fit_transform(df_eng[['equity_economic_disadvantage']])
    print(f"✓ equity_economic_disadvantage: Poverty rate [0-1]")

# 4.3 Environmental Justice Score
if 'lashade_ej_disadva' in df_eng.columns:
    # Convert categorical to binary (Yes=1, No/NaN=0)
    df_eng['equity_environmental_justice'] = (df_eng['lashade_ej_disadva'] == 'Yes').astype(float)
    print("✓ equity_environmental_justice: Environmental justice disadvantage (binary) [0-1]")

# 4.4 Health Vulnerability
health_cols = ['cva_no_health_insurance', 'cva_asthma', 'cva_cardiovascular_disease']
available_health = [col for col in health_cols if col in df_eng.columns]

if len(available_health) >= 2:
    df_eng['equity_health_vulnerability'] = df_eng[available_health].mean(axis=1).fillna(0)
    scaler_health = MinMaxScaler()
    df_eng['equity_health_vulnerability'] = scaler_health.fit_transform(df_eng[['equity_health_vulnerability']])
    print(f"✓ equity_health_vulnerability: Average of {len(available_health)} health metrics [0-1]")

# 4.5 Education Gap
if 'cva_no_high_school_diploma' in df_eng.columns:
    df_eng['equity_education_gap'] = df_eng['cva_no_high_school_diploma'].fillna(df_eng['cva_no_high_school_diploma'].median())
    scaler_edu = MinMaxScaler()
    df_eng['equity_education_gap'] = scaler_edu.fit_transform(df_eng[['equity_education_gap']])
    print("✓ equity_education_gap: Population without HS diploma [0-1]")

# 4.6 Housing Burden
if 'cva_rent_burden' in df_eng.columns:
    df_eng['equity_housing_burden'] = df_eng['cva_rent_burden'].fillna(df_eng['cva_rent_burden'].median())
    scaler_rent = MinMaxScaler()
    df_eng['equity_housing_burden'] = scaler_rent.fit_transform(df_eng[['equity_housing_burden']])
    print("✓ equity_housing_burden: Rent burden score [0-1]")

# 4.7 COMPOSITE: Equity Index
equity_components = [col for col in df_eng.columns if col.startswith('equity_')]
if len(equity_components) >= 2:
    df_eng['EQUITY_INDEX'] = df_eng[equity_components].mean(axis=1)
    print(f"\n✓✓ EQUITY_INDEX: Composite of {len(equity_components)} equity features")
    print(f"   Components: {equity_components}")
    print(f"   Range: [{df_eng['EQUITY_INDEX'].min():.3f}, {df_eng['EQUITY_INDEX'].max():.3f}]")

# ============================================================================
# COMPONENT 5: COVERAGE EFFICIENCY FEATURES (10% of reward)
# ============================================================================
print("\n" + "=" * 80)
print("COMPONENT 5: COVERAGE EFFICIENCY FEATURES")
print("=" * 80)

# 5.1 Current Shade Coverage Score (existing infrastructure)
shade_coverage_cols = ['lashade_tot1200', 'lashade_tot1500', 'lashade_tot1800']
available_shade = [col for col in shade_coverage_cols if col in df_eng.columns]

if available_shade:
    # Use mid-range (1500m) as primary
    df_eng['coverage_existing_shade'] = df_eng['lashade_tot1500'].fillna(df_eng['lashade_tot1500'].median())
    # Inverse normalize (lower existing coverage = higher need)
    scaler_shade = MinMaxScaler()
    df_eng['coverage_existing_shade'] = 1 - scaler_shade.fit_transform(df_eng[['coverage_existing_shade']])
    print("✓ coverage_existing_shade: Inverse of existing shade (gaps identified) [0-1]")

# 5.2 Spatial Isolation Score (distance to nearest neighbors)
print("\n  Computing spatial isolation using BallTree...")
coords = np.radians(df_eng[['latitude', 'longitude']].values)
tree = BallTree(coords, metric='haversine')

# Find distance to 3 nearest neighbors (excluding self)
distances, indices = tree.query(coords, k=4)  # k=4 to get 3 neighbors (excluding self)
avg_neighbor_distance = distances[:, 1:].mean(axis=1) * 6371  # Convert to km

df_eng['coverage_spatial_isolation'] = avg_neighbor_distance
scaler_isolation = MinMaxScaler()
df_eng['coverage_spatial_isolation'] = scaler_isolation.fit_transform(df_eng[['coverage_spatial_isolation']])
print(f"✓ coverage_spatial_isolation: Average distance to 3 nearest points [0-1]")
print(f"   Mean neighbor distance: {avg_neighbor_distance.mean():.3f} km")

# 5.3 Grid Coverage Priority (combine isolation with low shade)
if 'coverage_existing_shade' in df_eng.columns and 'coverage_spatial_isolation' in df_eng.columns:
    df_eng['coverage_priority_score'] = (
        0.6 * df_eng['coverage_existing_shade'] +  # 60% weight to shade gap
        0.4 * df_eng['coverage_spatial_isolation']  # 40% weight to spatial distribution
    )
    print("✓ coverage_priority_score: Weighted combination of shade gap + spatial distribution [0-1]")

# 5.4 COMPOSITE: Coverage Efficiency Index
coverage_components = [col for col in df_eng.columns if col.startswith('coverage_')]
if len(coverage_components) >= 2:
    df_eng['COVERAGE_EFFICIENCY_INDEX'] = df_eng[coverage_components].mean(axis=1)
    print(f"\n✓✓ COVERAGE_EFFICIENCY_INDEX: Composite of {len(coverage_components)} coverage features")
    print(f"   Components: {coverage_components}")
    print(f"   Range: [{df_eng['COVERAGE_EFFICIENCY_INDEX'].min():.3f}, {df_eng['COVERAGE_EFFICIENCY_INDEX'].max():.3f}]")

# ============================================================================
# INTERACTION FEATURES (Cross-component synergies)
# ============================================================================
print("\n" + "=" * 80)
print("INTERACTION FEATURES (Cross-Component Synergies)")
print("=" * 80)

# 6.1 Heat × Population (high temp areas with high population)
if 'HEAT_VULNERABILITY_INDEX' in df_eng.columns and 'POPULATION_IMPACT_INDEX' in df_eng.columns:
    df_eng['interact_heat_population'] = (
        df_eng['HEAT_VULNERABILITY_INDEX'] * df_eng['POPULATION_IMPACT_INDEX']
    )
    print("✓ interact_heat_population: Heat vulnerability × Population impact")

# 6.2 Heat × Equity (environmental justice priority)
if 'HEAT_VULNERABILITY_INDEX' in df_eng.columns and 'EQUITY_INDEX' in df_eng.columns:
    df_eng['interact_heat_equity'] = (
        df_eng['HEAT_VULNERABILITY_INDEX'] * df_eng['EQUITY_INDEX']
    )
    print("✓ interact_heat_equity: Heat vulnerability × Equity (EJ priority)")

# 6.3 Population × Accessibility (underserved high-traffic areas)
if 'POPULATION_IMPACT_INDEX' in df_eng.columns and 'ACCESSIBILITY_INDEX' in df_eng.columns:
    df_eng['interact_population_access'] = (
        df_eng['POPULATION_IMPACT_INDEX'] * df_eng['ACCESSIBILITY_INDEX']
    )
    print("✓ interact_population_access: Population × Accessibility gap")

# 6.4 Equity × Accessibility (equity in access)
if 'EQUITY_INDEX' in df_eng.columns and 'ACCESSIBILITY_INDEX' in df_eng.columns:
    df_eng['interact_equity_access'] = (
        df_eng['EQUITY_INDEX'] * df_eng['ACCESSIBILITY_INDEX']
    )
    print("✓ interact_equity_access: Equity × Accessibility (access equity)")

# ============================================================================
# FINAL COMPOSITE: REWARD FUNCTION SCORE
# ============================================================================
print("\n" + "=" * 80)
print("FINAL COMPOSITE: OVERALL REWARD POTENTIAL SCORE")
print("=" * 80)

# Weights from reward function specification
REWARD_WEIGHTS = {
    'heat_vulnerability': 0.30,
    'population_impact': 0.25,
    'accessibility': 0.20,
    'equity': 0.15,
    'coverage_efficiency': 0.10
}

composite_indices = {
    'HEAT_VULNERABILITY_INDEX': REWARD_WEIGHTS['heat_vulnerability'],
    'POPULATION_IMPACT_INDEX': REWARD_WEIGHTS['population_impact'],
    'ACCESSIBILITY_INDEX': REWARD_WEIGHTS['accessibility'],
    'EQUITY_INDEX': REWARD_WEIGHTS['equity'],
    'COVERAGE_EFFICIENCY_INDEX': REWARD_WEIGHTS['coverage_efficiency']
}

# Calculate weighted reward potential
available_indices = {k: v for k, v in composite_indices.items() if k in df_eng.columns}

if len(available_indices) >= 4:
    # Normalize weights to sum to 1.0
    total_weight = sum(available_indices.values())
    normalized_weights = {k: v/total_weight for k, v in available_indices.items()}

    df_eng['REWARD_POTENTIAL_SCORE'] = sum(
        df_eng[idx] * weight for idx, weight in normalized_weights.items()
    )

    print(f"\n✓✓✓ REWARD_POTENTIAL_SCORE: Weighted composite of {len(available_indices)} indices")
    print(f"\n   Weights applied:")
    for idx, weight in normalized_weights.items():
        print(f"      {idx:30s} {weight:.3f}")

    print(f"\n   Score Statistics:")
    print(f"      Mean:   {df_eng['REWARD_POTENTIAL_SCORE'].mean():.3f}")
    print(f"      Std:    {df_eng['REWARD_POTENTIAL_SCORE'].std():.3f}")
    print(f"      Min:    {df_eng['REWARD_POTENTIAL_SCORE'].min():.3f}")
    print(f"      Max:    {df_eng['REWARD_POTENTIAL_SCORE'].max():.3f}")
    print(f"      Range:  {df_eng['REWARD_POTENTIAL_SCORE'].max() - df_eng['REWARD_POTENTIAL_SCORE'].min():.3f}")

# ============================================================================
# PRIORITY RANKINGS
# ============================================================================
print("\n" + "=" * 80)
print("PRIORITY RANKINGS FOR SHADE PLACEMENT")
print("=" * 80)

if 'REWARD_POTENTIAL_SCORE' in df_eng.columns:
    # Rank all locations by reward potential
    df_eng['priority_rank'] = df_eng['REWARD_POTENTIAL_SCORE'].rank(ascending=False, method='min').astype(int)

    # Create priority tiers
    df_eng['priority_tier'] = pd.cut(
        df_eng['REWARD_POTENTIAL_SCORE'],
        bins=5,
        labels=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High']
    )

    print("\nPriority Tier Distribution:")
    tier_counts = df_eng['priority_tier'].value_counts().sort_index()
    for tier, count in tier_counts.items():
        pct = count / len(df_eng) * 100
        print(f"   {tier:15s}: {count:4d} locations ({pct:5.1f}%)")

    # Top 50 priority locations (matching 50 shade structures constraint)
    top_50 = df_eng.nlargest(50, 'REWARD_POTENTIAL_SCORE')

    print(f"\n✓ Top 50 Priority Locations Identified (for 50 shade structures)")
    print(f"   Mean Reward Potential: {top_50['REWARD_POTENTIAL_SCORE'].mean():.3f}")
    print(f"   Geographic Spread:")
    print(f"      Lat range: [{top_50['latitude'].min():.4f}, {top_50['latitude'].max():.4f}]")
    print(f"      Lon range: [{top_50['longitude'].min():.4f}, {top_50['longitude'].max():.4f}]")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 80)

# Count new features by category
new_features = {
    'Heat Vulnerability': [c for c in df_eng.columns if c.startswith('heat_')],
    'Population Impact': [c for c in df_eng.columns if c.startswith('pop_')],
    'Accessibility': [c for c in df_eng.columns if c.startswith('access_')],
    'Equity': [c for c in df_eng.columns if c.startswith('equity_')],
    'Coverage Efficiency': [c for c in df_eng.columns if c.startswith('coverage_')],
    'Interactions': [c for c in df_eng.columns if c.startswith('interact_')],
    'Composite Indices': [c for c in df_eng.columns if c.isupper() and '_INDEX' in c or 'SCORE' in c]
}

print(f"\nOriginal Features: {df.shape[1]}")
print(f"Engineered Features: {df_eng.shape[1] - df.shape[1]}")
print(f"Total Features: {df_eng.shape[1]}")

print("\nBreakdown by Category:")
total_new = 0
for category, features in new_features.items():
    count = len(features)
    total_new += count
    print(f"   {category:25s}: {count:2d} features")
    if count > 0 and count <= 5:
        for feat in features:
            print(f"      - {feat}")

print(f"\n   {'TOTAL NEW FEATURES':25s}: {total_new:2d}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING FEATURE-ENGINEERED DATASET")
print("=" * 80)

# Save full dataset
output_file = 'shade_optimization_data_usc_features.csv'
df_eng.to_csv(output_file, index=False)
print(f"\n✓ Saved: {output_file}")
print(f"   Shape: {df_eng.shape[0]} rows × {df_eng.shape[1]} columns")

# Save top 50 priority locations
if 'priority_rank' in df_eng.columns:
    top_50_file = 'shade_optimization_top50_priority.csv'
    top_50_df = df_eng[df_eng['priority_rank'] <= 50].sort_values('priority_rank')
    top_50_df.to_csv(top_50_file, index=False)
    print(f"\n✓ Saved: {top_50_file}")
    print(f"   Top 50 highest priority locations for shade placement")

# Save feature documentation
doc_file = 'eda_outputs_usc/feature_engineering_documentation.txt'
with open(doc_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("FEATURE ENGINEERING DOCUMENTATION\n")
    f.write("USC Shade Optimization Project\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Total Features: {df_eng.shape[1]}\n")
    f.write(f"Original Features: {df.shape[1]}\n")
    f.write(f"Engineered Features: {df_eng.shape[1] - df.shape[1]}\n\n")

    f.write("ENGINEERED FEATURES BY CATEGORY:\n")
    f.write("-" * 80 + "\n\n")

    for category, features in new_features.items():
        if features:
            f.write(f"{category.upper()} ({len(features)} features):\n")
            for feat in features:
                f.write(f"   - {feat}\n")
            f.write("\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("COMPOSITE INDICES (For Reward Function):\n")
    f.write("-" * 80 + "\n\n")

    if 'HEAT_VULNERABILITY_INDEX' in df_eng.columns:
        f.write("HEAT_VULNERABILITY_INDEX (30% weight):\n")
        for feat in [c for c in df_eng.columns if c.startswith('heat_') and not c.isupper()]:
            f.write(f"   - {feat}\n")
        f.write("\n")

    if 'POPULATION_IMPACT_INDEX' in df_eng.columns:
        f.write("POPULATION_IMPACT_INDEX (25% weight):\n")
        for feat in [c for c in df_eng.columns if c.startswith('pop_')]:
            f.write(f"   - {feat}\n")
        f.write("\n")

    if 'ACCESSIBILITY_INDEX' in df_eng.columns:
        f.write("ACCESSIBILITY_INDEX (20% weight):\n")
        for feat in [c for c in df_eng.columns if c.startswith('access_')]:
            f.write(f"   - {feat}\n")
        f.write("\n")

    if 'EQUITY_INDEX' in df_eng.columns:
        f.write("EQUITY_INDEX (15% weight):\n")
        for feat in [c for c in df_eng.columns if c.startswith('equity_')]:
            f.write(f"   - {feat}\n")
        f.write("\n")

    if 'COVERAGE_EFFICIENCY_INDEX' in df_eng.columns:
        f.write("COVERAGE_EFFICIENCY_INDEX (10% weight):\n")
        for feat in [c for c in df_eng.columns if c.startswith('coverage_')]:
            f.write(f"   - {feat}\n")
        f.write("\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("REWARD POTENTIAL SCORE:\n")
    f.write("-" * 80 + "\n\n")

    if 'REWARD_POTENTIAL_SCORE' in df_eng.columns:
        f.write("Weighted composite of all 5 component indices\n")
        f.write(f"Range: [{df_eng['REWARD_POTENTIAL_SCORE'].min():.3f}, {df_eng['REWARD_POTENTIAL_SCORE'].max():.3f}]\n")
        f.write(f"Mean: {df_eng['REWARD_POTENTIAL_SCORE'].mean():.3f}\n")
        f.write(f"Std: {df_eng['REWARD_POTENTIAL_SCORE'].std():.3f}\n\n")

        f.write("Use this score to:\n")
        f.write("   1. Initialize Q-values in RL agent\n")
        f.write("   2. Guide exploration during training\n")
        f.write("   3. Compare with greedy baseline\n")
        f.write("   4. Validate reward function components\n")

print(f"\n✓ Saved: {doc_file}")

print("\n" + "=" * 80)
print("FEATURE ENGINEERING COMPLETE!")
print("=" * 80)
print("\nNext Steps:")
print("   1. Use shade_optimization_data_usc_features.csv for RL training")
print("   2. Composite indices map directly to reward function components")
print("   3. REWARD_POTENTIAL_SCORE can initialize Q-values or guide exploration")
print("   4. Top 50 priority locations available for baseline comparison")
print("=" * 80)
