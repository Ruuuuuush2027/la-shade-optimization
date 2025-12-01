"""
Simple Feature Engineering for USC Shade Optimization
Creates individual engineered features and interactions only.
No composite indices or reward function components.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import BallTree
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FEATURE ENGINEERING - INDIVIDUAL FEATURES ONLY")
print("=" * 80)

# Load cleaned data
df = pd.read_csv('shade_optimization_data_usc_cleaned.csv')
print(f"\nLoaded Dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# Create copy for feature engineering
df_eng = df.copy()

# ============================================================================
# HEAT-RELATED FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("HEAT-RELATED FEATURES")
print("=" * 80)

# Normalized temperature severity
scaler = MinMaxScaler()
df_eng['temp_severity_norm'] = scaler.fit_transform(df_eng[['land_surface_temp_c']])
print("✓ temp_severity_norm: Normalized land surface temperature [0-1]")

# Normalized UHI
if 'urban_heat_idx' in df_eng.columns:
    df_eng['uhi_norm'] = scaler.fit_transform(df_eng[['urban_heat_idx']])
    print("✓ uhi_norm: Normalized urban heat island index [0-1]")

# Canopy gap normalized
if 'lashade_tc_gap' in df_eng.columns:
    df_eng['canopy_gap_norm'] = df_eng['lashade_tc_gap'].fillna(df_eng['lashade_tc_gap'].median())
    df_eng['canopy_gap_norm'] = scaler.fit_transform(df_eng[['canopy_gap_norm']])
    print("✓ canopy_gap_norm: Normalized tree canopy gap [0-1]")

# Canopy percentage of goal
if 'lashade_treecanopy' in df_eng.columns and 'lashade_tc_goal' in df_eng.columns:
    df_eng['canopy_pct_of_goal'] = (
        df_eng['lashade_treecanopy'] / (df_eng['lashade_tc_goal'] + 1e-6)
    ).clip(0, 2)  # Clip at 2x goal
    print("✓ canopy_pct_of_goal: Tree canopy as % of goal [0-2]")

# PM2.5 normalized
if 'pm25' in df_eng.columns:
    df_eng['pm25_norm'] = df_eng['pm25'].fillna(df_eng['pm25'].median())
    df_eng['pm25_norm'] = scaler.fit_transform(df_eng[['pm25_norm']])
    print("✓ pm25_norm: Normalized PM2.5 air quality [0-1]")

# Vegetation deficit (inverse of vegetation coverage)
if 'lashade_veg1500' in df_eng.columns:
    veg_norm = scaler.fit_transform(
        df_eng[['lashade_veg1500']].fillna(df_eng['lashade_veg1500'].median())
    )
    df_eng['vegetation_deficit'] = 1 - veg_norm.flatten()
    print("✓ vegetation_deficit: Inverse of vegetation within 1500m [0-1]")

# Combined shade metric (average of multiple radii)
shade_cols = ['lashade_tot1200', 'lashade_tot1500', 'lashade_tot1800']
available_shade = [col for col in shade_cols if col in df_eng.columns]
if len(available_shade) >= 2:
    df_eng['avg_shade_coverage'] = df_eng[available_shade].mean(axis=1)
    print(f"✓ avg_shade_coverage: Average shade across {len(available_shade)} radii")

# ============================================================================
# POPULATION-RELATED FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("POPULATION-RELATED FEATURES")
print("=" * 80)

# Population density (normalized)
pop_cols = ['cva_population', 'lashade_cbg_pop', 'lashade_acs_pop']
available_pop = [col for col in pop_cols if col in df_eng.columns]
if available_pop:
    df_eng['population_norm'] = df_eng['cva_population'].fillna(
        df_eng.get('lashade_cbg_pop', df_eng.get('lashade_acs_pop', 0))
    )
    df_eng['population_norm'] = scaler.fit_transform(df_eng[['population_norm']])
    print("✓ population_norm: Normalized population density [0-1]")

# Vulnerable population (children + elderly)
if 'cva_children' in df_eng.columns and 'cva_older_adults' in df_eng.columns:
    df_eng['vulnerable_population'] = (
        df_eng['cva_children'].fillna(0) + df_eng['cva_older_adults'].fillna(0)
    )
    df_eng['vulnerable_pop_norm'] = scaler.fit_transform(df_eng[['vulnerable_population']])
    print("✓ vulnerable_pop_norm: Normalized vulnerable population [0-1]")
elif 'lashade_child_perc' in df_eng.columns and 'lashade_seniorperc' in df_eng.columns:
    df_eng['vulnerable_pop_pct'] = (
        df_eng['lashade_child_perc'].fillna(0) + df_eng['lashade_seniorperc'].fillna(0)
    ) / 2
    print("✓ vulnerable_pop_pct: Average of child % and senior % [0-1]")

# Olympic venue proximity (inverse distance)
if 'dist_to_venue1' in df_eng.columns:
    df_eng['olympic_proximity'] = 1 / (df_eng['dist_to_venue1'] + 0.1)
    df_eng['olympic_proximity_norm'] = scaler.fit_transform(df_eng[['olympic_proximity']])
    print("✓ olympic_proximity_norm: Inverse distance to Olympic venue [0-1]")

# ============================================================================
# ACCESSIBILITY-RELATED FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("ACCESSIBILITY-RELATED FEATURES")
print("=" * 80)

# Distance to cooling centers (normalized)
if 'dist_to_ac_1' in df_eng.columns:
    df_eng['cooling_distance_norm'] = scaler.fit_transform(df_eng[['dist_to_ac_1']])
    print("✓ cooling_distance_norm: Distance to nearest cooling center [0-1]")

# Distance to hydration (normalized)
if 'dist_to_hydro_1' in df_eng.columns:
    df_eng['hydration_distance_norm'] = scaler.fit_transform(df_eng[['dist_to_hydro_1']])
    print("✓ hydration_distance_norm: Distance to nearest hydration station [0-1]")

# Average transit access (bus + metro)
transit_cols = ['dist_to_busstop_1', 'dist_to_metrostop_1']
available_transit = [col for col in transit_cols if col in df_eng.columns]
if len(available_transit) >= 2:
    df_eng['avg_transit_distance'] = df_eng[available_transit].mean(axis=1)
    df_eng['avg_transit_distance_norm'] = scaler.fit_transform(df_eng[['avg_transit_distance']])
    print("✓ avg_transit_distance_norm: Average distance to bus + metro [0-1]")

# Inverse transit access (closer = higher score)
if 'avg_transit_distance' in df_eng.columns:
    df_eng['transit_access_score'] = 1 / (df_eng['avg_transit_distance'] + 0.01)
    df_eng['transit_access_norm'] = scaler.fit_transform(df_eng[['transit_access_score']])
    print("✓ transit_access_norm: Inverse transit distance (accessibility) [0-1]")

# Planting opportunity (proximity to vacant sites)
vacant_cols = ['dist_to_vacant_park_1', 'dist_to_vacant_street_1']
available_vacant = [col for col in vacant_cols if col in df_eng.columns]
if len(available_vacant) >= 2:
    df_eng['avg_vacant_distance'] = df_eng[available_vacant].mean(axis=1)
    df_eng['planting_opportunity'] = 1 / (df_eng['avg_vacant_distance'] + 0.01)
    df_eng['planting_opportunity_norm'] = scaler.fit_transform(df_eng[['planting_opportunity']])
    print("✓ planting_opportunity_norm: Proximity to vacant planting sites [0-1]")

# Combined infrastructure access
infra_dist_cols = ['dist_to_ac_1', 'dist_to_hydro_1', 'dist_to_busstop_1']
available_infra = [col for col in infra_dist_cols if col in df_eng.columns]
if len(available_infra) >= 2:
    df_eng['avg_infrastructure_distance'] = df_eng[available_infra].mean(axis=1)
    print(f"✓ avg_infrastructure_distance: Average distance to {len(available_infra)} infrastructure types")

# CVA transit access (if available)
if 'cva_transit_access' in df_eng.columns:
    df_eng['cva_transit_norm'] = scaler.fit_transform(
        df_eng[['cva_transit_access']].fillna(df_eng['cva_transit_access'].median())
    )
    print("✓ cva_transit_norm: Normalized CVA transit access [0-1]")

# ============================================================================
# EQUITY-RELATED FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("EQUITY-RELATED FEATURES")
print("=" * 80)

# Social vulnerability (SOVI)
if 'cva_sovi_score' in df_eng.columns:
    df_eng['sovi_norm'] = scaler.fit_transform(
        df_eng[['cva_sovi_score']].fillna(df_eng['cva_sovi_score'].median())
    )
    print("✓ sovi_norm: Normalized social vulnerability index [0-1]")

# Poverty rate
poverty_cols = ['cva_poverty', 'lashade_pctpov']
available_poverty = [col for col in poverty_cols if col in df_eng.columns]
if available_poverty:
    df_eng['poverty_norm'] = df_eng[available_poverty[0]].fillna(
        df_eng[available_poverty[0]].median()
    )
    df_eng['poverty_norm'] = scaler.fit_transform(df_eng[['poverty_norm']])
    print("✓ poverty_norm: Normalized poverty rate [0-1]")

# Environmental justice (binary)
if 'lashade_ej_disadva' in df_eng.columns:
    df_eng['env_justice_binary'] = (df_eng['lashade_ej_disadva'] == 'Yes').astype(float)
    print("✓ env_justice_binary: Environmental justice disadvantage flag [0 or 1]")

# Health vulnerability (composite of health metrics)
health_cols = ['cva_no_health_insurance', 'cva_asthma', 'cva_cardiovascular_disease']
available_health = [col for col in health_cols if col in df_eng.columns]
if len(available_health) >= 2:
    df_eng['avg_health_vulnerability'] = df_eng[available_health].mean(axis=1).fillna(0)
    df_eng['health_vulnerability_norm'] = scaler.fit_transform(df_eng[['avg_health_vulnerability']])
    print(f"✓ health_vulnerability_norm: Average of {len(available_health)} health metrics [0-1]")

# Education gap
if 'cva_no_high_school_diploma' in df_eng.columns:
    df_eng['education_gap_norm'] = scaler.fit_transform(
        df_eng[['cva_no_high_school_diploma']].fillna(df_eng['cva_no_high_school_diploma'].median())
    )
    print("✓ education_gap_norm: Normalized education gap [0-1]")

# Housing burden
if 'cva_rent_burden' in df_eng.columns:
    df_eng['rent_burden_norm'] = scaler.fit_transform(
        df_eng[['cva_rent_burden']].fillna(df_eng['cva_rent_burden'].median())
    )
    print("✓ rent_burden_norm: Normalized rent burden [0-1]")

# Linguistic isolation
if 'cva_limited_english' in df_eng.columns:
    df_eng['limited_english_norm'] = scaler.fit_transform(
        df_eng[['cva_limited_english']].fillna(df_eng['cva_limited_english'].median())
    )
    print("✓ limited_english_norm: Normalized limited English proficiency [0-1]")

# ============================================================================
# SPATIAL/COVERAGE FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("SPATIAL/COVERAGE FEATURES")
print("=" * 80)

# Existing shade (inverse - higher means less shade)
if 'lashade_tot1500' in df_eng.columns:
    shade_norm = scaler.fit_transform(
        df_eng[['lashade_tot1500']].fillna(df_eng['lashade_tot1500'].median())
    )
    df_eng['shade_gap'] = 1 - shade_norm.flatten()
    print("✓ shade_gap: Inverse of existing shade coverage [0-1]")

# Spatial isolation (distance to neighbors)
print("\n  Computing spatial isolation using BallTree...")
coords = np.radians(df_eng[['latitude', 'longitude']].values)
tree = BallTree(coords, metric='haversine')

# Find distance to k nearest neighbors
k_neighbors = 3
distances, indices = tree.query(coords, k=k_neighbors + 1)  # +1 to exclude self
avg_neighbor_distance = distances[:, 1:].mean(axis=1) * 6371  # Convert to km

df_eng['spatial_isolation_km'] = avg_neighbor_distance
df_eng['spatial_isolation_norm'] = scaler.fit_transform(df_eng[['spatial_isolation_km']])
print(f"✓ spatial_isolation_km: Average distance to {k_neighbors} nearest neighbors [km]")
print(f"✓ spatial_isolation_norm: Normalized spatial isolation [0-1]")
print(f"   Mean neighbor distance: {avg_neighbor_distance.mean():.3f} km")

# ============================================================================
# INTERACTION FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("INTERACTION FEATURES")
print("=" * 80)

# Heat × Population (hot areas with high population)
if 'temp_severity_norm' in df_eng.columns and 'population_norm' in df_eng.columns:
    df_eng['heat_x_population'] = df_eng['temp_severity_norm'] * df_eng['population_norm']
    print("✓ heat_x_population: Temperature × Population interaction")

# Heat × Poverty (hot areas with high poverty - EJ concern)
if 'temp_severity_norm' in df_eng.columns and 'poverty_norm' in df_eng.columns:
    df_eng['heat_x_poverty'] = df_eng['temp_severity_norm'] * df_eng['poverty_norm']
    print("✓ heat_x_poverty: Temperature × Poverty (environmental justice)")

# Heat × Vulnerable Population
if 'temp_severity_norm' in df_eng.columns and 'vulnerable_pop_norm' in df_eng.columns:
    df_eng['heat_x_vulnerable'] = df_eng['temp_severity_norm'] * df_eng['vulnerable_pop_norm']
    print("✓ heat_x_vulnerable: Temperature × Vulnerable population")

# Heat × Social Vulnerability
if 'temp_severity_norm' in df_eng.columns and 'sovi_norm' in df_eng.columns:
    df_eng['heat_x_sovi'] = df_eng['temp_severity_norm'] * df_eng['sovi_norm']
    print("✓ heat_x_sovi: Temperature × Social vulnerability")

# Population × Transit Access (high population with poor transit)
if 'population_norm' in df_eng.columns and 'avg_transit_distance_norm' in df_eng.columns:
    df_eng['population_x_transit_gap'] = df_eng['population_norm'] * df_eng['avg_transit_distance_norm']
    print("✓ population_x_transit_gap: Population × Transit distance gap")

# Poverty × Infrastructure Access (poor areas with poor infrastructure)
if 'poverty_norm' in df_eng.columns and 'cooling_distance_norm' in df_eng.columns:
    df_eng['poverty_x_cooling_gap'] = df_eng['poverty_norm'] * df_eng['cooling_distance_norm']
    print("✓ poverty_x_cooling_gap: Poverty × Cooling infrastructure gap")

# Vegetation Deficit × UHI (areas lacking vegetation in hot zones)
if 'vegetation_deficit' in df_eng.columns and 'uhi_norm' in df_eng.columns:
    df_eng['veg_deficit_x_uhi'] = df_eng['vegetation_deficit'] * df_eng['uhi_norm']
    print("✓ veg_deficit_x_uhi: Vegetation deficit × Urban heat island")

# Canopy Gap × Air Quality (low canopy in polluted areas)
if 'canopy_gap_norm' in df_eng.columns and 'pm25_norm' in df_eng.columns:
    df_eng['canopy_gap_x_pm25'] = df_eng['canopy_gap_norm'] * df_eng['pm25_norm']
    print("✓ canopy_gap_x_pm25: Canopy gap × PM2.5 pollution")

# Olympic Proximity × Population (high traffic event areas)
if 'olympic_proximity_norm' in df_eng.columns and 'population_norm' in df_eng.columns:
    df_eng['olympic_x_population'] = df_eng['olympic_proximity_norm'] * df_eng['population_norm']
    print("✓ olympic_x_population: Olympic proximity × Population")

# Shade Gap × Spatial Isolation (underserved isolated areas)
if 'shade_gap' in df_eng.columns and 'spatial_isolation_norm' in df_eng.columns:
    df_eng['shade_gap_x_isolation'] = df_eng['shade_gap'] * df_eng['spatial_isolation_norm']
    print("✓ shade_gap_x_isolation: Shade gap × Spatial isolation")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 80)

# Count new features
new_features = [col for col in df_eng.columns if col not in df.columns]

print(f"\nOriginal Features: {df.shape[1]}")
print(f"Engineered Features: {len(new_features)}")
print(f"Total Features: {df_eng.shape[1]}")

# Categorize new features
categories = {
    'Heat-Related': [f for f in new_features if any(x in f for x in ['temp', 'uhi', 'canopy', 'pm25', 'veg'])],
    'Population-Related': [f for f in new_features if any(x in f for x in ['population', 'vulnerable', 'olympic'])],
    'Accessibility-Related': [f for f in new_features if any(x in f for x in ['cooling', 'hydration', 'transit', 'planting', 'infrastructure'])],
    'Equity-Related': [f for f in new_features if any(x in f for x in ['sovi', 'poverty', 'justice', 'health', 'education', 'rent', 'english'])],
    'Spatial-Related': [f for f in new_features if any(x in f for x in ['shade_gap', 'spatial', 'isolation'])],
    'Interactions': [f for f in new_features if '_x_' in f]
}

print("\nBreakdown by Category:")
for category, features in categories.items():
    if features:
        print(f"\n{category} ({len(features)} features):")
        for feat in features:
            print(f"   - {feat}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save feature-engineered dataset
output_file = 'shade_optimization_data_usc_simple_features.csv'
df_eng.to_csv(output_file, index=False)
print(f"\n✓ Saved: {output_file}")
print(f"   Shape: {df_eng.shape[0]} rows × {df_eng.shape[1]} columns")

# Save feature list
feature_list_file = 'eda_outputs_usc/simple_feature_list.txt'
with open(feature_list_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("ENGINEERED FEATURES LIST\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Total Features: {df_eng.shape[1]}\n")
    f.write(f"Original Features: {df.shape[1]}\n")
    f.write(f"Engineered Features: {len(new_features)}\n\n")

    for category, features in categories.items():
        if features:
            f.write(f"{category.upper()} ({len(features)} features):\n")
            f.write("-" * 80 + "\n")
            for feat in features:
                f.write(f"   {feat}\n")
            f.write("\n")

print(f"✓ Saved: {feature_list_file}")

print("\n" + "=" * 80)
print("FEATURE ENGINEERING COMPLETE!")
print("=" * 80)
print(f"\nGenerated {len(new_features)} new features")
print(f"Dataset ready: {output_file}")
print("=" * 80)
