"""
Comprehensive Exploratory Data Analysis Script
==============================================
This script performs thorough EDA on the environmental/urban dataset covering:
1. Data Overview and Summary Statistics
2. Missing Value Analysis  
3. Distribution Analysis
4. Correlation Analysis
5. Outlier Detection
6. Pattern Recognition
7. Geographic Analysis
8. Feature Relationships
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the data
print("="*80)
print("LOADING DATA")
print("="*80)

# You'll need to update this path to your actual CSV file location
# For now, I'll create sample data from your provided text
data_text = """latitude,longitude,dist_to_ac_1,dist_to_ac_3,dist_to_hydro_1,dist_to_hydro_3,install_year_hydro_3,dist_to_busline_1,dist_to_busline_3,dist_to_busstop_1,dist_to_busstop_3,dist_to_metroline_1,dist_to_metroline_3,dist_to_metrostop_1,dist_to_metrostop_3,lashade_ua_pop,lashade_cbg_pop,lashade_acs_pop,lashade_biome,lashade_tc_goal,lashade_treecanopy,lashade_tc_gap,lashade_pctpoc,lashade_pctpov,lashade_unemplrate,lashade_dep_ratio,lashade_dep_perc,lashade_linguistic,lashade_health_nor,lashade_temp_diff,lashade_tes,lashade_holc_grade,lashade_child_perc,lashade_seniorperc,lashade_ej_disadva,lashade_rank,lashade_rankgrpsz,lashade_bld1200,lashade_veg1200,lashade_bld1500,lashade_veg1500,lashade_bld1800,lashade_veg1800,lashade_tot1200,lashade_tot1500,lashade_tot1800,dist_to_venue1,closest_venue_sport,dist_to_venue3,cva_population,cva_children,cva_older_adults,cva_older_adults_living_alone,cva_limited_english,cva_no_high_school_diploma,cva_female,cva_female_householder,cva_disability,cva_no_health_insurance,cva_living_in_group_quarters,cva_mobile_homes,cva_rent_burden,cva_renters,cva_median_income,cva_poverty,cva_households_without_vehicle_acce,cva_outdoor_workers,cva_unemployed,cva_foreign_born,cva_no_internet_subscription,cva_voter_turnout_rate,cva_sovi_score,cva_asthma,cva_cardiovascular_disease,cva_transit_access,pm25,pm25_percentile,tree_percent_w,urban_heat_idx,urban_heat_idx_percentile,dist_to_vacant_park_1,dist_to_vacant_park_3,dist_to_vacant_street_1,dist_to_vacant_street_3"""

# Save sample data to CSV for loading
import io
from io import StringIO

# Use StringIO to create DataFrame from the text data
df = pd.read_csv(StringIO(data_text))

# To use with your actual file, uncomment and modify:
df = pd.read_csv('shade_optimization_data.csv')

print(f"Data loaded successfully!")
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print()

# ================================================================================
# 1. DATA OVERVIEW
# ================================================================================
print("="*80)
print("1. DATA OVERVIEW")
print("="*80)

# Basic information
print("\n📊 Dataset Information:")
print(f"• Number of observations: {df.shape[0]:,}")
print(f"• Number of features: {df.shape[1]:,}")
print(f"• Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Data types
print("\n📋 Data Types Distribution:")
dtype_counts = df.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"• {dtype}: {count} columns ({count/len(df.columns)*100:.1f}%)")

# Identify different feature categories based on column names
print("\n🏷️ Feature Categories Identified:")
distance_cols = [col for col in df.columns if 'dist_' in col]
lashade_cols = [col for col in df.columns if 'lashade_' in col]
cva_cols = [col for col in df.columns if 'cva_' in col]
geo_cols = ['latitude', 'longitude']
environmental_cols = ['pm25', 'pm25_percentile', 'tree_percent_w', 'urban_heat_idx', 'urban_heat_idx_percentile']
venue_cols = [col for col in df.columns if 'venue' in col]

print(f"• Geographic features: {len(geo_cols)}")
print(f"• Distance features: {len(distance_cols)}")
print(f"• LA Shade features: {len(lashade_cols)}")
print(f"• CVA (vulnerability) features: {len(cva_cols)}")
print(f"• Environmental features: {len(environmental_cols)}")
print(f"• Venue features: {len(venue_cols)}")

# ================================================================================
# 2. MISSING VALUE ANALYSIS
# ================================================================================
print("\n" + "="*80)
print("2. MISSING VALUE ANALYSIS")
print("="*80)

# Calculate missing values
missing_df = pd.DataFrame({
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)

if len(missing_df) > 0:
    print("\n⚠️ Columns with Missing Values:")
    print(missing_df.to_string())
    
    # Visualize missing values pattern
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot of missing values
    if len(missing_df) <= 20:
        missing_df.plot(kind='bar', y='Missing_Percentage', ax=axes[0], color='coral')
        axes[0].set_title('Missing Values by Column (%)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Columns')
        axes[0].set_ylabel('Missing %')
        axes[0].tick_params(axis='x', rotation=45)
    else:
        top_missing = missing_df.head(20)
        top_missing.plot(kind='bar', y='Missing_Percentage', ax=axes[0], color='coral')
        axes[0].set_title('Top 20 Columns with Missing Values (%)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Columns')
        axes[0].set_ylabel('Missing %')
        axes[0].tick_params(axis='x', rotation=45)
    
    # Heatmap of missing values
    import missingno as msno
    msno.matrix(df, ax=axes[1], sparkline=False)
    axes[1].set_title('Missing Value Pattern Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('missing_values_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("\n✅ No missing values found in the dataset!")

# ================================================================================
# 3. SUMMARY STATISTICS
# ================================================================================
print("\n" + "="*80)
print("3. SUMMARY STATISTICS")
print("="*80)

# Get numeric columns only
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Basic statistics for numeric columns
print("\n📈 Statistical Summary (sample of key features):")
key_features = ['latitude', 'longitude', 'pm25', 'urban_heat_idx', 'tree_percent_w']
available_features = [f for f in key_features if f in df.columns]

if available_features:
    summary_stats = df[available_features].describe()
    print(summary_stats.round(2))

# Additional statistics
print("\n📊 Additional Statistics (for numeric columns):")
additional_stats = pd.DataFrame({
    'Variance': df[numeric_cols].var(),
    'Skewness': df[numeric_cols].skew(),
    'Kurtosis': df[numeric_cols].kurtosis(),
    'Coefficient_of_Variation': (df[numeric_cols].std() / df[numeric_cols].mean()) * 100
})

# Show top 10 most variable features
print("\n🔝 Top 10 Most Variable Features (by CV):")
top_variable = additional_stats.nlargest(10, 'Coefficient_of_Variation')
print(top_variable[['Coefficient_of_Variation']].round(2))

# ================================================================================
# 4. DISTRIBUTION ANALYSIS
# ================================================================================
print("\n" + "="*80)
print("4. DISTRIBUTION ANALYSIS")
print("="*80)

# Select key features for distribution analysis
dist_features = ['pm25', 'urban_heat_idx', 'tree_percent_w', 'dist_to_busline_1', 
                 'dist_to_metrostop_1', 'dist_to_venue1']
dist_features = [f for f in dist_features if f in df.columns]

if len(dist_features) > 0:
    # Create distribution plots
    n_features = len(dist_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, feature in enumerate(dist_features):
        if feature in df.columns and df[feature].notna().any():
            # Plot histogram with KDE
            ax = axes[idx]
            data = df[feature].dropna()
            
            # Histogram
            ax.hist(data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            
            # KDE overlay
            if len(data) > 1:
                density = stats.gaussian_kde(data)
                x = np.linspace(data.min(), data.max(), 100)
                ax.plot(x, density(x), 'r-', linewidth=2, label='KDE')
            
            # Add normal distribution overlay
            mu, sigma = data.mean(), data.std()
            x = np.linspace(data.min(), data.max(), 100)
            ax.plot(x, stats.norm.pdf(x, mu, sigma), 'g--', linewidth=2, label='Normal')
            
            ax.set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add text with skewness
            skew_val = data.skew()
            ax.text(0.7, 0.9, f'Skew: {skew_val:.2f}', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide empty subplots
    for idx in range(len(dist_features), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Distribution Analysis of Key Features', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# ================================================================================
# 5. OUTLIER DETECTION
# ================================================================================
print("\n" + "="*80)
print("5. OUTLIER DETECTION")
print("="*80)

def detect_outliers_iqr(df, column):
    """Detect outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers), lower_bound, upper_bound

# Detect outliers for key features
outlier_summary = []
outlier_features = ['pm25', 'urban_heat_idx', 'tree_percent_w', 'dist_to_venue1']
outlier_features = [f for f in outlier_features if f in df.columns]

print("\n🔍 Outlier Detection Results (IQR Method):")
for feature in outlier_features:
    if feature in df.columns and df[feature].notna().any():
        n_outliers, lower, upper = detect_outliers_iqr(df, feature)
        outlier_pct = (n_outliers / len(df)) * 100
        outlier_summary.append({
            'Feature': feature,
            'Outliers': n_outliers,
            'Outlier_Pct': outlier_pct,
            'Lower_Bound': lower,
            'Upper_Bound': upper
        })
        print(f"• {feature}: {n_outliers} outliers ({outlier_pct:.1f}%)")

# Create boxplots for outlier visualization
if len(outlier_features) > 0:
    fig, axes = plt.subplots(1, len(outlier_features), figsize=(4*len(outlier_features), 6))
    if len(outlier_features) == 1:
        axes = [axes]
    
    for idx, feature in enumerate(outlier_features):
        if feature in df.columns and df[feature].notna().any():
            data = df[feature].dropna()
            axes[idx].boxplot(data, vert=True)
            axes[idx].set_title(f'{feature}\nOutlier Analysis', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Value')
            axes[idx].grid(True, alpha=0.3)
            
            # Add outlier count
            n_outliers, _, _ = detect_outliers_iqr(df, feature)
            axes[idx].text(0.5, 0.02, f'Outliers: {n_outliers}', 
                          transform=axes[idx].transAxes,
                          ha='center',
                          bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.suptitle('Outlier Detection using Boxplots', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outlier_detection.png', dpi=300, bbox_inches='tight')
    plt.show()

# ================================================================================
# 6. CORRELATION ANALYSIS
# ================================================================================
print("\n" + "="*80)
print("6. CORRELATION ANALYSIS")
print("="*80)

# Calculate correlation matrix for numeric columns
numeric_df = df.select_dtypes(include=[np.number])

if len(numeric_df.columns) > 1:
    correlation_matrix = numeric_df.corr()
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.7:
                high_corr_pairs.append({
                    'Feature_1': correlation_matrix.columns[i],
                    'Feature_2': correlation_matrix.columns[j],
                    'Correlation': correlation_matrix.iloc[i, j]
                })
    
    if high_corr_pairs:
        print("\n⚡ Highly Correlated Feature Pairs (|r| > 0.7):")
        high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', 
                                                                  key=lambda x: abs(x), 
                                                                  ascending=False)
        print(high_corr_df.head(20).to_string(index=False))
    
    # Create correlation heatmap for select features
    # Select features from different categories for correlation analysis
    selected_features = []
    
    # Add some features from each category (if they exist)
    feature_samples = {
        'Environmental': ['pm25', 'urban_heat_idx', 'tree_percent_w'],
        'Distance': ['dist_to_busline_1', 'dist_to_metrostop_1', 'dist_to_venue1'],
        'Geographic': ['latitude', 'longitude']
    }
    
    for category, features in feature_samples.items():
        for feature in features:
            if feature in numeric_df.columns:
                selected_features.append(feature)
    
    if len(selected_features) > 1:
        # Create correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_subset = numeric_df[selected_features].corr()
        
        sns.heatmap(corr_subset, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   fmt='.2f', ax=ax)
        ax.set_title('Correlation Matrix - Key Features', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

# ================================================================================
# 7. GEOGRAPHIC ANALYSIS
# ================================================================================
print("\n" + "="*80)
print("7. GEOGRAPHIC ANALYSIS")
print("="*80)

if 'latitude' in df.columns and 'longitude' in df.columns:
    print("\n🗺️ Geographic Coverage:")
    print(f"• Latitude range: [{df['latitude'].min():.4f}, {df['latitude'].max():.4f}]")
    print(f"• Longitude range: [{df['longitude'].min():.4f}, {df['longitude'].max():.4f}]")
    print(f"• Geographic span: {df['latitude'].max() - df['latitude'].min():.4f}° N-S, "
          f"{df['longitude'].max() - df['longitude'].min():.4f}° E-W")
    
    # Create geographic scatter plots with environmental variables
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Basic geographic distribution
    axes[0, 0].scatter(df['longitude'], df['latitude'], alpha=0.6, s=20)
    axes[0, 0].set_xlabel('Longitude')
    axes[0, 0].set_ylabel('Latitude')
    axes[0, 0].set_title('Geographic Distribution of Data Points', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: PM2.5 concentration
    if 'pm25' in df.columns:
        scatter = axes[0, 1].scatter(df['longitude'], df['latitude'], 
                                     c=df['pm25'], cmap='YlOrRd', 
                                     alpha=0.7, s=30)
        axes[0, 1].set_xlabel('Longitude')
        axes[0, 1].set_ylabel('Latitude')
        axes[0, 1].set_title('PM2.5 Concentration by Location', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 1], label='PM2.5')
    
    # Plot 3: Urban Heat Index
    if 'urban_heat_idx' in df.columns:
        scatter = axes[1, 0].scatter(df['longitude'], df['latitude'], 
                                     c=df['urban_heat_idx'], cmap='hot', 
                                     alpha=0.7, s=30)
        axes[1, 0].set_xlabel('Longitude')
        axes[1, 0].set_ylabel('Latitude')
        axes[1, 0].set_title('Urban Heat Index by Location', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0], label='Urban Heat Index')
    
    # Plot 4: Tree Coverage
    if 'tree_percent_w' in df.columns:
        scatter = axes[1, 1].scatter(df['longitude'], df['latitude'], 
                                     c=df['tree_percent_w'], cmap='YlGn', 
                                     alpha=0.7, s=30)
        axes[1, 1].set_xlabel('Longitude')
        axes[1, 1].set_ylabel('Latitude')
        axes[1, 1].set_title('Tree Coverage % by Location', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 1], label='Tree Coverage %')
    
    plt.suptitle('Geographic Analysis of Environmental Variables', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('geographic_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# ================================================================================
# 8. FEATURE RELATIONSHIPS
# ================================================================================
print("\n" + "="*80)
print("8. KEY FEATURE RELATIONSHIPS")
print("="*80)

# Analyze relationships between environmental variables and distances
relationship_pairs = [
    ('urban_heat_idx', 'tree_percent_w', 'Heat vs Tree Coverage'),
    ('pm25', 'dist_to_busline_1', 'PM2.5 vs Distance to Bus'),
    ('urban_heat_idx', 'dist_to_venue1', 'Heat vs Distance to Venue'),
    ('tree_percent_w', 'dist_to_metrostop_1', 'Tree Coverage vs Distance to Metro')
]

valid_pairs = [(x, y, title) for x, y, title in relationship_pairs 
               if x in df.columns and y in df.columns]

if valid_pairs:
    n_pairs = len(valid_pairs)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (x_col, y_col, title) in enumerate(valid_pairs[:4]):
        # Create scatter plot with regression line
        valid_data = df[[x_col, y_col]].dropna()
        
        if len(valid_data) > 0:
            axes[idx].scatter(valid_data[x_col], valid_data[y_col], 
                            alpha=0.5, s=30, color='blue')
            
            # Add regression line
            z = np.polyfit(valid_data[x_col], valid_data[y_col], 1)
            p = np.poly1d(z)
            axes[idx].plot(valid_data[x_col].sort_values(), 
                         p(valid_data[x_col].sort_values()),
                         "r--", linewidth=2, alpha=0.8)
            
            # Calculate correlation
            correlation = valid_data[x_col].corr(valid_data[y_col])
            
            axes[idx].set_xlabel(x_col)
            axes[idx].set_ylabel(y_col)
            axes[idx].set_title(f'{title}\n(r = {correlation:.3f})', fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(valid_pairs), 4):
        axes[idx].set_visible(False)
    
    plt.suptitle('Analysis of Key Feature Relationships', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('feature_relationships.png', dpi=300, bbox_inches='tight')
    plt.show()

# ================================================================================
# 9. VENUE ANALYSIS
# ================================================================================
print("\n" + "="*80)
print("9. VENUE ANALYSIS")
print("="*80)

if 'closest_venue_sport' in df.columns:
    print("\n🏟️ Closest Venue Sports Distribution:")
    venue_counts = df['closest_venue_sport'].value_counts()
    print(venue_counts.to_string())
    
    # Create venue visualization
    if len(venue_counts) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart
        venue_counts.plot(kind='bar', ax=axes[0], color='steelblue')
        axes[0].set_title('Distribution of Closest Venue Sports', fontweight='bold')
        axes[0].set_xlabel('Sport Type')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Distance analysis by sport type
        if 'dist_to_venue1' in df.columns:
            df.boxplot(column='dist_to_venue1', by='closest_venue_sport', 
                      ax=axes[1], grid=False)
            axes[1].set_title('Distance to Venue by Sport Type', fontweight='bold')
            axes[1].set_xlabel('Sport Type')
            axes[1].set_ylabel('Distance to Venue')
            plt.sca(axes[1])
            plt.xticks(rotation=45)
            axes[1].get_figure().suptitle('')  # Remove the automatic title
        
        plt.tight_layout()
        plt.savefig('venue_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

# ================================================================================
# 10. DATA QUALITY REPORT
# ================================================================================
print("\n" + "="*80)
print("10. DATA QUALITY REPORT")
print("="*80)

print("\n📋 Data Quality Summary:")

# Check for duplicates
n_duplicates = df.duplicated().sum()
print(f"• Duplicate rows: {n_duplicates} ({n_duplicates/len(df)*100:.2f}%)")

# Check for constant columns
constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
if constant_cols:
    print(f"• Constant columns: {len(constant_cols)} - {constant_cols[:5]}...")
else:
    print("• Constant columns: None")

# Check for high cardinality categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
high_cardinality = []
for col in categorical_cols:
    if df[col].nunique() > 50:
        high_cardinality.append(col)
if high_cardinality:
    print(f"• High cardinality categorical columns: {high_cardinality}")

# Check data ranges for anomalies
print("\n🔍 Potential Data Quality Issues:")
issues_found = False

# Check latitude/longitude ranges (assuming US data)
if 'latitude' in df.columns:
    lat_outliers = df[(df['latitude'] < 24) | (df['latitude'] > 50)]
    if len(lat_outliers) > 0:
        print(f"• {len(lat_outliers)} rows with unusual latitude values")
        issues_found = True

if 'longitude' in df.columns:
    lon_outliers = df[(df['longitude'] > -66) | (df['longitude'] < -125)]
    if len(lon_outliers) > 0:
        print(f"• {len(lon_outliers)} rows with unusual longitude values")
        issues_found = True

# Check for negative distances
distance_cols = [col for col in df.columns if 'dist_' in col]
for col in distance_cols:
    if col in df.columns:
        negative_distances = df[df[col] < 0]
        if len(negative_distances) > 0:
            print(f"• {len(negative_distances)} negative values in {col}")
            issues_found = True

if not issues_found:
    print("• No major data quality issues detected!")

# ================================================================================
# 11. KEY INSIGHTS AND RECOMMENDATIONS
# ================================================================================
print("\n" + "="*80)
print("11. KEY INSIGHTS AND HYPOTHESES")
print("="*80)

print("\n💡 Key Insights from EDA:")
print("1. Geographic Pattern: Data appears to cover a specific urban area")
print("2. Environmental Factors: Multiple environmental metrics are tracked")
print("3. Accessibility: Distance to various amenities is comprehensively measured")
print("4. Vulnerability Assessment: CVA metrics suggest social vulnerability analysis")
print("5. Urban Planning: LA Shade metrics indicate urban canopy/shade analysis")

print("\n🔬 Hypotheses for Further Investigation:")
print("1. Urban heat islands correlate negatively with tree coverage")
print("2. PM2.5 levels vary with distance to major transportation routes")
print("3. Social vulnerability indices correlate with environmental quality metrics")
print("4. Access to venues/amenities varies by geographic location")
print("5. Tree canopy gaps may predict higher urban heat indices")

print("\n📊 Recommended Next Steps:")
print("1. Perform spatial clustering to identify geographic patterns")
print("2. Build predictive models for environmental quality metrics")
print("3. Conduct time series analysis if temporal data becomes available")
print("4. Investigate causal relationships between urban planning features and outcomes")
print("5. Create composite indices for overall environmental quality and accessibility")

# ================================================================================
# SAVE SUMMARY REPORT
# ================================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\n📁 Output Files Generated:")
print("• missing_values_analysis.png - Missing value patterns")
print("• distribution_analysis.png - Distribution of key features")
print("• outlier_detection.png - Outlier analysis")
print("• correlation_matrix.png - Feature correlations")
print("• geographic_analysis.png - Spatial distribution of variables")
print("• feature_relationships.png - Key feature relationships")
print("• venue_analysis.png - Venue-related analysis")

# Create a summary report
summary_report = f"""
EXPLORATORY DATA ANALYSIS SUMMARY REPORT
=========================================

Dataset Overview:
- Rows: {df.shape[0]}
- Columns: {df.shape[1]}
- Missing Values: {df.isnull().sum().sum()}
- Duplicate Rows: {n_duplicates}

Feature Categories:
- Geographic: {len(geo_cols)} features
- Distance-based: {len(distance_cols)} features
- LA Shade: {len(lashade_cols)} features
- Vulnerability (CVA): {len(cva_cols)} features
- Environmental: {len(environmental_cols)} features

Data Quality:
- Completeness: {100 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.1f}%
- Numeric Features: {len(numeric_cols)}
- Categorical Features: {len(categorical_cols)}

Generated: {pd.Timestamp.now()}
"""

# Save the summary report
with open('eda_summary_report.txt', 'w') as f:
    f.write(summary_report)

print("\n✅ EDA Complete! Check the generated files for detailed visualizations and insights.")