# Feature Documentation: shade_optimization_data_usc_simple_features.csv

**Dataset**: 1,155 grid points × 121 features
**Coordinate System**: WGS84 (EPSG:4326) - [longitude, latitude]
**Coverage Area**: Los Angeles County (lat 33.70°-34.85°, lon -118.95° to -117.65°)

---

## Table of Contents
1. [Location Features (2)](#location-features)
2. [Infrastructure Distance Features (13)](#infrastructure-distance-features)
3. [LA Shade Features (31)](#la-shade-features)
4. [Olympic Venue Features (3)](#olympic-venue-features)
5. [CVA Social Vulnerability Features (26)](#cva-social-vulnerability-features)
6. [Environmental Features (5)](#environmental-features)
7. [Engineered & Normalized Features (37)](#engineered--normalized-features)
8. [Interaction Features (10)](#interaction-features)

---

## Location Features
**Count**: 2

| Feature | Description | Units |
|---------|-------------|-------|
| `latitude` | Grid point latitude | Decimal degrees |
| `longitude` | Grid point longitude | Decimal degrees |

---

## Infrastructure Distance Features
**Count**: 13

### Cooling/Heating Centers
| Feature | Description | Units |
|---------|-------------|-------|
| `dist_to_ac_1` | Distance to nearest cooling center | km |
| `dist_to_ac_3` | Mean distance to 3 nearest cooling centers | km |

### Hydration Stations
| Feature | Description | Units |
|---------|-------------|-------|
| `dist_to_hydro_1` | Distance to nearest hydration station | km |
| `dist_to_hydro_3` | Mean distance to 3 nearest hydration stations | km |
| `install_year_hydro_3` | Installation year of 3rd nearest hydration station | Year |

### Public Transit
| Feature | Description | Units |
|---------|-------------|-------|
| `dist_to_busline_1` | Distance to nearest bus line | km |
| `dist_to_busline_3` | Mean distance to 3 nearest bus lines | km |
| `dist_to_busstop_1` | Distance to nearest bus stop | km |
| `dist_to_busstop_3` | Mean distance to 3 nearest bus stops | km |
| `dist_to_metroline_1` | Distance to nearest metro line | km |
| `dist_to_metroline_3` | Mean distance to 3 nearest metro lines | km |
| `dist_to_metrostop_1` | Distance to nearest metro stop | km |
| `dist_to_metrostop_3` | Mean distance to 3 nearest metro stops | km |

---

## LA Shade Features
**Count**: 31
**Source**: LA Shade GeoJSON dataset

### Population Metrics
| Feature | Description | Units |
|---------|-------------|-------|
| `lashade_ua_pop` | Urban area population | Count |
| `lashade_cbg_pop` | Census block group population | Count |
| `lashade_acs_pop` | ACS (American Community Survey) population | Count |

### Tree Canopy Metrics
| Feature | Description | Units |
|---------|-------------|-------|
| `lashade_biome` | Biome classification | Category |
| `lashade_tc_goal` | Tree canopy coverage goal | Percentage |
| `lashade_treecanopy` | Actual tree canopy coverage | Percentage |
| `lashade_tc_gap` | Tree canopy gap (goal - actual) | Percentage |

### Demographic Metrics
| Feature | Description | Units |
|---------|-------------|-------|
| `lashade_pctpoc` | Percentage people of color | Percentage |
| `lashade_pctpov` | Percentage living in poverty | Percentage |
| `lashade_unemplrate` | Unemployment rate | Percentage |
| `lashade_dep_ratio` | Dependency ratio | Ratio |
| `lashade_dep_perc` | Dependency percentage | Percentage |
| `lashade_linguistic` | Linguistic isolation score | Score |
| `lashade_child_perc` | Percentage children | Percentage |
| `lashade_seniorperc` | Percentage senior citizens | Percentage |

### Health & Heat Metrics
| Feature | Description | Units |
|---------|-------------|-------|
| `lashade_health_nor` | Normalized health vulnerability | Normalized [0,1] |
| `lashade_temp_diff` | Temperature differential | °C |
| `lashade_tes` | Thermal environmental score | Score |

### Environmental Justice
| Feature | Description | Units |
|---------|-------------|-------|
| `lashade_holc_grade` | Historical redlining grade (HOLC) | Category (A-D) |
| `lashade_ej_disadva` | Environmental justice disadvantage | Score |
| `lashade_rank` | Composite vulnerability rank | Rank |
| `lashade_rankgrpsz` | Rank group size | Count |

### Shade Coverage by Time of Day
| Feature | Description | Units |
|---------|-------------|-------|
| `lashade_bld1200` | Building shade at 12:00 PM | Percentage |
| `lashade_veg1200` | Vegetation shade at 12:00 PM | Percentage |
| `lashade_tot1200` | Total shade at 12:00 PM | Percentage |
| `lashade_bld1500` | Building shade at 3:00 PM | Percentage |
| `lashade_veg1500` | Vegetation shade at 3:00 PM | Percentage |
| `lashade_tot1500` | Total shade at 3:00 PM | Percentage |
| `lashade_bld1800` | Building shade at 6:00 PM | Percentage |
| `lashade_veg1800` | Vegetation shade at 6:00 PM | Percentage |
| `lashade_tot1800` | Total shade at 6:00 PM | Percentage |

---

## Olympic Venue Features
**Count**: 3

| Feature | Description | Units |
|---------|-------------|-------|
| `dist_to_venue1` | Distance to nearest Olympic venue | km |
| `closest_venue_sport` | Sport of nearest Olympic venue | Category |
| `dist_to_venue3` | Mean distance to 3 nearest Olympic venues | km |

---

## CVA Social Vulnerability Features
**Count**: 26
**Source**: Climate Vulnerability Assessment (CVA)

### Population Characteristics
| Feature | Description | Units |
|---------|-------------|-------|
| `cva_population` | Total population | Count |
| `cva_children` | Number of children | Count |
| `cva_older_adults` | Number of older adults (65+) | Count |
| `cva_older_adults_living_alone` | Older adults living alone | Count |
| `cva_female` | Female population | Count |
| `cva_female_householder` | Female householders | Count |
| `cva_foreign_born` | Foreign-born residents | Count |

### Socioeconomic Vulnerability
| Feature | Description | Units |
|---------|-------------|-------|
| `cva_limited_english` | Limited English proficiency | Count |
| `cva_no_high_school_diploma` | No high school diploma | Count |
| `cva_median_income` | Median household income | USD |
| `cva_poverty` | Population below poverty line | Count |
| `cva_unemployed` | Unemployed population | Count |
| `cva_outdoor_workers` | Outdoor workers | Count |
| `cva_renters` | Renter households | Count |
| `cva_rent_burden` | Rent-burdened households (>30% income) | Count |

### Housing & Access
| Feature | Description | Units |
|---------|-------------|-------|
| `cva_living_in_group_quarters` | Living in group quarters | Count |
| `cva_mobile_homes` | Mobile home residents | Count |
| `cva_households_without_vehicle_acce` | Households without vehicle access | Count |
| `cva_no_internet_subscription` | No internet subscription | Count |

### Health Metrics
| Feature | Description | Units |
|---------|-------------|-------|
| `cva_disability` | Population with disability | Count |
| `cva_no_health_insurance` | Without health insurance | Count |
| `cva_asthma` | Asthma prevalence | Rate |
| `cva_cardiovascular_disease` | Cardiovascular disease prevalence | Rate |

### Composite Indices
| Feature | Description | Units |
|---------|-------------|-------|
| `cva_sovi_score` | Social Vulnerability Index (SoVI) | Score |
| `cva_transit_access` | Transit accessibility score | Score |
| `cva_voter_turnout_rate` | Voter turnout rate | Percentage |

---

## Environmental Features
**Count**: 5

### Air Quality
| Feature | Description | Units |
|---------|-------------|-------|
| `pm25` | PM2.5 air pollution concentration | μg/m³ |
| `pm25_percentile` | PM2.5 percentile ranking | Percentile [0-100] |

### Tree Coverage
| Feature | Description | Units |
|---------|-------------|-------|
| `tree_percent_w` | Tree canopy percentage (weighted) | Percentage |

### Urban Heat
| Feature | Description | Units |
|---------|-------------|-------|
| `urban_heat_idx` | Urban Heat Island index | Index |
| `urban_heat_idx_percentile` | UHI percentile ranking | Percentile [0-100] |

### Vacant Planting Sites
| Feature | Description | Units |
|---------|-------------|-------|
| `dist_to_vacant_park_1` | Distance to nearest vacant park site | km |
| `dist_to_vacant_park_3` | Mean distance to 3 nearest vacant park sites | km |
| `dist_to_vacant_street_1` | Distance to nearest vacant street site | km |
| `dist_to_vacant_street_3` | Mean distance to 3 nearest vacant street sites | km |

---

## Engineered & Normalized Features
**Count**: 37

### Temperature & Heat Metrics
| Feature | Description | Formula/Source | Range |
|---------|-------------|----------------|-------|
| `land_surface_temp_c` | Land surface temperature | Derived from satellite data | °C |
| `temp_severity_norm` | Normalized temperature severity | Min-max normalized | [0,1] |
| `uhi_norm` | Normalized Urban Heat Island | Min-max normalized | [0,1] |

### Canopy & Vegetation
| Feature | Description | Formula/Source | Range |
|---------|-------------|----------------|-------|
| `canopy_gap_norm` | Normalized canopy gap | `(tc_goal - tc_actual)` normalized | [0,1] |
| `canopy_pct_of_goal` | Canopy as percentage of goal | `tc_actual / tc_goal` | [0,1+] |
| `vegetation_deficit` | Vegetation deficit score | Composite metric | Float |
| `avg_shade_coverage` | Average shade across time periods | Mean of `tot1200`, `tot1500`, `tot1800` | Percentage |
| `shade_gap` | Overall shade deficit | Engineered composite | Float |

### Air Quality
| Feature | Description | Formula/Source | Range |
|---------|-------------|----------------|-------|
| `pm25_norm` | Normalized PM2.5 | Min-max normalized | [0,1] |

### Population Metrics
| Feature | Description | Formula/Source | Range |
|---------|-------------|----------------|-------|
| `population_norm` | Normalized population | Min-max normalized | [0,1] |
| `vulnerable_population` | Vulnerable population count | Sum of vulnerable groups | Count |
| `vulnerable_pop_norm` | Normalized vulnerable population | Min-max normalized | [0,1] |

### Olympic Proximity
| Feature | Description | Formula/Source | Range |
|---------|-------------|----------------|-------|
| `olympic_proximity` | Inverse distance to Olympic venues | `1 / (1 + dist_to_venue1)` | [0,1] |
| `olympic_proximity_norm` | Normalized Olympic proximity | Min-max normalized | [0,1] |

### Infrastructure Access
| Feature | Description | Formula/Source | Range |
|---------|-------------|----------------|-------|
| `cooling_distance_norm` | Normalized cooling center distance | Min-max normalized | [0,1] |
| `hydration_distance_norm` | Normalized hydration station distance | Min-max normalized | [0,1] |
| `avg_transit_distance` | Average distance to transit | Mean of bus/metro distances | km |
| `avg_transit_distance_norm` | Normalized transit distance | Min-max normalized | [0,1] |
| `transit_access_score` | Transit accessibility score | Inverse distance composite | Score |
| `transit_access_norm` | Normalized transit access | Min-max normalized | [0,1] |
| `avg_infrastructure_distance` | Average infrastructure distance | Mean of cooling/hydration/transit | km |

### Planting Opportunities
| Feature | Description | Formula/Source | Range |
|---------|-------------|----------------|-------|
| `avg_vacant_distance` | Average distance to vacant sites | Mean of park/street vacant distances | km |
| `planting_opportunity` | Planting opportunity score | Composite metric | Score |
| `planting_opportunity_norm` | Normalized planting opportunity | Min-max normalized | [0,1] |

### Socioeconomic Normalized
| Feature | Description | Formula/Source | Range |
|---------|-------------|----------------|-------|
| `cva_transit_norm` | Normalized CVA transit access | Min-max normalized | [0,1] |
| `sovi_norm` | Normalized Social Vulnerability Index | Min-max normalized | [0,1] |
| `poverty_norm` | Normalized poverty rate | Min-max normalized | [0,1] |
| `education_gap_norm` | Normalized education gap | Min-max normalized | [0,1] |
| `rent_burden_norm` | Normalized rent burden | Min-max normalized | [0,1] |
| `limited_english_norm` | Normalized limited English proficiency | Min-max normalized | [0,1] |

### Health & Vulnerability
| Feature | Description | Formula/Source | Range |
|---------|-------------|----------------|-------|
| `avg_health_vulnerability` | Average health vulnerability | Mean of asthma, CVD, etc. | Score |
| `health_vulnerability_norm` | Normalized health vulnerability | Min-max normalized | [0,1] |

### Environmental Justice
| Feature | Description | Formula/Source | Range |
|---------|-------------|----------------|-------|
| `env_justice_binary` | Environmental justice binary flag | Threshold-based classification | {0,1} |

### Spatial Metrics
| Feature | Description | Formula/Source | Range |
|---------|-------------|----------------|-------|
| `spatial_isolation_km` | Spatial isolation distance | Distance to nearest infrastructure cluster | km |
| `spatial_isolation_norm` | Normalized spatial isolation | Min-max normalized | [0,1] |

---

## Interaction Features
**Count**: 10

These features capture multiplicative relationships between key dimensions:

| Feature | Description | Formula | Use Case |
|---------|-------------|---------|----------|
| `heat_x_population` | Heat exposure × population density | `uhi_norm × population_norm` | Identifies heat-stressed populous areas |
| `heat_x_poverty` | Heat exposure × poverty rate | `uhi_norm × poverty_norm` | Heat burden on low-income areas |
| `heat_x_vulnerable` | Heat × vulnerable population | `uhi_norm × vulnerable_pop_norm` | Heat impact on vulnerable groups |
| `heat_x_sovi` | Heat × social vulnerability | `uhi_norm × sovi_norm` | Compounded heat + social stress |
| `population_x_transit_gap` | Population × transit inaccessibility | `population_norm × (1 - transit_access_norm)` | Transit-underserved populous areas |
| `poverty_x_cooling_gap` | Poverty × cooling inaccessibility | `poverty_norm × cooling_distance_norm` | Cooling access equity |
| `veg_deficit_x_uhi` | Vegetation deficit × heat island | `vegetation_deficit × uhi_norm` | Areas needing tree cover most |
| `canopy_gap_x_pm25` | Canopy gap × air pollution | `canopy_gap_norm × pm25_norm` | Air quality + shade deficits |
| `olympic_x_population` | Olympic proximity × population | `olympic_proximity_norm × population_norm` | High-traffic Olympic areas |
| `shade_gap_x_isolation` | Shade gap × spatial isolation | `shade_gap × spatial_isolation_norm` | Underserved isolated areas |

---

## Feature Engineering Notes

### Normalization Method
All `*_norm` features use **min-max normalization**:
```
normalized = (value - min) / (max - min)
```
Range: [0, 1] where 0 = minimum observed, 1 = maximum observed

### Key Composite Indices

#### Environmental Exposure Index
Not directly present but can be computed as:
```
env_exposure = w1·(1 - tree_canopy) + w2·pm25_norm + w3·impervious_ratio
```

#### Heat Vulnerability Score
Composite of:
- `uhi_norm` (urban heat intensity)
- `canopy_gap_norm` (lack of shade)
- `pm25_norm` (air quality)
- `temp_severity_norm` (temperature extremes)

#### Equity Priority Score
Composite of:
- `sovi_norm` (social vulnerability)
- `poverty_norm` (economic disadvantage)
- `vulnerable_pop_norm` (at-risk populations)
- `env_justice_binary` (environmental justice designation)

---

## Data Quality Notes

### Missing Data Strategy
- **LA Shade features**: Spatial imputation using BallTree with haversine metric (K=1 nearest neighbor)
- **CVA features**: Simple median/mode imputation
- Features with >60% missingness are excluded (except LA Shade features)

### Outlier Handling
- Normalized features cap extreme values during min-max scaling
- Distance features may have right-skewed distributions (long tail for remote areas)

### Coordinate Reference System
- **Input**: WGS84 (EPSG:4326)
- **Distance calculations**: Haversine formula (great-circle distance on sphere)
- **Units**: All distances in kilometers (km)

---

## Usage in RL Optimization

### State Representation
Each grid point's feature vector serves as the state representation for Q-learning.

### Reward Function Components
The 121 features support 5 reward components:

1. **Heat Vulnerability (30%)**: `uhi_norm`, `temp_severity_norm`, `canopy_gap_norm`, `pm25_norm`
2. **Population Impact (25%)**: `population_norm`, `olympic_proximity_norm`, `transit_access_norm`
3. **Accessibility (20%)**: `cooling_distance_norm`, `hydration_distance_norm`, `planting_opportunity_norm`
4. **Equity (15%)**: `sovi_norm`, `poverty_norm`, `vulnerable_pop_norm`, `env_justice_binary`
5. **Coverage Efficiency (10%)**: Spatial distance to existing shade placements (state-dependent)

### Feature Selection for Training
Not all 121 features may be used in final RL training. Key features are selected based on:
- Correlation analysis (removing highly collinear features)
- Domain importance (reward component alignment)
- Statistical significance (variance, information gain)

---

## Comparison to Other Datasets

### vs. `shade_optimization_data.csv` (Original)
- **Grid points**: 1,155 vs 2,650 (subset/filtered)
- **Features**: 121 vs 84 (more engineered features)
- **Processing**: Includes normalized + interaction features

### vs. `shade_optimization_data_cleaned.csv`
- **Grid points**: 1,155 vs 2,650
- **Features**: More interaction terms and normalized variants
- **Purpose**: USC-specific analysis with simpler feature engineering

---

## References
- **LA Shade Dataset**: City of Los Angeles Tree Canopy & Shade Analysis
- **CVA Data**: Los Angeles County Climate Vulnerability Assessment
- **Olympic Venues**: LA 2028 Olympic Games venue locations
- **Transit Data**: LA Metro and LADOT public transit networks
- **Environmental Justice**: CalEnviroScreen 4.0 + Historical HOLC redlining data

---

**Last Updated**: 2025-12-01
**Dataset Version**: USC Simple Features v1.0
**Contact**: See project README for maintainer information
