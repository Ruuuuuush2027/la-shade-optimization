# la-shade-optimization

proposal doc: https://docs.google.com/document/d/13zeRmjD0UWovD3XOSwQIl6TwmejcEmR9/edit

the dataset to be optimized on is `shade_optimization_data.csv`

Our data features:
```
{
    # D1: Cooling/Heating Centers (2 features)
    'dist_to_ac_1': 'Distance in kilometers to the nearest cooling/heating center facility',
    'dist_to_ac_3': 'Average distance in kilometers to the 3 closest cooling/heating centers',
    
    # D2: Hydration Stations (3 features)
    'dist_to_hydro_1': 'Distance in kilometers to the closest water hydration station',
    'dist_to_hydro_3': 'Average distance in kilometers to the 3 closest hydration stations',
    'install_year_hydro_3': 'Average installation year of the 3 closest hydration stations',
    
    # D3: Bus Lines (2 features)
    'dist_to_busline_1': 'Distance in kilometers to the nearest bus line',
    'dist_to_busline_3': 'Average distance in kilometers to the 3 closest bus lines',
    
    # D4: Bus Stops (2 features)
    'dist_to_busstop_1': 'Distance in kilometers to the nearest bus stop',
    'dist_to_busstop_3': 'Average distance in kilometers to the 3 closest bus stops',
    
    # D5: Metro Lines (2 features)
    'dist_to_metroline_1': 'Distance in kilometers to the nearest metro line',
    'dist_to_metroline_3': 'Average distance in kilometers to the 3 closest metro lines',
    
    # D6: Metro Stations (2 features)
    'dist_to_metrostop_1': 'Distance in kilometers to the nearest metro station',
    'dist_to_metrostop_3': 'Average distance in kilometers to the 3 closest metro stations',
    
    # D7: LA Shade Data (31 features)
    'lashade_ua_pop': 'Sum of the clipped block group population in the urban area',
    'lashade_cbg_pop': 'Total population of the clipped block group (part within urban area) from 2020 Census',
    'lashade_acs_pop': 'Total population of the entire block group (includes non-urban) from ACS Community Survey',
    'lashade_biome': 'Biome classification of the block group',
    'lashade_tc_goal': 'Tree canopy goal for the block group (range: 0-1)',
    'lashade_treecanopy': 'Current tree canopy percentage of the block group (range: 0-1)',
    'lashade_tc_gap': 'Tree canopy gap - difference between goal and actual canopy (range: 0-1)',
    'lashade_pctpoc': 'Percent of people of color in the block group (range: 0-1)',
    'lashade_pctpov': 'Percent of people in poverty in the block group (range: 0-1)',
    'lashade_unemplrate': 'Unemployment rate in the block group (range: 0-1)',
    'lashade_dep_ratio': 'Dependency ratio (children + seniors / working age adults 18-64)',
    'lashade_dep_perc': 'Percent of population that are children and seniors (range: 0-1)',
    'lashade_linguistic': 'Percent of households with linguistic isolation (range: 0-1)',
    'lashade_health_nor': 'Normalized health burden index of the block group (range: 0-1)',
    'lashade_temp_diff': 'Temperature difference compared to urban area average (heat extremity)',
    'lashade_tes': 'Tree Equity Score of the block group (range: 0-100)',
    'lashade_holc_grade': "Home Owner's Loan Corporation grade (A, B, C, or D)",
    'lashade_child_perc': 'Percent of children in the block group (range: 0-1)',
    'lashade_seniorperc': 'Percent of seniors in the block group (range: 0-1)',
    'lashade_ej_disadva': 'Whether the community is designated "disadvantaged" by US EPA (Yes/No)',
    'lashade_rank': 'Rank of the Tree Equity Score within the municipality',
    'lashade_rankgrpsz': 'Number of block groups in the municipality that are ranked',
    'lashade_bld1200': 'Percent shade cast by buildings at noon (12:00 PM)',
    'lashade_veg1200': 'Percent shade cast by trees and other vegetation at noon (12:00 PM)',
    'lashade_bld1500': 'Percent shade cast by buildings at 3:00 PM',
    'lashade_veg1500': 'Percent shade cast by trees and other vegetation at 3:00 PM',
    'lashade_bld1800': 'Percent shade cast by buildings at 6:00 PM',
    'lashade_veg1800': 'Percent shade cast by trees and other vegetation at 6:00 PM',
    'lashade_tot1200': 'Total percent shade (buildings + vegetation) at noon (12:00 PM)',
    'lashade_tot1500': 'Total percent shade (buildings + vegetation) at 3:00 PM',
    'lashade_tot1800': 'Total percent shade (buildings + vegetation) at 6:00 PM',
    
    # D8: LA28 Olympic Venues (3 features)
    'dist_to_venue1': 'Distance in kilometers to the closest LA 2028 Olympic/Paralympic venue',
    'closest_venue_sport': 'Sport/Activity held at the closest LA 2028 Olympic/Paralympic venue',
    'dist_to_venue3': 'Average distance in kilometers to the 3 closest LA 2028 Olympic/Paralympic venues',
    
    # D9: CVA Social Sensitivity Index (26 features)
    'cva_population': 'Total population of the census tract',
    'cva_children': 'Percent of children 18 and under (range: 0-1)',
    'cva_older_adults': 'Percent of persons 65 and over (range: 0-1)',
    'cva_older_adults_living_alone': 'Percent of households where householder is 65+ and living alone (range: 0-1)',
    'cva_limited_english': 'Percent of limited English speaking households (range: 0-1)',
    'cva_no_high_school_diploma': 'Percent of persons 25+ without a high school diploma (range: 0-1)',
    'cva_female': 'Percent female population (range: 0-1)',
    'cva_female_householder': 'Percent of family households with female householder and no spouse present (range: 0-1)',
    'cva_disability': 'Percent of civilian noninstitutionalized population with mental or physical disability (range: 0-1)',
    'cva_no_health_insurance': 'Percent of civilian noninstitutionalized population without health insurance (range: 0-1)',
    'cva_living_in_group_quarters': 'Percent of persons living in institutionalized or uninstitutionalized group quarters (range: 0-1)',
    'cva_mobile_homes': 'Percent of occupied housing units which are mobile homes or other types (range: 0-1)',
    'cva_rent_burden': 'Percent of renter-occupied housing units where rent is 30%+ of household income (range: 0-1)',
    'cva_renters': 'Percent of housing units which are renter-occupied (range: 0-1)',
    'cva_median_income': 'Median household income of census tract in dollars',
    'cva_poverty': 'Percent of population earning below 100% of federal poverty threshold (range: 0-1)',
    'cva_households_without_vehicle_acce': 'Percent of households without access to a personal vehicle (range: 0-1)',
    'cva_outdoor_workers': 'Percent of civilian employed population in natural resources, construction, and maintenance occupations (range: 0-1)',
    'cva_unemployed': 'Percent of population over 16 that is unemployed and eligible for labor force (range: 0-1)',
    'cva_foreign_born': 'Percent of total population not born in the United States or Puerto Rico (range: 0-1)',
    'cva_no_internet_subscription': 'Percent of population without an internet subscription (range: 0-1)',
    'cva_voter_turnout_rate': 'Percentage of registered voters voting in the general election (range: 0-1)',
    'cva_sovi_score': 'Social Vulnerability/Sensitivity score',
    'cva_asthma': 'Age-adjusted rate of emergency department visits for asthma per 10,000',
    'cva_cardiovascular_disease': 'Age-adjusted rate of emergency department visits for heart attacks per 10,000',
    'cva_transit_access': 'Percent of tract area in High Quality Transit Area (HQTA) (range: 0-1)',
    
    # D10: PM2.5 Air Quality (2 features)
    'pm25': 'PM2.5 fine particulate matter air quality value from 2015',
    'pm25_percentile': 'PM2.5 quality percentile (lower percentile = better quality, range: 0-1)',
    
    # D11: Tree Canopy Coverage (1 feature)
    'tree_percent_w': 'Percentage of land with tree canopy coverage weighted by population size (range: 0-1)',
    
    # D12: Urban Heat Island (2 features)
    'urban_heat_idx': 'Urban Heat Island intensity value (temperature difference)',
    'urban_heat_idx_percentile': 'UHI percentile rank (higher percentile = more extreme heat, range: 0-1)',
    
    # D13: Vacant Park Planting Sites (2 features)
    'dist_to_vacant_park_1': 'Distance in kilometers to the nearest vacant tree planting site in a park',
    'dist_to_vacant_park_3': 'Average distance in kilometers to the 3 closest vacant tree planting sites in parks',
    
    # D14: Vacant Street Planting Sites (2 features)
    'dist_to_vacant_street_1': 'Distance in kilometers to the nearest vacant tree planting site along a street',
    'dist_to_vacant_street_3': 'Average distance in kilometers to the 3 closest vacant tree planting sites along streets'
}
```