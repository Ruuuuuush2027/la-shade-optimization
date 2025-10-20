import math
import json
from geopy.distance import geodesic
from statistics import mean
from datetime import datetime

from shapely.geometry import Point, LineString, MultiLineString, Polygon, MultiPolygon, shape
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os

LA_COUNTY_BOUNDS = {
    "min_lat": 33.70,  # Southern boundary (near Long Beach)
    "max_lat": 34.85,  # Northern boundary (near Palmdale)
    "min_lon": -118.95, # Western boundary (near Ventura County line)
    "max_lon": -117.65  # Eastern boundary (near San Bernardino County line)
}

# A spacing of 1.5 miles ensures overlap for a 1-mile radius (2-mile diameter)
COVERAGE_RADIUS_MILES = 1.0  
CENTER_SPACING_MILES = 1.5

# Earth's radius in miles
EARTH_RADIUS_MILES = 3958.8
def degrees_to_radians(degrees: float) -> float:
    return degrees * (math.pi / 180.0)

def calculate_grid_steps(center_lat: float, spacing_miles: float) -> tuple[float, float]:
    # 1. Latitude Step (constant everywhere)
    # Distance = Arc Length = Radius * Angle (in radians)
    # Angle (radians) = Distance / Radius
    # Lat Step (degrees) = (Distance / Radius) * (180 / pi)
    lat_step_deg = (spacing_miles / EARTH_RADIUS_MILES) * (180.0 / math.pi)

    # 2. Longitude Step (varies by latitude)
    # The radius of the parallel circle at 'center_lat' is R * cos(lat).
    # Lon Step (degrees) = (Distance / (R * cos(lat))) * (180 / pi)
    lat_rad = degrees_to_radians(center_lat)
    lon_step_deg = lat_step_deg / math.cos(lat_rad)

    return lat_step_deg, lon_step_deg

def generate_coverage_grid(bounds: dict, spacing_miles: float) -> list[tuple[float, float]]:
    """
    Returns a list of (latitude, longitude) tuples for coverage area
    """
    min_lat, max_lat = bounds["min_lat"], bounds["max_lat"]
    min_lon, max_lon = bounds["min_lon"], bounds["max_lon"]

    # Calculate steps
    center_lat = (min_lat + max_lat) / 2.0
    lat_step, lon_step = calculate_grid_steps(center_lat, spacing_miles)

    coverage_points = []
    current_lat = min_lat

    # Iterate through latitudes
    while current_lat <= max_lat:
        current_lon = min_lon
        # Iterate through longitudes
        while current_lon <= max_lon:
            # Store the point as (latitude, longitude)
            coverage_points.append((current_lat, current_lon))
            current_lon += lon_step
        current_lat += lat_step

    return coverage_points

points = generate_coverage_grid(LA_COUNTY_BOUNDS, CENTER_SPACING_MILES)

FEATURE_DESCRIPTIONS = {
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

FEATURE_NAMES = list(FEATURE_DESCRIPTIONS.keys())

ac_centers_la_data = None
with open('dataset/Cooling_Heating_Centers_in_Los_Angeles.geojson', 'r') as file:
    # Load the JSON data from the file object
    data = json.load(file)
    ac_centers_la_data = data['features']

def calculate_ac_center_features(input_lat: float, input_lon: float) -> list:
    """
    Computes distance features (nearest and average of 3 closest) 
    from an input point to cooling/heating centers in the ac_centers_la_data.

    Args:
        input_lat: The latitude of the input point.
        input_lon: The longitude of the input point.

    Returns:
        A list containing two floats: 
        [distance to nearest center (km), average distance to 3 closest centers (km)].
    """
    
    input_point = (input_lat, input_lon)
    
    # 1. Calculate all distances
    try:
        all_distances = []
        for facility in ac_centers_la_data:
            # Coordinates are stored as [longitude, latitude] in the dataset
            lon, lat = facility['geometry']['coordinates']
            facility_point = (lat, lon)
            
            # Calculate distance using geodesic (great-circle distance) in kilometers
            distance_km = geodesic(input_point, facility_point).km
            all_distances.append(distance_km)
            
    except KeyError as e:
        print(f"Error processing data item: Missing key {e}")
        return [math.nan, math.nan]
    
    # 2. Sort the distances
    all_distances.sort()
    
    if not all_distances:
        print("ac_centers_la_data is empty.")
        return [math.nan, math.nan]

    # 3. Extract features
    nearest_distance = all_distances[0]
    if len(all_distances) >= 3:
        closest_three_distances = all_distances[:3]
        average_closest_three = sum(closest_three_distances) / 3
    else:
        average_closest_three = sum(all_distances) / len(all_distances)

    # 4. Return the results in a single arraylist (Python list)
    return [nearest_distance, average_closest_three]

# test_lat = 34.0300
# test_lon = -118.4700

# results_ac = calculate_ac_center_features(input_lat, input_lon)
# results_ac
# ALL_FEATURE_NAMES.extend(['dist_to_ac_1', 'dist_to_ac_3'])

hydra_stations_data = None
with open('dataset/Hydration_Stations_August_2022.geojson', 'r') as file:
    data = json.load(file)
    hydra_stations_data = data['features']

def calculate_water_station_features(input_lat: float, input_lon: float):
    """
    Computes distance and installation year features from an input point 
    to water stations in the hydra_stations_data dataset.

    Args:
        input_lat: The latitude of the input point.
        input_lon: The longitude of the input point.

    Returns:
        A list containing the computed features:
        [distance to closest water station (km), 
         average distance to closest 3 water stations (km), 
         average year of installation of closest 3 water stations (year)].
    """
    
    input_point = (input_lat, input_lon)
    
    # 1. Calculate all distances and extract relevant properties
    station_data_with_distance = []
    
    if not hydra_stations_data:
        print("hydra_stations_data is empty.")
        # Return [distance, average_distance, average_year] with NaN
        return [math.nan, math.nan, math.nan]

    try:
        for station in hydra_stations_data:
            # Coordinates are stored as [longitude, latitude] in the dataset
            lon, lat = station['geometry']['coordinates']
            station_point = (lat, lon)
            
            distance_km = geodesic(input_point, station_point).km
            
            # Extract the year of installation
            # Use 'Year' if available, otherwise attempt to parse 'Installation'
            year = station['properties'].get('Year')
            if year is None:
                try:
                    install_date_str = station['properties'].get('Installation')
                    if install_date_str:
                        year = datetime.fromisoformat(install_date_str.replace('Z', '+00:00')).year
                except ValueError:
                    year = None

            station_data_with_distance.append({
                'distance': distance_km,
                'year': year
            })
            
    except KeyError as e:
        print(f"Error processing data item: Missing key {e}")
        return [math.nan, math.nan, math.nan]

    station_data_with_distance.sort(key=lambda x: x['distance'])
    
    N = 3
    
    closest_N_stations = station_data_with_distance[:N]
    
    distance_to_closest = closest_N_stations[0]['distance'] if closest_N_stations else math.nan
    
    closest_distances = [d['distance'] for d in closest_N_stations]
    closest_years = [d['year'] for d in closest_N_stations if d['year'] is not None]

    if closest_distances:
        
        average_distance_closest_3 = mean(closest_distances)
    else:
        average_distance_closest_3 = math.nan

    # Feature 3: Average year of installation of closest water stations
    if closest_years:
        # The result should be an integer representing the average year
        average_year_closest_stations = int(round(mean(closest_years)))
    else:
        average_year_closest_stations = math.nan

    # 5. Return the results
    return [distance_to_closest, average_distance_closest_3, average_year_closest_stations]

# results_hydro = calculate_water_station_features(input_lat, input_lon)
# results_hydro
# ALL_FEATURE_NAMES.extend(['dist_to_hydro_1', 'dist_to_hydro_3', 'install_year_hydro_3'])

bus_lines_data = None
with open('dataset/LA_Bus_Lines.geojson', 'r') as file:
    data = json.load(file)
    bus_lines_data = data['features']

def calculate_bus_line_features(input_lat: float, input_lon: float) -> list:
    """
    Computes distance features (nearest and average of 3 closest)
    from an input point to bus lines in the bus_lines_data.

    Args:
        input_lat: The latitude of the input point.
        input_lon: The longitude of the input point.

    Returns:
        A list containing two floats:
        [distance to nearest bus line (km), average distance to 3 closest bus lines (km)].
    """

    # Input point in (lat, lon) format for geopy
    input_point_geopy = (input_lat, input_lon)
    # Input point in (lon, lat) format for shapely (standard GeoJSON/GIS order)
    input_point_shapely = Point(input_lon, input_lat)

    # 1. Calculate all distances
    all_distances_km = []

    if not bus_lines_data:
        print("bus_lines_data is empty.")
        return [math.nan, math.nan]

    for line_feature in bus_lines_data:
        try:
            # Coordinates are stored as [longitude, latitude] in the dataset
            coords_lon_lat = line_feature['geometry']['coordinates']

            bus_line = LineString(coords_lon_lat)

            # Find the closest point on the line geometry to the input point.
            # a) project(): Finds the distance along the line to the closest point.
            # b) interpolate(): Returns the Point object at that distance.
            closest_point_on_line_shapely = bus_line.interpolate(bus_line.project(input_point_shapely))

            # Convert the closest point back to (lat, lon) for geodesic distance
            closest_point_on_line_geopy = (closest_point_on_line_shapely.y, closest_point_on_line_shapely.x)

            # Calculate distance using geodesic (great-circle distance) in kilometers
            distance_km = geodesic(input_point_geopy, closest_point_on_line_geopy).km
            all_distances_km.append(distance_km)

        except KeyError as e:
            print(f"Error processing data item: Missing key {e}")
            continue
        except Exception as e:
            print(f"Error processing geometry: {e}")
            continue

    if not all_distances_km:
        print("No valid geometries found in bus_lines_data.")
        return [math.nan, math.nan]

    all_distances_km.sort()

    nearest_distance = all_distances_km[0]

    num_distances = len(all_distances_km)
    if num_distances >= 3:
        closest_three_distances = all_distances_km[:3]
        average_closest_three = sum(closest_three_distances) / 3
    else:
        average_closest_three = sum(all_distances_km) / num_distances

    return [nearest_distance, average_closest_three]

# --- Example of running the function ---
# Example 1: A point close to the first LineString's start (-118.49026, 34.01383)
# input_lat, input_lon = 34.0138, -118.4902
# result_1 = calculate_bus_line_features(input_lat, input_lon)
# result_1
# ALL_FEATURE_NAMES.extend(['dist_to_busline_1', 'dist_to_busline_3'])
bus_stops_data = None
with open('dataset/LA_Metro_Bus_Stops_06_2025.geojson', 'r') as file:
    data = json.load(file)
    bus_stops_data = data['features']

def calculate_bus_stop_features(input_lat: float, input_lon: float) -> list:
    """
    Computes distance features (nearest and average of 3 closest)
    from an input point to bus stops in the global bus_stops_data.

    Args:
        input_lat: The latitude of the input point.
        input_lon: The longitude of the input point.

    Returns:
        A list containing two floats:
        [distance to nearest stop (km), average distance to 3 closest stops (km)].
        Returns [math.nan, math.nan] if the bus_stops_data is empty or invalid.
    """

    if not bus_stops_data:
        print("bus_stops_data is empty.")
        return [math.nan, math.nan]

    input_point = (input_lat, input_lon)

    # 1. Calculate all distances
    all_distances = []
    try:
        for stop in bus_stops_data:
            lon, lat = stop['geometry']['coordinates']
            stop_point = (lat, lon)

            distance_km = geodesic(input_point, stop_point).km
            all_distances.append(distance_km)

    except KeyError as e:
        print(f"Error processing data item: Missing key {e}")
        return [math.nan, math.nan]
    except Exception as e:
        print(f"An unexpected error occurred during distance calculation: {e}")
        return [math.nan, math.nan]

    all_distances.sort()

    nearest_distance = all_distances[0]

    num_closest = min(3, len(all_distances))

    closest_distances = all_distances[:num_closest]
    average_closest = sum(closest_distances) / num_closest

    return [nearest_distance, average_closest]

# input_lat, input_lon = 34.0138, -118.4902
# result_1 = calculate_bus_stop_features(input_lat, input_lon)
# result_1
# ALL_FEATURE_NAMES.extend(['dist_to_busstop_1', 'dist_to_busstop_3'])
metro_lines_data = None
with open('dataset/LA_Metro_Lines.geojson', 'r') as file:
    data = json.load(file)
    metro_lines_data = data['features']

def calculate_metro_line_features(input_lat: float, input_lon: float) -> list:
    """
    Computes distance features (nearest and average of 3 closest)
    from an input point to metro lines in the metro_lines_data.

    Args:
        input_lat: The latitude of the input point.
        input_lon: The longitude of the input point.

    Returns:
        A list containing two floats:
        [distance to nearest metro line (km), average distance to 3 closest metro lines (km)].
        Returns [math.nan, math.nan] if the metro_lines_data is empty or invalid.
    """

    input_point_geopy = (input_lat, input_lon)
    input_point_shapely = Point(input_lon, input_lat)

    all_distances_km = []

    if 'metro_lines_data' not in globals() or not metro_lines_data:
        print("metro_lines_data is either not defined or is empty.")
        return [math.nan, math.nan]

    for line_feature in metro_lines_data:
        try:
            geometry_type = line_feature['geometry']['type']
            coords_lon_lat = line_feature['geometry']['coordinates']

            if geometry_type == 'LineString':
                line_geometry = LineString(coords_lon_lat)
            elif geometry_type == 'MultiLineString': # also has line string type
                line_geometry = MultiLineString(coords_lon_lat)
            else:
                continue

            closest_point_on_line_shapely = line_geometry.interpolate(line_geometry.project(input_point_shapely))

            closest_point_on_line_geopy = (closest_point_on_line_shapely.y, closest_point_on_line_shapely.x)

            distance_km = geodesic(input_point_geopy, closest_point_on_line_geopy).km
            all_distances_km.append(distance_km)

        except KeyError as e:
            print(f"Error processing data item: Missing key {e} in a feature.")
            continue
        except Exception as e:
            print(f"Error processing geometry: {e}")
            continue

    if not all_distances_km:
        print("No valid line geometries were processed in metro_lines_data.")
        return [math.nan, math.nan]

    all_distances_km.sort()

    nearest_distance = all_distances_km[0]

    num_distances = len(all_distances_km)
    if num_distances >= 3:
        closest_three_distances = all_distances_km[:3]
        average_closest_three = sum(closest_three_distances) / 3
    else:
        average_closest_three = sum(all_distances_km) / num_distances

    return [nearest_distance, average_closest_three]

# input_lat, input_lon = 34.0138, -118.4902
# result_1 = calculate_metro_line_features(input_lat, input_lon)
# result_1
# ALL_FEATURE_NAMES.extend(['dist_to_metroline_1', 'dist_to_metroline_3'])

metro_stations_data = None
with open('dataset/LA_Metro_Stations.geojson', 'r') as file:
    data = json.load(file)
    metro_stations_data = data['features']

def calculate_metro_stop_features(input_lat: float, input_lon: float) -> list:
    """
    Computes distance features (nearest and average of 3 closest)
    from an input point to bus stops in the global metro_stations_data.

    Args:
        input_lat: The latitude of the input point.
        input_lon: The longitude of the input point.

    Returns:
        A list containing two floats:
        [distance to nearest stop (km), average distance to 3 closest stops (km)].
        Returns [math.nan, math.nan] if the metro_stations_data is empty or invalid.
    """

    if not metro_stations_data:
        print("metro_stations_data is empty.")
        return [math.nan, math.nan]

    input_point = (input_lat, input_lon)

    # 1. Calculate all distances
    all_distances = []
    try:
        for stop in metro_stations_data:
            lon, lat = stop['geometry']['coordinates']
            stop_point = (lat, lon)

            distance_km = geodesic(input_point, stop_point).km
            all_distances.append(distance_km)

    except KeyError as e:
        print(f"Error processing data item: Missing key {e}")
        return [math.nan, math.nan]
    except Exception as e:
        print(f"An unexpected error occurred during distance calculation: {e}")
        return [math.nan, math.nan]

    all_distances.sort()

    nearest_distance = all_distances[0]

    num_closest = min(3, len(all_distances))

    closest_distances = all_distances[:num_closest]
    average_closest = sum(closest_distances) / num_closest

    return [nearest_distance, average_closest]

# input_lat, input_lon = 34.0138, -118.4902
# result_1 = calculate_metro_stop_features(input_lat, input_lon)
# result_1
# ALL_FEATURE_NAMES.extend(['dist_to_metrostop_1', 'dist_to_metrostop_3'])

la_shading_data = None
with open('dataset/LA_shade.geojson', 'r') as file:
    data = json.load(file)
    la_shading_data = data['features']

def get_la_shade_features(lat: float, lon: float) -> list:
    """
    Finds the first GeoJSON Polygon in the la_shading_data that contains the given
    latitude and longitude point and returns a list of its specified properties.

    Args:
        lat: The latitude of the point.
        lon: The longitude of the point.
        la_shading_data: A list of GeoJSON Feature dictionaries (your data).

    Returns:
        A list of feature values for the first containing polygon, or a
        list of None values if no containing polygon is found. The features
        are returned in the following order and represent data for the census
        block group containing the point:

        - ua_pop (int): The sum of the clipped block group population in the urban area.
        - cbg_pop (int): The total population of the clipped block group (part within urban area) from the 2020 Census.
        - acs_pop (float): The total population of the entire block group (includes non-urban) from the ACS Community Survey.
        - biome (str): The biome of the block group.
        - tc_goal (float): The tree canopy goal of the block group [range: 0-1].
        - treecanopy (float): The tree canopy percentage of the block group [range: 0-1].
        - tc_gap (float): The tree canopy gap of the block group (goal minus canopy) [range: 0-1].
        - pctpoc (float): The percent of people of color inside the block group [range: 0-1].
        - pctpov (float): The percent of people in poverty inside the block group [range: 0-1].
        - unemplrate (float): The unemployment rate inside of the block group [range: 0-1].
        - dep_ratio (float): The dependency ratio (childrens + seniors / 18-64 adults).
        - dep_perc (float): The percent of the population that are children and seniors [range: 0-1].
        - linguistic (float): The percent of households with linguistic isolation [range: 0-1].
        - health_nor (float): The normalized health burden index of the block group [range: 0-1].
        - temp_diff (float): Heat extremity difference vs. urban area average.
        - tes (int): The Tree Equity Score of the block group [range: 0-100].
        - holc_grade (str): Home Owner's Loan Corporation grade (A, B, C, D).
        - child_perc (float): The percent of children inside of the block group [range: 0-1].
        - seniorperc (float): The percent of seniors inside of the block group [range: 0-1].
        - ej_disadva (str): Is the community "disadvantaged" by the US EPA (Yes/No).
        - rank (float): The rank of the Tree Equity Score in the municipality.
        - rankgrpsz (int): The number of block groups in the municipality which are ranked.
        - _bld1200 (float): percent shade cast by buildings at noon.
        - _veg1200 (float): percent shade cast by trees and other non-building features at noon.
        - _bld1500 (float): percent shade cast by buildings at 3 p.m.
        - _veg1500 (float): percent shade cast by trees and other non-building features at 3 p.m.
        - _bld1800 (float): percent shade cast by buildings at 6 p.m.
        - _veg1800 (float): percent shade cast by trees and other non-building features at 6 p.m.
        - _tot1200 (float): total percent shade cast at noon.
        - _tot1500 (float): total percent shade cast at 3 p.m.
        - _tot1800 (float): total percent shade cast at 6 p.m.
    """
    # Keys for the desired features in the order they should be returned
    FEATURE_KEYS = [
        'ua_pop', 'cbg_pop', 'acs_pop', 'biome', 'tc_goal', 'treecanopy',
        'tc_gap', 'pctpoc', 'pctpov', 'unemplrate', 'dep_ratio', 'dep_perc',
        'linguistic', 'health_nor', 'temp_diff', 'tes', 'holc_grade',
        'child_perc', 'seniorperc', 'ej_disadva', 'rank', 'rankgrpsz',
        '_bld1200', '_veg1200', '_bld1500', '_veg1500', '_bld1800',
        '_veg1800', '_tot1200', '_tot1500', '_tot1800'
    ]

    # Create a shapely Point object. Note: Coordinates are in (longitude, latitude) order.
    point = Point(lon, lat)

    for feature in la_shading_data:
        # Ensure it is a valid GeoJSON Feature with Polygon geometry
        if feature.get('geometry', {}).get('type') == 'Polygon':
            coordinates = feature['geometry']['coordinates']

            # GeoJSON polygons are a list of rings: [exterior, hole1, hole2, ...]
            # Shapely's Polygon constructor takes the shell (exterior) and a list of holes.
            shell = coordinates[0]
            holes = coordinates[1:]

            try:
                # Create a shapely Polygon object
                polygon = Polygon(shell, holes)

                # Check if the point is within or on the boundary of the polygon
                if polygon.contains(point) or polygon.boundary.contains(point):
                    properties = feature.get('properties', {})
                    # Extract the features in the specified order
                    result_features = [properties.get(key) for key in FEATURE_KEYS]
                    return result_features
            except Exception:
                # Skip invalid polygon geometries
                continue

    # Return a list of None values if no containing polygon is found
    return [None] * len(FEATURE_KEYS)

# input_lat, input_lon = 34.0138, -118.4902
# result_1 = get_la_shade_features(input_lat, input_lon)
# result_1
# ALL_FEATURE_NAMES.extend([
#     'lashade_ua_pop', 'lashade_cbg_pop', 'lashade_acs_pop', 'lashade_biome', 'lashade_tc_goal',
#     'lashade_treecanopy', 'lashade_tc_gap', 'lashade_pctpoc', 'lashade_pctpov', 'lashade_unemplrate',
#     'lashade_dep_ratio', 'lashade_dep_perc', 'lashade_linguistic', 'lashade_health_nor', 'lashade_temp_diff',
#     'lashade_tes', 'lashade_holc_grade', 'lashade_child_perc', 'lashade_seniorperc', 'lashade_ej_disadva',
#     'lashade_rank', 'lashade_rankgrpsz', 'lashade_bld1200', 'lashade_veg1200', 'lashade_bld1500',
#     'lashade_veg1500', 'lashade_bld1800', 'lashade_veg1800', 'lashade_tot1200', 'lashade_tot1500',
#     'lashade_tot1800'])

la_venues_data = None
with open('dataset/LA28_venues.geojson', 'r') as file:
    data = json.load(file)
    la_venues_data = data['features']

def calculate_venue_features(input_lat: float, input_lon: float):
    """
    Computes distance and venue features from an input point to venues in the la_venues_data.

    Args:
        input_lat: The latitude of the input point.
        input_lon: The longitude of the input point.

    Returns:
        A list containing:
        [distance to closest venue (km), closest venue Sport/Activity (str), 
         average distance to closest 3 venues (km)].
        Returns [math.nan, "N/A", math.nan] if the la_venues_data is empty or invalid.
    """
    
    input_point: Tuple[float, float] = (input_lat, input_lon)
    all_venues_distances: List[Dict[str, Any]] = []

    if not la_venues_data:
        return [math.nan, "N/A", math.nan]

    try:
        for venue in la_venues_data:
            lon: float = venue['geometry']['coordinates'][0]
            lat: float = venue['geometry']['coordinates'][1]
            venue_point: Tuple[float, float] = (lat, lon)
            
            distance_km: float = geodesic(input_point, venue_point).km
            
            all_venues_distances.append({
                'distance': distance_km,
                'sport': venue['properties']['Sport/Activity']
            })
            
    except KeyError as e:
        print(f"Error processing data item: Missing key {e}")
        return [math.nan, "N/A", math.nan]
    except IndexError as e:
        print(f"Error processing data item: Malformed coordinates {e}")
        return [math.nan, "N/A", math.nan]

    all_venues_distances.sort(key=lambda x: x['distance'])
    
    # Extract features
    # Feature 1: distance to closest venue
    closest_venue_data: Dict[str, Any] = all_venues_distances[0]
    distance_to_closest_venue: float = closest_venue_data['distance']
    
    # Feature 2: closest venue Sport/Activity
    closest_venue_sport: str = closest_venue_data['sport']
    
    # Feature 3: average distance to closest 3 venues
    num_closest_for_avg: int = min(3, len(all_venues_distances))
    closest_distances: List[float] = [
        item['distance'] for item in all_venues_distances[:num_closest_for_avg]
    ]
    
    average_closest_three: float
    if num_closest_for_avg > 0:
        average_closest_three = sum(closest_distances) / num_closest_for_avg
    else:
        average_closest_three = math.nan 

    return [distance_to_closest_venue, closest_venue_sport, average_closest_three]

# test_lat = 34.0522 
# test_lon = -118.2437 
# calculate_venue_features(input_lat, input_lon)

# ALL_FEATURE_NAMES.extend(['dist_to_venue1', 'closest_venue_sport', 'dist_to_venue3'])

la_cva_data = None
with open('dataset/Los_Angeles_County_CVA_Social_Sensitivity_Index.geojson', 'r') as file:
    data = json.load(file)
    la_cva_data = data['features']

def get_la_cva_features(lat: float, lon: float):
    """
    Finds the first GeoJSON Polygon in the global la_cva_data that contains the given
    latitude and longitude point and returns a list of its specified properties.

    NOTE: This function assumes that the global variable la_cva_data is loaded
    with the GeoJSON Feature data.

    Args:
        lat: The latitude of the point.
        lon: The longitude of the point.

    Returns:
        A list of feature values for the first containing polygon, or a
        list of None values if no containing polygon is found. The features
        are returned in the following order:
        
        - Population (int): The population of the census tract (included as CVA Population in prompt, but key is 'Population' in format)
        - Children (float): Percent children 18 and under
        - Older_Adults (float): Percent persons 65 and over
        - Older_Adults_Living_Alone (float): Percent of households in which the householder is 65 and over who and living alone
        - Limited_English (float): Percent limited English speaking households
        - No_High_School_Diploma (float): Percent of persons 25 and older without a high school diploma
        - Female (float): Percent female
        - Female_Householder (float): Percent of family households that have a female householder with no spouse present
        - Disability (float): Percent of civilian noninstitutionalized population with either mental or physical disability
        - No_Health_Insurance (float): Percent of civilian noninstitutionalized population without health insurance
        - Living_in_Group_Quarters (float): Percent of persons living in (either institutionalized or un-institutionalized) group quarters
        - Mobile_Homes (float): Percent of occupied housing units which are mobile homes or "other types of housing"
        - Rent_Burden (float): Percent of renter-occupied housing units where rent is 30% or more of household income
        - Renters (float): Percentage of housing units which are renter-occupied per census tract
        - Median_Income (int): Median household income of census tract
        - Poverty (float): Percent of the population earning below 100% of the federal poverty threshold
        - Households_Without_Vehicle_Acce (float): Percent of households without access to a personal vehicle
        - Outdoor_Workers (float): Percentage of civilian employed population in "Natural resources, construction, and maintenance occupations"
        - Unemployed (float): Percent of the population over the age of 16 that is unemployed and eligible for the labor force
        - Foreign_Born (float): Percent of the total population who was not born in the United States or Puerto Rico
        - No_Internet_Subscription (float): Percent of the population without an internet subscription
        - Voter_Turnout_Rate (float): Percentage of registered voters voting in the general election
        - SoVI_Score (float): Social Vulnerability/Sensitivity score
        - Asthma (float): Age-adjusted rate of emergency department visits for asthma
        - Cardiovascular_Disease (float): Age-adjusted rate of emergency department visits for heart attacks per 10,000
        - Transit_Access (float): % of tract area in HQTA (High Quality Transit Area)
    """
    # Keys for the desired features in the order they should be returned
    # NOTE: The keys must match those in the 'properties' dictionary of the GeoJSON data.
    FEATURE_KEYS = [
        'Population',
        'Children',
        'Older_Adults',
        'Older_Adults_Living_Alone',
        'Limited_English',
        'No_High_School_Diploma',
        'Female',
        'Female_Householder',
        'Disability',
        'No_Health_Insurance',
        'Living_in_Group_Quarters',
        'Mobile_Homes',
        'Rent_Burden',
        'Renters',
        'Median_Income',
        'Poverty',
        'Households_Without_Vehicle_Acce', # Matches without_vehicle_access
        'Outdoor_Workers',
        'Unemployed',
        'Foreign_Born',
        'No_Internet_Subscription',
        'Voter_Turnout_Rate', # Matches voter_turnout
        'SoVI_Score', # Matches svi_score
        'Asthma',
        'Cardiovascular_Disease', # Matches cardiovascular
        'Transit_Access'
    ]
    
    point = Point(lon, lat)

    for feature in la_cva_data:
        # Check if it is a valid GeoJSON Feature with Polygon geometry
        if feature.get('geometry', {}).get('type') == 'Polygon':
            coordinates = feature['geometry']['coordinates']
            try:
                shell = coordinates[0]
                holes = coordinates[1:] if len(coordinates) > 1 else None

                polygon = Polygon(shell, holes)

                if polygon.contains(point) or polygon.boundary.contains(point):
                    properties = feature.get('properties', {})
                    result_features = [properties.get(key) for key in FEATURE_KEYS]
                    return result_features
            except Exception:
                # Skip invalid polygon geometries or other errors during processing
                continue

    # Return a list of None if no containing polygon is found
    return [None] * len(FEATURE_KEYS)

# ALL_FEATURE_NAMES.extend([
#     'cva_population', 'cva_children', 'cva_older_adults', 'cva_older_adults_living_alone', 'cva_limited_english',
#     'cva_no_high_school_diploma', 'cva_female', 'cva_female_householder', 'cva_disability', 'cva_no_health_insurance',
#     'cva_living_in_group_quarters', 'cva_mobile_homes', 'cva_rent_burden', 'cva_renters', 'cva_median_income',
#     'cva_poverty', 'cva_households_without_vehicle_acce', 'cva_outdoor_workers', 'cva_unemployed', 'cva_foreign_born',
#     'cva_no_internet_subscription', 'cva_voter_turnout_rate', 'cva_sovi_score', 'cva_asthma', 'cva_cardiovascular_disease',
#     'cva_transit_access'])

la_air_pm25_data = None
with open('dataset/pm25_la2015.geojson', 'r') as file:
    data = json.load(file)
    la_air_pm25_data = data['features']

def get_pm25_features(lat: float, lon: float):
    """
    Finds the first GeoJSON Feature (Polygon or MultiPolygon) in the dataset 
    that contains the given latitude and longitude point and returns a list of 
    its specified properties.

    Args:
        lat: The latitude of the point.
        lon: The longitude of the point.

    Returns:
        A list of feature values for the first containing feature, or a
        list of None values if no containing feature is found. The features
        are returned in the following order:

        - value (float): PM2.5 value.
        - percentile (float): PM2.5 quality percentile based on the value. e.g. 0.17 = in the lowest 17 percentage, good quality, lower than 83% others 
    """
    FEATURE_KEYS = ['value', 'percentile']

    point = Point(lon, lat)

    for feature in la_air_pm25_data:
        geometry_dict = feature.get('geometry')
        if not geometry_dict:
            continue

        try:
            geom = shape(geometry_dict)

            if geom.contains(point) or geom.boundary.contains(point):
                properties = feature.get('properties', {})
                # Extract the features in the specified order
                result_features = [properties.get(key) for key in FEATURE_KEYS]
                return result_features
        except Exception:
            # Skip invalid geometries or parsing errors
            continue

    return [None] * len(FEATURE_KEYS)

# test_lon, test_lat = -118.004951, 34.04583
# get_pm25_features(input_lat, input_lon)
# ALL_FEATURE_NAMES.extend(['pm25', 'pm25_percentile'])

la_tree_canopy_data = None
with open('dataset/Tree_Canopy_Coverage.geojson', 'r') as file:
    data = json.load(file)
    la_tree_canopy_data = data['features']

def get_canopy_feature(lat: float, lon: float):
    """
    Finds the first GeoJSON Polygon or MultiPolygon in the dataset that contains
    the given latitude and longitude point and returns a list of its specified features.

    Args:
        lat: The latitude of the point.
        lon: The longitude of the point.

    Returns:
        A list containing the 'tree_pw' feature value:

        - tree_pw (float | None): Percentage of Land With Tree Canopy Coverage
          (Weighted by Population Size). Returns 0.0 if the original property is None.
          Returns NaN (represented by None in Python's standard types) if the
          point falls outside all polygons.
    """
    # Key for the desired feature
    FEATURE_KEYS = ['Tree_PW']

    point = Point(lon, lat)

    for feature in la_tree_canopy_data:
        geometry = feature.get('geometry')

        if geometry and geometry.get('type') in ['Polygon', 'MultiPolygon']:
            try:
                geom_shape = shape(geometry)

                if geom_shape.contains(point) or geom_shape.boundary.contains(point):
                    properties = feature.get('properties', {})
                    tree_pw_value = properties.get(FEATURE_KEYS[0])

                    if tree_pw_value is None:
                        return [0.0]

                    try:
                        return [float(tree_pw_value)]
                    except (ValueError, TypeError):
                        return [0.0] # Treat non-numeric valid values as the 'None' case

            except Exception:
                continue
    return [None]

# test_lon, test_lat = -117.948806794898, 34.0894069176852
# get_canopy_feature(input_lat, input_lon)
# ALL_FEATURE_NAMES.extend(['tree_percent_w'])

la_uhi_data = None
with open('dataset/uhi_la.geojson', 'r') as file:
    data = json.load(file)
    la_uhi_data = data['features']

def get_uhi_features(lat: float, lon: float):
    """
    Finds the first GeoJSON Feature (Polygon or MultiPolygon) in the global la_uhi_data
    that contains the given latitude and longitude point and returns a list of 
    its specified Urban Heat Index (UHI) properties.

    Args:
        lat: The latitude of the point (e.g., 34.06).
        lon: The longitude of the point (e.g., -117.85).

    Returns:
        A list of UHI feature values for the first containing feature, or a
        list of None values if no containing feature is found. The features
        are returned in the following order:

        - value (float or None): The Urban Heat Index value (e.g., temperature difference).
        - percentile (float or None): The percentile rank of the UHI value across the entire dataset.
                                      (e.g., 0.95 means the value is higher than 95% of all other values, indicating high heat).
    """
    FEATURE_KEYS = ['value', 'percentile']

    point = Point(lon, lat) 

    if not la_uhi_data:
        print("Error: la_uhi_data is empty or not defined.")
        return [None] * len(FEATURE_KEYS)

    for feature in la_uhi_data:
        geometry_dict = feature.get('geometry')
        if not geometry_dict:
            continue
        try:
            geom = shape(geometry_dict)
            if geom.contains(point) or geom.boundary.contains(point):
                properties = feature.get('properties', {})
                result_features = [properties.get(key) for key in FEATURE_KEYS]
                return result_features
        except Exception as e:
            print(f"Skipping invalid geometry due to error: {e}")
            continue

    return [None] * len(FEATURE_KEYS)

# test_lon, test_lat = -117.948806794898, 34.0894069176852
# get_uhi_features(input_lat, input_lon)
# ALL_FEATURE_NAMES.extend(['urban_heat_idx', 'urban_heat_idx_percentile'])
vacant_planting_park_data = None
with open('dataset/vacant_trees_park.geojson', 'r') as file:
    data = json.load(file)
    vacant_planting_park_data = data['features']

def calculate_vacant_park_features(input_lat: float, input_lon: float):
    """
    Computes distance features (nearest and average of 3 closest) 
    from an input point to vacant sites within the global vacant_planting_park_data.

    Vacant sites are filtered based on the 'Species' property containing 'Vacant Site'.
    Distances are calculated in kilometers (km).

    Args:
        input_lat: The latitude of the input point.
        input_lon: The longitude of the input point.

    Returns:
        A list containing two floats: 
        [distance to nearest vacant site (km), average distance to 3 closest vacant sites (km)].
        Returns [math.nan, math.nan] if no vacant sites are found.
    """
    
    input_point = (input_lat, input_lon)
    all_distances = []

    try:
        for site in vacant_planting_park_data:
            # Extract coordinates from the 'geometry' field. 
            # GeoJSON standard is [longitude, latitude].
            lon, lat = site['geometry']['coordinates']
            facility_point = (lat, lon)
            
            # Calculate distance using geodesic (great-circle distance) in kilometers
            distance_km = geodesic(input_point, facility_point).km
            all_distances.append(distance_km)
            
    except KeyError as e:
        print(f"Error processing data item: Missing required key {e}")
        return [math.nan, math.nan]
    
    all_distances.sort()
    
    nearest_distance = all_distances[0]
    
    num_points = len(all_distances)
    
    if num_points >= 3:
        closest_points_for_avg = all_distances[:3]
        average_closest_three = sum(closest_points_for_avg) / 3
    else:
        average_closest_three = sum(all_distances) / num_points

    return [nearest_distance, average_closest_three]


# test_lat = 34.045
# test_lon = -118.417

# calculate_vacant_park_features(input_lat, input_lon)
# ALL_FEATURE_NAMES.extend(['dist_to_vacant_park_1', 'dist_to_vacant_park_3'])
vacant_planting_street_data = None
with open('dataset/vacant_trees_street.geojson', 'r') as file:
    data = json.load(file)
    vacant_planting_street_data = data['features']

def calculate_vacant_street_features(input_lat: float, input_lon: float):
    """
    Computes distance features (nearest and average of 3 closest) 
    from an input point to vacant sites within the global vacant_planting_street_data.

    Vacant sites are filtered based on the 'Species' property containing 'Vacant Site'.
    Distances are calculated in kilometers (km).

    Args:
        input_lat: The latitude of the input point.
        input_lon: The longitude of the input point.

    Returns:
        A list containing two floats: 
        [distance to nearest vacant site (km), average distance to 3 closest vacant sites (km)].
        Returns [math.nan, math.nan] if no vacant sites are found.
    """
    
    input_point = (input_lat, input_lon)
    all_distances = []

    try:
        for site in vacant_planting_street_data:
            # Extract coordinates from the 'geometry' field. 
            # GeoJSON standard is [longitude, latitude].
            lon, lat = site['geometry']['coordinates']
            facility_point = (lat, lon)
            
            # Calculate distance using geodesic (great-circle distance) in kilometers
            distance_km = geodesic(input_point, facility_point).km
            all_distances.append(distance_km)
            
    except KeyError as e:
        print(f"Error processing data item: Missing required key {e}")
        return [math.nan, math.nan]
    
    all_distances.sort()
    
    nearest_distance = all_distances[0]
    
    num_points = len(all_distances)
    
    if num_points >= 3:
        closest_points_for_avg = all_distances[:3]
        average_closest_three = sum(closest_points_for_avg) / 3
    else:
        average_closest_three = sum(all_distances) / num_points

    return [nearest_distance, average_closest_three]


# test_lat = 34.045
# test_lon = -118.417

# calculate_vacant_street_features(input_lat, input_lon)
# ALL_FEATURE_NAMES.extend(['dist_to_vacant_street_1', 'dist_to_vacant_street_3'])
def extract_features(input_lat, input_lon):
    """
    Extract features of the selected point given (latitude, longitude) from all 14 datasets

    Args:
        input_lat: The latitude of the input point.
        input_lon: The longitude of the input point.

    Returns:
        A list containing all features extracted from 14 datasets, which is around 82 features
    """
    result = []
    result.extend(calculate_ac_center_features(input_lat, input_lon))
    result.extend(calculate_water_station_features(input_lat, input_lon))
    result.extend(calculate_bus_line_features(input_lat, input_lon))
    result.extend(calculate_bus_stop_features(input_lat, input_lon))
    result.extend(calculate_metro_line_features(input_lat, input_lon))
    result.extend(calculate_metro_stop_features(input_lat, input_lon))
    result.extend(get_la_shade_features(input_lat, input_lon))
    result.extend(calculate_venue_features(input_lat, input_lon))
    result.extend(get_la_cva_features(input_lat, input_lon))
    result.extend(get_pm25_features(input_lat, input_lon))
    result.extend(get_canopy_feature(input_lat, input_lon))
    result.extend(get_uhi_features(input_lat, input_lon))
    result.extend(calculate_vacant_park_features(input_lat, input_lon))
    result.extend(calculate_vacant_street_features(input_lat, input_lon))
    return result

# result = extract_features(34.0138, -118.4902)
# print(len(result))
# print(len(FEATURE_NAMES)) # make sure result and feature names align
# print(result)

def process_point(point):
    """
    Wrapper function to extract features for a single point.
    This function will be called in parallel by multiple processes.
    """
    lat, lon = point
    features = extract_features(lat, lon)
    return [lat, lon] + features

if __name__ == '__main__':
    # Get number of CPUs available
    num_cpus = cpu_count()
    print(f"Using {num_cpus} CPU cores for parallel processing")
    print(f"Processing {len(points)} points...")
    
    # Create a pool of worker processes
    with Pool(processes=num_cpus) as pool:
        # Process all points in parallel with progress bar
        all_features = list(tqdm(
            pool.imap(process_point, points),
            total=len(points),
            desc="Extracting features"
        ))
    
    # Create DataFrame with proper column names
    column_names = ['latitude', 'longitude'] + FEATURE_NAMES
    df = pd.DataFrame(all_features, columns=column_names)
    
    # Save to CSV
    output_filename = 'la_coverage_points_features.csv'
    df.to_csv(output_filename, index=False)
    
    print(f"\nDataset saved to {output_filename}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())