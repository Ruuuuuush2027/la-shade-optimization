# Dataset Summary

15 GeoJSON files with Los Angeles geographic data covering infrastructure, environmental conditions, social vulnerability, and urban planning datasets.

**Coordinate System:** CRS84 (WGS84) or EPSG:4269 | **Format:** `[longitude, latitude]`

---

## 1. Cooling_Heating_Centers_in_Los_Angeles.geojson

Public cooling and heating center facilities across Los Angeles including libraries, recreation centers, and senior centers (~30 locations).

```json
{
  "type": "Feature",
  "properties": {
    "OBJECTID": 64,
    "GlobalID": "19c09964-74b3-47b8-b213-6b2415619201",
    "FacilityName": "Canoga Park Senior Citizen Center",
    "IndoorOutdoor": "Indoor",
    "Address": "7326 Jordan Ave, Canoga Park, CA, 91303, USA",
    "Bureau": "Valley",
    "CouncilDistrict": "03",
    "AltFacPhone": " "
  },
  "geometry": {
    "type": "Point",
    "coordinates": [-118.602643087136997, 34.203724738176497]
  }
}
```

---

## 2. Hydration_Stations_August_2022.geojson

Public water hydration stations installed throughout LA at airports, parks, and public buildings (~140 stations).

```json
{
  "type": "Feature",
  "properties": {
    "HS_ID": "HS00070",
    "Facility": "Los Angeles International Airport",
    "Address": "1 World Way",
    "Address_2": "T1 - Departure, Near CPK",
    "City": "Los Angeles",
    "Zip": "90045",
    "Installation": "2021-06-30T00:00:00Z",
    "Year": 2021,
    "Indoor_Outdoor": "Indoor",
    "Partners": "LAWA",
    "Council_District": "CD 11",
    "DAC": "TRUE",
    "ObjectId": 1
  },
  "geometry": {
    "type": "Point",
    "coordinates": [-118.398162004496001, 33.944686019909199]
  }
}
```

---

## 3. LA_Bus_Lines.geojson

LA Metro bus route line geometries showing the path of each bus route.

```json
{
  "type": "Feature",
  "properties": {
    "route_number": "720",
    "route_name": "Wilshire/Whittier"
  },
  "geometry": {
    "type": "LineString",
    "coordinates": [
      [-118.2876, 34.0614],
      [-118.2889, 34.0615],
      [-118.2902, 34.0616]
    ]
  }
}
```

---

## 4. LA_Metro_Bus_Stops_06_2025.geojson

All LA Metro bus stop locations with service line information (11,848 stops).

```json
{
  "type": "Feature",
  "properties": {
    "STOPNUM": 1,
    "STOPNAME": "Paramount / Slauson",
    "LAT": 33.973248,
    "LONG": -118.113113,
    "LINE_DIR1": "265-S",
    "LINE_DIR2": null,
    "LINE_DIR3": 0,
    "LINE_DIR4": 0,
    "LINE_DIR5": 0,
    "LINE_DIR6": 0,
    "LINE_DIR7": 0,
    "LINE_DIR8": 0,
    "LINE_DIR9": 0,
    "LINE_DIR10": 0,
    "LINE_DIR11": 0,
    "LINE_DIR12": 0,
    "LINE_DIR13": 0,
    "LINE_DIR14": 0
  },
  "geometry": {
    "type": "Point",
    "coordinates": [-118.113113, 33.973248]
  }
}
```

---

## 5. LA_Metro_Lines.geojson

LA Metro rail line route geometries for A, B, C, D, E, and K lines (~6 lines).

```json
{
  "type": "Feature",
  "properties": {
    "line_name": "A Line (Blue)",
    "line_code": "A"
  },
  "geometry": {
    "type": "LineString",
    "coordinates": [
      [-118.1929, 33.7681],
      [-118.1937, 33.7723],
      [-118.1894, 33.7818]
    ]
  }
}
```

---

## 6. LA_Metro_Stations.geojson

LA Metro rail station point locations for all metro lines (~107 stations).

```json
{
  "type": "Feature",
  "id": 0,
  "geometry": {
    "type": "Point",
    "coordinates": [-118.192921, 33.768070999999999]
  },
  "properties": {
    "FID": 0,
    "STOP_ID": "80101",
    "STOP_NAME": "Downtown Long Beach Station",
    "STOP_LAT": 33.768070999999999,
    "STOP_LON": -118.192921
  }
}
```

---

## 7. LA_shade.geojson

Census tract-level data combining tree canopy coverage with social vulnerability and demographic indicators (~6,550 tracts).

```json
{
  "type": "Feature",
  "properties": {
    "GEOID": "060371011101",
    "place": "Los Angeles",
    "county": "Los Angeles County",
    "cbg_pop": 2050,
    "treecanopy": 0.2315,
    "tc_gap": 0.0685,
    "pctpoc": 0.42822085889570499,
    "pctpov": 0.22944785276073601,
    "unemplrate": 0.05331599479844,
    "temp_diff": -3.14,
    "holc_grade": "C",
    "ej_disadva": "Yes",
    "rank": 413.0
  },
  "geometry": {
    "type": "Polygon",
    "coordinates": [
      [
        [-118.287333, 34.255904],
        [-118.288618, 34.255912],
        [-118.289801, 34.255919],
        [-118.287333, 34.255904]
      ]
    ]
  }
}
```

---

## 8. LA28_venues.geojson

Los Angeles 2028 Olympic and Paralympic venue locations (~38 venues).

```json
{
  "type": "Feature",
  "properties": {
    "Sport/Activity": "Basketball (3x3)",
    "LA28 Venue Region": "Valley Sports Park",
    "LA28 Venue Name": "Sepulveda Basin Recreation Area",
    "Venue Common Name": "Sepulveda Basin River Recreation Zone",
    "City": "Encino, CA",
    "Notes": null,
    "x": -118.4898898,
    "y": 34.17663423
  },
  "geometry": {
    "type": "Point",
    "coordinates": [-118.4898898, 34.17663423]
  }
}
```

---

## 9. Los_Angeles_County_CVA_Social_Sensitivity_Index.geojson

Climate Vulnerability Assessment (CVA) social sensitivity scores by census tract with comprehensive demographic and vulnerability metrics (~2,333 tracts).

```json
{
  "type": "Feature",
  "properties": {
    "Census_Tract": "Census Tract 5709.02",
    "County": "Los Angeles County",
    "CSA_Label": "City of Lakewood",
    "Population": 3765,
    "Children": 27.8,
    "Older_Adults": 14.0,
    "Limited_English": 0.0,
    "No_High_School_Diploma": 13.9,
    "Asthma": 41.2,
    "Disability": 7.4,
    "No_Health_Insurance": 6.5,
    "Rent_Burden": 39.2,
    "Median_Income": 87377,
    "Poverty": 1.5,
    "Unemployed": 5.6,
    "SoVI_Score": 2.25,
    "SoVI_Thirds": 3,
    ...
  },
  "geometry": {
    "type": "Polygon",
    "coordinates": [
      [
        [-118.12509, 33.860319],
        [-118.122865, 33.860318],
        [-118.11706, 33.858277],
        [-118.12509, 33.860319]
      ]
    ]
  }
}
```

---

## 10. pm25_la2015.geojson

PM2.5 fine particulate matter air quality data from 2015 by census tract (~2,331 tracts).

```json
{
  "type": "Feature",
  "properties": {
    "geoid": "06037408301",
    "name": "4083.01",
    "population": 5569,
    "value": 12.07676548,
    "percentile": 0.17201540436457,
    "numerator": null,
    "denominator": null,
    "se": null
  },
  "geometry": {
    "type": "MultiPolygon",
    "coordinates": [
      [
        [
          [-118.009588, 34.051625],
          [-118.00783, 34.056612],
          [-118.006322, 34.055671],
          [-118.009588, 34.051625]
        ]
      ]
    ]
  }
}
```

---

## 11. Tree_Canopy_Coverage.geojson

Tree canopy coverage data, a single record of all coordinates.

```json
{
  "summary": "Partial GeoJSON FeatureCollection Structure",
  "top_level_type": "FeatureCollection",
  "coordinate_reference_system": {
    "type": "name",
    "name": "EPSG:4326"
  },
  "number_of_features_started": 1,
  "feature_properties_summary": [
    {
      "id": 1,
      "geometry_type": "Polygon",
      "coordinates": [
      [
        [
          [-118.009588, 34.051625],
          [-118.00783, 34.056612],
          [-118.006322, 34.055671],
          [-118.009588, 34.051625],
          ...
        ]
      ]
    ]
    }
  ]
}
```

---

## 12. uhi_la.geojson

Urban Heat Island (UHI) intensity data showing elevated temperature areas by census tract (~2,197 tracts).

```json
{
  "type": "Feature",
  "properties": {
    "geoid": "06037403500",
    "name": "4035",
    "population": 1835.0,
    "value": 20422.1,
    "percentile": 0.84319526627218933,
    "numerator": null,
    "denominator": null,
    "se": null
  },
  "geometry": {
    "type": "MultiPolygon",
    "coordinates": [
      [
        [
          [-117.872297, 34.071704],
          [-117.868118, 34.071438],
          [-117.866185, 34.070908],
          [-117.872297, 34.071704]
        ]
      ]
    ]
  }
}
```

---

## 13. vacant_trees_park.geojson

Vacant tree planting site locations in LA parks (12,261 sites). Where one can and plan to plant trees

```json
{
  "type": "Feature",
  "properties": {
    "Park Name": "Griffith Park",
    "Latitude": 34.121380086239,
    "Longitude": -118.274161437467,
    "Species": "Vacant Site-Large (RAP) (Vacant Site-Large)",
    "Diameter": 0,
    "Heritage Tree": "No",
    "Historic Tree": "No",
    "Protected Tree": "No"
  },
  "geometry": {
    "type": "Point",
    "coordinates": [-118.274161437467001, 34.121380086239]
  }
}
```

---

## 14. vacant_trees_street.geojson

Vacant tree planting site locations along LA streets (large file >50MB with many thousands of sites). Where one can and plan to plant trees

```json
{
  "type": "Feature",
  "properties": {
    "Street Name": "Example St",
    "Latitude": 34.0522,
    "Longitude": -118.2437,
    "Species": "Vacant Site-Medium",
    "Diameter": 0,
    "Heritage Tree": "No",
    "Historic Tree": "No",
    "Protected Tree": "No"
  },
  "geometry": {
    "type": "Point",
    "coordinates": [-118.2437, 34.0522]
  }
}
```

---

## Dataset Categories

### Infrastructure & Transportation
- **LA_Bus_Lines.geojson** - Bus routes
- **LA_Metro_Lines.geojson** - Rail routes
- **LA_Metro_Bus_Stops_06_2025.geojson** - Bus stops
- **LA_Metro_Stations.geojson** - Rail stations

### Public Facilities
- **Cooling_Heating_Centers_in_Los_Angeles.geojson** - Climate relief centers
- **Hydration_Stations_August_2022.geojson** - Water stations
- **LA28_venues.geojson** - Olympic venues

### Environmental Data
- **pm25_la2015.geojson** - Air quality (PM2.5)
- **uhi_la.geojson** - Urban heat islands
- **Tree_Canopy_Coverage.geojson** - Tree canopy (empty)
- **LA_shade.geojson** - Tree canopy + social data
- **vacant_trees_park.geojson** - Park planting sites
- **vacant_trees_street.geojson** - Street planting sites

### Social & Demographics
- **Los_Angeles_County_CVA_Social_Sensitivity_Index.geojson** - Vulnerability assessment
- **LA_shade.geojson** - Combined environmental and social indicators

---

## File Format Notes

- **Compact Format:** Most files have each feature on a single line (minimal whitespace)
- **Pretty-Printed:** LA_Bus_Lines and LA_Metro_Lines use indented formatting
- **Ultra-Compact:** LA_Metro_Stations has entire features array on one line
- **Large Files:** vacant_trees_street.geojson is >50MB

## Common Properties

All datasets follow the GeoJSON standard:
- `type`: "Feature" or "FeatureCollection"
- `properties`: Object with dataset-specific attributes
- `geometry`: Spatial data (Point, LineString, Polygon, or MultiPolygon)
- `coordinates`: Array in `[longitude, latitude]` format

## Data Source Links
1. cooling heating center
   1. Cooling_Heating_Centers_in_Los_Angeles.geojson
   2. https://www.arcgis.com/home/item.html?id=1c245346797a4d259b23e012cdb49b06
2. hydration station
   1. Hydration_Stations_August_2022.geojson
   2. https://visionzero.geohub.lacity.org/datasets/hydration-stations-august-2022/explore
3. transportation
   1. LA_Bus_Lines.geojson, LA_Metro_Bus_Stops_06_2025.geojson, LA_Metro_Lines.geojson, LA_Metro_Stations.geojson
   2. https://developer.metro.net/gis-data/
4. LA Shade
   1. LA_shade.geojson
   2. https://innovation.luskin.ucla.edu/how-to-use-the-shade-data-map-in-tree-equity-score/
5. olympic venues
   1. A28_venues.geojson
   2. https://geohub.lacity.org/content/f89cbfe086f445a2baac9636f4a7ea1e
6. Climate Vulnerability Assessment
   1. Los_Angeles_County_CVA_Social_Sensitivity_Index.geojson
   2. https://www.arcgis.com/home/item.html?id=5a7ed763b4c34853b8e057b8c5b618df
7. PM 2.5 Air Quality
   1. pm25_la2015.geojson
   2. https://oehha.ca.gov/calenviroscreen/download-data
8. Tree Canopy Coverage
   1. Tree_Canopy_Coverage.geojson
   2. https://data.lacounty.gov/datasets/lacounty::tree-canopy-coverage/about
9. Urban Heat Island (UHI)
   1. uhi_la.geojson
   2. https://calepa.ca.gov/climate/urban-heat-island-index-for-california/
10. Vacant Tree Planting site
    1. vacant_trees_park.geojson, vacant_trees_street.geojson
    2. https://losangelesca.treekeepersoftware.com/index.cfm?deviceWidth=2560