# Groundwater Depth Analysis – Detailed Documentation

This module processes groundwater depth data from the Atal Jal dataset and aggregates it at the subdistrict level for Rajasthan. It handles pre-monsoon and post-monsoon measurements from 2015-2022 and generates spatial aggregations for forest management applications.

---

## 1. What the script does

### Data Processing Pipeline

1. **Data Loading**:
   - Loads groundwater depth CSV data from Atal Jal dataset
   - Imports subdistrict boundaries from shapefile
   - Sets up DuckDB with spatial extensions

2. **Data Transformation**:
   - Unpivots groundwater measurements from wide to long format
   - Extracts year and monsoon period from column names
   - Converts ground level measurements to numeric values
   - Handles missing values and data quality issues

3. **Spatial Aggregation**:
   - Performs spatial join between groundwater points and subdistrict boundaries
   - Calculates average groundwater depth per subdistrict per period
   - Filters for Rajasthan state and range-type regions only

4. **Output Generation**:
   - Exports aggregated data as GeoJSON and CSV files
   - Creates separate files for each year and monsoon period
   - Generates 32 output files (16 years × 2 periods)

---

## 2. Data Sources

### Input Data
- **Groundwater Data**: `Atal_Jal_Disclosed_Ground_Water_Level-2015-2022-utf8.csv`
  - Contains well locations, depths, and measurements
  - Covers pre-monsoon and post-monsoon periods
  - Includes metadata: well type, source, aquifer information

- **Spatial Boundaries**: `ranges-withcode-geom.shp`
  - Subdistrict/range boundaries for Rajasthan
  - Includes administrative codes and names
  - Used for spatial aggregation

### Output Data
- **GeoJSON Files**: `groundwater_aggregated_{year}_{period}.geojson`
- **CSV Files**: `groundwater_aggregated_{year}_{period}.csv`

---

## 3. Data Processing Details

### Groundwater Data Structure
The script processes columns in the format:
- `premonsoon_2015_meters_below_ground_level`
- `postmonsoon_2015_meters_below_ground_level`
- ... (continues through 2022)

### Data Quality Handling
- **Missing Values**: `'na'` and `'dry'` values converted to NULL
- **Dry Wells**: `'dry'` converted to 0 meters depth
- **Invalid Data**: Non-numeric values filtered out
- **Spatial Filtering**: Only Rajasthan state data processed

### Aggregation Logic
```sql
SELECT 
    name, 
    {year} as year, 
    '{period}' as period, 
    AVG(groundwater_depth) AS AverageGroundwaterDepth,
    any_value(geom) as geom
FROM groundwater g 
JOIN regions s ON ST_Within(g.geom, s.geom) 
WHERE s.type = 'range' 
    AND g.groundwater_depth IS NOT NULL 
    AND g.measurement_year = {year} 
    AND g.monsoon_period = '{period}'
GROUP BY name
```

---

## 4. Configuration

### Database Setup
```python
db_path = "dfhms.db"
table_name = "regions"
output_table = "canopy_density_data"
```

### Data Paths
```python
groundwater_data_path = 'data/groundwater_depth/Atal_Jal_Disclosed_Ground_Water_Level-2015-2022-utf8.csv'
subdistricts_path = 'data/gis/ranges-withcode-geom.shp'
```

### Processing Periods
The script processes 16 periods (2015-2022, pre/post monsoon):
```python
periods = [
    (2015, 'premonsoon'), (2015, 'postmonsoon'),
    (2016, 'premonsoon'), (2016, 'postmonsoon'),
    # ... continues through 2022
]
```

---

## 5. Dependencies

- `duckdb` - Database operations and spatial queries
- `geopandas` - Geospatial data handling
- `pandas` - Data manipulation
- `zipfile` - Data extraction (if needed)
- `requests` - Data downloading (if needed)

---

## 6. Usage Instructions

### Prerequisites
1. Install required dependencies
2. Ensure data files are in correct locations
3. Set up DuckDB with spatial extension

### Running the Script
```bash
cd forest-stack/common/scripts/groundwater_depth/
python groundwater_depth.py
```

### Expected Output
- 32 GeoJSON files with groundwater depth aggregations
- 32 CSV files with tabular data
- Files named: `groundwater_aggregated_{year}_{period}.{ext}`

---

## 7. Output Data Structure

### GeoJSON Structure
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "name": "Subdistrict Name",
        "year": 2020,
        "period": "premonsoon",
        "AverageGroundwaterDepth": 15.5
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [...]
      }
    }
  ]
}
```

### CSV Structure
| name | year | period | AverageGroundwaterDepth |
|------|------|--------|------------------------|
| Subdistrict A | 2020 | premonsoon | 15.5 |
| Subdistrict B | 2020 | premonsoon | 22.3 |

---

## 8. Performance Considerations

### Memory Usage
- DuckDB handles large datasets efficiently
- Spatial joins are optimized for performance
- Consider processing in batches for very large datasets

### Processing Time
- Depends on data size and system resources
- Spatial operations are the most time-consuming
- Parallel processing could be implemented for large datasets

---

## 9. Data Quality Validation

### Validation Checks
- Verify all periods are processed
- Check for missing subdistricts
- Validate groundwater depth ranges (typically 0-100 meters)
- Ensure spatial accuracy of point-in-polygon operations

### Common Issues
- **Missing Data**: Some periods may have limited measurements
- **Spatial Accuracy**: Well locations may not align perfectly with boundaries
- **Data Gaps**: Some subdistricts may lack measurements for certain periods

---

## 10. Integration with Forest-Stack

This module provides groundwater depth data for:
- **Forest Health Monitoring**: Water stress indicators
- **Climate Analysis**: Drought impact assessment
- **Ecosystem Services**: Water availability for forest ecosystems
- **Management Planning**: Groundwater-dependent forest areas

---

## 11. Troubleshooting

### Common Issues

1. **File Not Found**: Ensure data files are in correct paths
2. **Database Errors**: Check DuckDB installation and spatial extension
3. **Memory Issues**: Process data in smaller batches
4. **Spatial Join Failures**: Verify coordinate reference systems

### Debug Tips

- Check data file formats and encodings
- Validate spatial data integrity
- Monitor memory usage during processing
- Test with smaller date ranges first

---

## 12. Future Enhancements

### Potential Improvements
- Add data quality metrics and validation
- Implement parallel processing for large datasets
- Add temporal trend analysis
- Include uncertainty estimates
- Support for additional data sources

### Integration Opportunities
- Connect with real-time groundwater monitoring
- Link with climate data for correlation analysis
- Integrate with forest health indicators
- Add visualization and reporting capabilities
