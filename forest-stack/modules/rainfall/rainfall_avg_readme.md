# Rainfall Average Processing – Detailed Documentation

This module processes daily rainfall data from the India Meteorological Department (IMD) and generates area-weighted averages for forest management regions. It handles both NetCDF files and real-time downloads, with comprehensive caching and parallel processing for optimal performance.

---

## 1. What the script does

### Data Processing Pipeline

1. **Data Acquisition**:
   - Downloads daily rainfall data from IMD website
   - Supports both NetCDF files and real-time downloads
   - Implements intelligent caching system

2. **Data Processing**:
   - Converts IMD grid data to xarray datasets
   - Handles 0.25° × 0.25° spatial resolution
   - Processes data for India-wide coverage (66.5°E-100°E, 6.5°N-38.5°N)

3. **Spatial Aggregation**:
   - Performs spatial join with forest management regions
   - Calculates area-weighted rainfall averages
   - Generates fortnightly aggregations (fn1: 1-15, fn2: 16-end)

4. **Database Integration**:
   - Creates DuckDB tables with spatial indexing
   - Generates SQL insert statements for PostgreSQL
   - Provides comprehensive data validation

---

## 2. Data Sources

### IMD Rainfall Data
- **Source**: India Meteorological Department (IMD)
- **Resolution**: 0.25° × 0.25° (approximately 25km × 25km)
- **Coverage**: India (66.5°E-100°E, 6.5°N-38.5°N)
- **Format**: Binary GRD files or NetCDF
- **Temporal**: Daily data from 1901-present

### Data Access
- **Real-time**: IMD website downloads
- **Archived**: NetCDF files for bulk processing
- **Caching**: Local storage for performance

---

## 3. Core Classes and Functions

### `IMDDataProcessor` Class

#### Initialization
```python
processor = IMDDataProcessor(cache_dir="./data/IMD", log_dir="logs")
```

#### Key Methods

##### `download_daily_data(date_val, timeout=15)`
Downloads daily rainfall data for a specific date.

**Parameters**:
- `date_val`: Date object for data retrieval
- `timeout`: Request timeout in seconds

**Process**:
1. Checks for cached data first
2. Attempts NetCDF file loading
3. Falls back to IMD website download
4. Caches results for future use

##### `process_rainfall_data(data, date_val)`
Processes raw rainfall data into xarray dataset.

**Parameters**:
- `data`: Raw binary data from IMD
- `date_val`: Date for temporal coordinate

**Returns**:
- xarray Dataset with proper coordinates and attributes

##### `create_duckdb_tables(conn, dataset)`
Creates and populates DuckDB tables with rainfall data.

**Process**:
1. Converts xarray to pandas DataFrame
2. Performs spatial join with regions
3. Calculates area-weighted averages
4. Creates spatial indexes

---

## 4. Data Processing Details

### Grid Configuration
```python
grid_coords = {
    'rainfall': {
        'lon': np.linspace(66.5, 100, num=135),    # 135 longitude points
        'lat': np.linspace(6.5, 38.5, num=129),    # 129 latitude points
        'shape': (129, 135)                         # Grid dimensions
    }
}
```

### Data Conversion Process
1. **Binary to Array**: Converts GRD files to numpy arrays
2. **Reshaping**: Reshapes to grid dimensions (129, 135)
3. **Quality Control**: Filters negative values (sets to NaN)
4. **xarray Creation**: Creates properly structured dataset

### Spatial Processing
```sql
WITH spatial_join AS (
    SELECT
        rainfall as value,
        fortnight,
        time as date,
        s.code as range_code
    FROM rainfall_data r
    JOIN regions2 s
    ON ST_Contains(s.geom::GEOMETRY, ST_Point(lon, lat))
    WHERE s.type = 'range'
),
aggregated_data AS (
    SELECT
        range_code,
        fortnight,
        date,
        AVG(value) as avg_value,
        COUNT(*) as point_count
    FROM spatial_join
    GROUP BY range_code, fortnight, date
)
```

---

## 5. Configuration

### Directory Structure
```
data/
├── IMD/                    # Cached rainfall data
├── logs/                   # Processing logs
└── rainfall_data.parquet   # Processed data cache
```

### Database Configuration
```python
db_path = 'dfhms.db'
conn = duckdb.connect(db_path)
conn.execute("INSTALL spatial;")
conn.execute("LOAD spatial;")
```

### Processing Parameters
```python
# Date range for processing
start_date = datetime(2013, 1, 1)
end_date = datetime(2024, 11, 30)

# Grid configuration
grid_resolution = 0.25  # degrees
grid_points = (129, 135)  # lat, lon
```

---

## 6. Caching System

### Cache Strategy
- **File-based Caching**: Individual daily files cached locally
- **Parquet Storage**: Processed data stored in Parquet format
- **Intelligent Loading**: Checks cache before downloading

### Cache Files
```
data/IMD/
├── rain_ind0.25_13_01_01.grd
├── rain_ind0.25_13_01_02.grd
└── ...
```

### Cache Benefits
- **Performance**: Avoids redundant downloads
- **Reliability**: Works offline with cached data
- **Efficiency**: Reduces bandwidth usage

---

## 7. Output Data Structure

### Database Tables

#### `rainfall_data`
| lat | lon | prcp | time | fortnight |
|-----|-----|------|------|-----------|
| 6.5 | 66.5 | 0.0 | 2023-01-01 | 1 |
| 6.5 | 66.75 | 2.5 | 2023-01-01 | 1 |

#### `rainfall_data_agg`
| range_code | value | measurement_date | start_date | end_date | label | point_count |
|------------|-------|------------------|------------|----------|-------|-------------|
| RJ001 | 15.5 | 2023-01-01 | 2023-01-01 | 2023-01-15 | average_rainfall_2023_jan_fn1 | 45 |

### SQL Insert Generation
```sql
WITH region_codes AS (
    SELECT DISTINCT id as region_id, code
    FROM public.regions 
    WHERE code IN ({codes}) AND deleted_at IS NULL
),
dataset_snapshots AS (
    SELECT dps.id as snapshot_id, dps.label
    FROM public.data_product_snapshots dps
    JOIN public.data_products dp ON dp.id = dps.data_product_id
    WHERE dp.name = 'Average rainfall'
),
raw_data(region_code, snapshot_label, value) AS (
    VALUES {values}
)
INSERT INTO public.data_product_snapshots_data
(data_product_snapshot_id, region_id, value, created_at, updated_at)
SELECT gs.snapshot_id, rc.region_id, to_jsonb(rd.value), NOW(), NOW()
FROM raw_data rd
JOIN region_codes rc ON rc.code = rd.region_code
JOIN dataset_snapshots gs ON gs.label = rd.snapshot_label;
```

---

## 8. Performance Optimization

### Parallel Processing
- **Download**: ThreadPoolExecutor for I/O operations
- **Data Processing**: Concurrent processing of multiple dates
- **Memory Management**: Efficient data structures

### Memory Management
```python
# Use Polars for efficient data processing
df = pl.from_pandas(df)
df = df.with_columns([
    pl.col("lat").cast(pl.Float64),
    pl.col("lon").cast(pl.Float64),
    pl.col("prcp").cast(pl.Float64).alias("rainfall")
])
```

### Database Optimization
```sql
-- Create spatial indexes
CREATE INDEX IF NOT EXISTS idx_rainfall_code ON rainfall_data_agg(range_code);
CREATE INDEX IF NOT EXISTS idx_rainfall_date ON rainfall_data_agg(measurement_date);
CREATE INDEX IF NOT EXISTS idx_rainfall_start ON rainfall_data_agg(start_date);
CREATE INDEX IF NOT EXISTS idx_rainfall_end ON rainfall_data_agg(end_date);
```

---

## 9. Dependencies

### Core Libraries
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `xarray` - Multi-dimensional data arrays
- `duckdb` - Database operations
- `geopandas` - Geospatial data handling

### Additional Packages
- `polars` - High-performance data processing
- `requests` - HTTP requests for data download
- `tqdm` - Progress tracking
- `concurrent.futures` - Parallel processing
- `pathlib` - File path handling

### System Requirements
- Python 3.8+
- DuckDB with spatial extension
- Sufficient disk space for caching
- Stable internet connection for downloads

---

## 10. Usage Instructions

### Prerequisites
1. Install all dependencies
2. Set up DuckDB with spatial extension
3. Ensure data directory structure exists
4. Configure logging if needed

### Running the Script
```bash
cd forest-stack/modules/rainfall/
python rainfall_avg.py
```

### Configuration
```python
# Main execution
start_date = datetime(2013, 1, 1)
end_date = datetime(2024, 11, 30)
db_path = 'dfhms.db'
```

---

## 11. Data Quality and Validation

### Quality Checks
- **Data Completeness**: Validates all dates are processed
- **Spatial Coverage**: Ensures proper region assignment
- **Value Validation**: Checks for reasonable rainfall values
- **Temporal Consistency**: Validates date sequences

### Validation Functions
```python
def validate_data_quality(conn):
    # Check for missing values
    missing_check = conn.execute("""
        SELECT range_code, COUNT(*) as total_records,
               SUM(CASE WHEN value IS NULL THEN 1 ELSE 0 END) as missing_rainfall
        FROM rainfall_data_agg
        GROUP BY range_code
        HAVING SUM(CASE WHEN value IS NULL THEN 1 ELSE 0 END) > 0
    """).fetchdf()
    
    # Check for anomalous values
    anomaly_check = conn.execute("""
        SELECT range_code, measurement_date, value as rainfall
        FROM rainfall_data_agg
        WHERE value < 0 OR value > 1000
    """).fetchdf()
```

---

## 12. Integration with Forest-Stack

### Forest Management Applications
- **Drought Monitoring**: Rainfall deficit analysis
- **Ecosystem Health**: Water availability assessment
- **Climate Analysis**: Precipitation trend analysis
- **Management Planning**: Seasonal water planning

### Data Products
- **Daily Averages**: Point-in-time rainfall data
- **Fortnightly Aggregations**: Regular monitoring intervals
- **Spatial Patterns**: Regional rainfall variations
- **Temporal Analysis**: Long-term precipitation trends

---

## 13. Troubleshooting

### Common Issues

1. **Download Failures**: Check internet connectivity and IMD server status
2. **Memory Issues**: Process smaller date ranges or increase system memory
3. **Database Errors**: Verify DuckDB installation and spatial extension
4. **Cache Problems**: Clear cache directory and restart processing

### Debug Tips
- Check log files for detailed error messages
- Monitor system resources during processing
- Validate input data formats and paths
- Test with small date ranges first

### Performance Issues
- **Slow Downloads**: Check network connectivity and server load
- **Memory Problems**: Reduce batch size or increase system memory
- **Processing Delays**: Verify database performance and indexes
- **Disk Space**: Monitor cache directory size

---

## 14. Future Enhancements

### Potential Improvements
- **Real-time Processing**: Automated daily updates
- **Additional Metrics**: Standard deviation, percentiles, extremes
- **Visualization**: Rainfall maps and trend charts
- **API Integration**: Direct database updates
- **Quality Metrics**: Data completeness and accuracy scores

### Integration Opportunities
- **Climate Data**: Correlation with temperature and humidity
- **Forest Health**: Integration with NDVI and canopy density
- **Alert Systems**: Automated drought/flood warnings
- **Reporting**: Automated report generation
- **Mobile Access**: Mobile-friendly data access
