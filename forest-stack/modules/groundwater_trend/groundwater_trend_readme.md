# Groundwater Trend Analysis – Detailed Documentation

This module processes groundwater trend data from NASA's GLDAS (Global Land Data Assimilation System) dataset and generates fortnightly aggregations for forest management applications. It downloads, processes, and analyzes groundwater storage trends using parallel processing for optimal performance.

---

## 1. What the script does

### Data Processing Pipeline

1. **Data Download**:
   - Downloads GLDAS CLSM025 daily groundwater data from NASA servers
   - Handles authentication and parallel downloads
   - Converts NetCDF files to GeoTIFF format

2. **Spatial Processing**:
   - Exports region polygons to GeoPackage for exactextract
   - Performs area-weighted extraction using exactextract
   - Processes data in parallel across multiple CPU cores

3. **Temporal Aggregation**:
   - Calculates daily means for each region
   - Aggregates data into fortnightly periods (fn1: 1-15, fn2: 16-end)
   - Generates temporal labels for database integration

4. **Database Integration**:
   - Creates SQL insert statements for PostgreSQL/PostGIS
   - Updates local DuckDB with processed data
   - Generates comprehensive output files

---

## 2. Data Sources

### Input Data
- **GLDAS Dataset**: NASA's Global Land Data Assimilation System
  - Daily groundwater storage data (GWS_tavg)
  - 0.25° spatial resolution
  - Available from 2000-present
  - NetCDF format with daily temporal resolution

### Authentication
- Requires NASA Earthdata credentials
- Set via environment variables: `USERNAME` and `PASSWORD`
- Uses `.env` file for secure credential management

---

## 3. Configuration

### Environment Variables
```bash
# .env file
USERNAME=your_nasa_username
PASSWORD=your_nasa_password
```

### Directory Structure
```
data/
├── groundwater_trends_nc/     # NetCDF files
├── groundwater_trends_tiff/   # Converted TIFF files
├── groundwater_trends_csv/    # Extracted CSV data
└── cache/groundwater_trends/  # Cached results
```

### Performance Settings
```python
N_CORES = multiprocessing.cpu_count()
MEMORY_LIMIT = int(psutil.virtual_memory().total * 0.75)
CHUNK_SIZE = max(1, MEMORY_LIMIT // (1024**3))
```

---

## 4. Core Functions

### `download_data(start_date: str, end_date: str)`
Downloads GLDAS data for specified date range.

**Parameters**:
- `start_date`: Start date in 'YYYYMMDD' format
- `end_date`: End date in 'YYYYMMDD' format

**Process**:
1. Generates date list for the period
2. Downloads NetCDF files in parallel
3. Converts to GeoTIFF format using GDAL

### `process_tiff_file(tiff_path, poly_path)`
Processes individual TIFF files using exactextract.

**Parameters**:
- `tiff_path`: Path to input TIFF file
- `poly_path`: Path to polygon GeoPackage

**Returns**:
- DataFrame with region codes and mean values

### `aggregate_fortnightly(start_date: str, end_date: str)`
Main aggregation function for fortnightly analysis.

**Process**:
1. Loads cached data or processes new files
2. Calculates daily means per region
3. Aggregates into fortnightly periods
4. Generates SQL insert statements

---

## 5. Data Processing Details

### Download Process
```python
url = (f"https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/"
       f"GLDAS_CLSM025_DA1_D_EP.2.2/{year_}/{month_}/"
       f"GLDAS_CLSM025_DA1_D_EP.A{day_str}.022.nc4")
```

### Conversion Process
```bash
gdal_translate -of GTiff NETCDF:"{nc_file}":GWS_tavg "{tiff_file}"
```

### Extraction Process
```bash
./exactextract -r INDEX:{tiff_path} -p {poly_path} -f "code" -s "mean(INDEX)" -o {out_csv}
```

### Fortnightly Aggregation
- **fn1**: Days 1-15 of each month
- **fn2**: Days 16-end of each month
- **Label Format**: `groundwater_trend_{year}_{month}_{fn}`

---

## 6. Caching System

### Cache Strategy
- **Parquet Files**: Cached results for each year
- **Hash Validation**: File integrity checking
- **Incremental Processing**: Only processes new/changed data

### Cache Files
```
cache/groundwater_trends/
├── measurements_2013.parquet
├── measurements_2014.parquet
└── temp_full.parquet
```

---

## 7. Output Data Structure

### CSV Output
| range_code | value | start_date | end_date | label |
|------------|-------|------------|----------|-------|
| RJ001 | 0.15 | 2023-01-01 | 2023-01-15 | groundwater_trend_2023_january_fn1 |
| RJ001 | 0.18 | 2023-01-16 | 2023-01-31 | groundwater_trend_2023_january_fn2 |

### SQL Insert Template
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
    WHERE dp.name = 'Groundwater trend'
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
- **Processing**: ProcessPoolExecutor for CPU-intensive tasks
- **Memory Management**: Configurable memory limits and chunk sizes

### Memory Management
```python
# Use 75% of available RAM
MEMORY_LIMIT = int(psutil.virtual_memory().total * 0.75)
con.execute(f"SET memory_limit='{MEMORY_LIMIT}B'")
```

### Caching Strategy
- **Year-based Caching**: Separate cache files per year
- **Hash Validation**: Prevents reprocessing unchanged data
- **Incremental Updates**: Only processes new data

---

## 9. Dependencies

### Core Libraries
- `duckdb` - Database operations and spatial queries
- `pandas` - Data manipulation and analysis
- `geopandas` - Geospatial data handling
- `rasterio` - Raster data processing
- `exactextract` - Area-weighted raster extraction

### System Tools
- `gdal` - Geospatial data conversion
- `wget` - Data downloading
- `exactextract` - Raster extraction tool

### Python Packages
- `concurrent.futures` - Parallel processing
- `multiprocessing` - CPU core management
- `psutil` - System resource monitoring
- `tqdm` - Progress tracking
- `python-dotenv` - Environment variable management

---

## 10. Usage Instructions

### Prerequisites
1. Install all dependencies
2. Set up NASA Earthdata account
3. Configure environment variables
4. Install exactextract tool

### Running the Script
```bash
cd forest-stack/modules/groundwater_trend/
python groundwater_trend.py
```

### Configuration
```python
# Example execution
start_date = "20130101"
end_date = "20241130"
```

---

## 11. Data Quality and Validation

### Quality Checks
- **Data Completeness**: Validates all dates are processed
- **Spatial Accuracy**: Ensures proper region assignment
- **Temporal Consistency**: Checks for data gaps
- **Value Ranges**: Validates groundwater storage values

### Error Handling
- **Download Failures**: Retries and logs failed downloads
- **Processing Errors**: Continues with available data
- **Memory Issues**: Implements chunked processing
- **File Corruption**: Hash-based validation

---

## 12. Integration with Forest-Stack

### Forest Management Applications
- **Drought Monitoring**: Groundwater stress indicators
- **Ecosystem Health**: Water availability for forest ecosystems
- **Climate Adaptation**: Long-term trend analysis
- **Management Planning**: Water-dependent forest areas

### Data Products
- **Fortnightly Trends**: Regular monitoring intervals
- **Annual Summaries**: Year-over-year comparisons
- **Spatial Patterns**: Regional groundwater variations
- **Temporal Analysis**: Long-term trend identification

---

## 13. Troubleshooting

### Common Issues

1. **Authentication Errors**: Check NASA Earthdata credentials
2. **Download Failures**: Verify internet connectivity and server status
3. **Memory Issues**: Reduce chunk size or process smaller date ranges
4. **Processing Errors**: Check exactextract installation and permissions

### Debug Tips
- Monitor system resources during processing
- Check log files for specific error messages
- Validate input data formats and paths
- Test with small date ranges first

### Performance Issues
- **Slow Downloads**: Check network connectivity and server load
- **Memory Problems**: Reduce memory limit or chunk size
- **Processing Delays**: Verify exactextract tool performance
- **Disk Space**: Monitor cache directory size

---

## 14. Future Enhancements

### Potential Improvements
- **Real-time Processing**: Automated daily updates
- **Additional Metrics**: Standard deviation, percentiles
- **Visualization**: Trend charts and maps
- **API Integration**: Direct database updates
- **Quality Metrics**: Data completeness and accuracy scores

### Integration Opportunities
- **Climate Data**: Correlation with precipitation and temperature
- **Forest Health**: Integration with NDVI and canopy density
- **Alert Systems**: Automated drought warnings
- **Reporting**: Automated report generation
