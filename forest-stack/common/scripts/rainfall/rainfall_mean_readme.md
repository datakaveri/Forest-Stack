# Normal Rainfall Calculation – Detailed Documentation

This module calculates normal rainfall values for forest management regions using historical rainfall data. It processes daily rainfall TIFF files, computes long-term averages, and generates normal rainfall values for fortnightly periods to support forest management and climate analysis.

---

## 1. What the script does

### Data Processing Pipeline

1. **Data Loading**:
   - Loads subdistrict/range boundaries from DuckDB
   - Processes historical rainfall TIFF files
   - Filters data to last 30 years for normal calculation

2. **Spatial Processing**:
   - Performs area-weighted extraction using exactextract
   - Processes each TIFF file for all regions
   - Handles multiprocessing for efficient computation

3. **Normal Calculation**:
   - Calculates long-term averages for each day of year
   - Groups data by region, month, and day
   - Computes normal rainfall for fortnightly periods

4. **Output Generation**:
   - Creates SQL insert statements for database integration
   - Generates normal rainfall values for each region
   - Provides comprehensive data for forest management

---

## 2. Data Sources

### Input Data
- **Rainfall TIFF Files**: Daily rainfall raster data
  - Format: GeoTIFF files with daily temporal resolution
  - Naming convention: `YYYYMMDD_rainfall.tif`
  - Coverage: Forest management regions

- **Spatial Boundaries**: `regions2` table in DuckDB
  - Subdistrict/range boundaries
  - Administrative codes and names
  - Used for spatial aggregation

### Data Filtering
- **Temporal Filter**: Last 30 years of data
- **Spatial Filter**: Range-type regions only
- **Quality Filter**: Non-null rainfall values

---

## 3. Core Functions

### `process_tif_file(tif_file, subdistricts)`
Processes individual TIFF files and extracts rainfall data.

**Parameters**:
- `tif_file`: Path to input TIFF file
- `subdistricts`: GeoDataFrame with region boundaries

**Process**:
1. Extracts date from filename
2. Performs area-weighted extraction using exactextract
3. Filters out null values
4. Returns rainfall data with metadata

### `main()`
Main processing function that orchestrates the entire pipeline.

**Process**:
1. Connects to DuckDB and loads spatial data
2. Filters TIFF files to last 30 years
3. Processes files in parallel
4. Calculates normal rainfall values
5. Generates SQL output

---

## 4. Data Processing Details

### File Processing
```python
def process_tif_file(tif_file, subdistricts):
    date = datetime.strptime(tif_file.name.split('_')[0], '%Y%m%d')
    rainfall_data = []
    
    with rasterio.open(tif_file) as src:
        means = exactextract.exact_extract(src, subdistricts, ['mean'])
        
        for j, feature in enumerate(means):
            mean_value = feature['properties']['mean']
            if not pd.isna(mean_value):
                rainfall_data.append({
                    'code': subdistricts.iloc[j]['code'],
                    'date': date,
                    'value': mean_value,
                    'month': date.month,
                    'day': date.day
                })
```

### Normal Calculation
```python
# Calculate normal rainfall values
normal_rainfall = df.groupby(['code', 'month', 'day'])['value'].mean().reset_index()

# Process each region
for code in normal_rainfall['code'].unique():
    code_data = normal_rainfall[normal_rainfall['code'] == code]
    for month in range(1, 13):
        month_data = code_data[code_data['month'] == month]
        
        # First fortnight (days 1-15)
        fn1_mean = month_data[month_data['day'] <= 15]['value'].sum()
        
        # Second fortnight (days 16-end)
        fn2_mean = month_data[month_data['day'] > 15]['value'].sum()
```

---

## 5. Configuration

### Database Configuration
```python
conn = duckdb.connect('dfhms.db')
conn.execute("INSTALL spatial;")
conn.execute("LOAD spatial;")
```

### Data Paths
```python
DATA_DIR = Path("data/rainfall_data")
```

### Processing Parameters
```python
# Calculate cutoff date (30 years ago from today)
today = datetime.now()
cutoff_date = today - timedelta(days=30*365)

# Multiprocessing configuration
num_cores = 1  # Can be increased for parallel processing
```

---

## 6. Spatial Processing

### Region Loading
```sql
SELECT code, name, ST_AsWKB(geom) as geom_wkb 
FROM regions2 
WHERE geom IS NOT NULL AND type = 'range'
```

### ExactExtract Command
```bash
./exactextract -r INDEX:{tiff_path} -p {poly_path} -f "code" -s "mean(INDEX)" -o {out_csv}
```

### Data Structure
- **Input**: Daily rainfall TIFF files
- **Processing**: Area-weighted extraction per region
- **Output**: Normal rainfall values by region and fortnight

---

## 7. Output Data Structure

### Normal Rainfall Calculation
| code | month | day | value |
|------|-------|-----|-------|
| RJ001 | 1 | 1 | 0.5 |
| RJ001 | 1 | 2 | 0.8 |
| RJ001 | 1 | 15 | 2.1 |

### Fortnightly Aggregation
- **fn1**: Days 1-15 of each month
- **fn2**: Days 16-end of each month
- **Label Format**: `normal_rainfall_{month}_{fn}`

### SQL Output Structure
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
    WHERE dp.name = 'Normal rainfall'
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

### Multiprocessing
```python
# Set up multiprocessing
num_cores = 1  # Adjust based on system capabilities
pool = mp.Pool(num_cores)
process_func = partial(process_tif_file, subdistricts=subdistricts)

# Process files with progress bar
with tqdm(total=total_files, desc="Processing rainfall files") as pbar:
    for result in pool.imap_unordered(process_func, tif_files):
        results.extend(result)
        pbar.update()
```

### Memory Management
- **Chunked Processing**: Processes files in manageable chunks
- **Memory Cleanup**: Explicitly deletes large objects
- **Efficient Data Structures**: Uses pandas for data manipulation

---

## 9. Dependencies

### Core Libraries
- `duckdb` - Database operations and spatial queries
- `rasterio` - Raster data processing
- `exactextract` - Area-weighted raster extraction
- `pandas` - Data manipulation
- `geopandas` - Geospatial data handling

### Additional Packages
- `numpy` - Numerical operations
- `shapely` - Geometric operations
- `tqdm` - Progress tracking
- `multiprocessing` - Parallel processing
- `pathlib` - File path handling

### System Tools
- `exactextract` - Raster extraction tool (external binary)

---

## 10. Usage Instructions

### Prerequisites
1. Install all dependencies
2. Set up DuckDB with spatial extension
3. Install exactextract tool
4. Ensure rainfall TIFF files are available

### Running the Script
```bash
cd forest-stack/common/scripts/rainfall/
python rainfall_mean.py
```

### Data Requirements
- Rainfall TIFF files in `data/rainfall_data/` directory
- Files named in format: `YYYYMMDD_rainfall.tif`
- DuckDB database with `regions2` table

---

## 11. Data Quality and Validation

### Quality Checks
- **Data Completeness**: Validates all required files are processed
- **Spatial Accuracy**: Ensures proper region assignment
- **Temporal Coverage**: Checks for sufficient historical data
- **Value Validation**: Validates rainfall value ranges

### Normal Calculation Validation
- **Sufficient Data**: Ensures minimum number of observations per day
- **Temporal Consistency**: Validates date sequences
- **Spatial Coverage**: Checks all regions have data
- **Statistical Validity**: Validates normal calculation methodology

---

## 12. Integration with Forest-Stack

### Forest Management Applications
- **Climate Baseline**: Normal rainfall for comparison
- **Drought Assessment**: Deviation from normal conditions
- **Seasonal Planning**: Fortnightly normal values
- **Risk Assessment**: Historical rainfall patterns

### Data Products
- **Normal Rainfall**: Long-term averages by region
- **Fortnightly Normals**: Regular monitoring intervals
- **Spatial Patterns**: Regional rainfall variations
- **Temporal Analysis**: Seasonal rainfall patterns

---

## 13. Troubleshooting

### Common Issues

1. **File Not Found**: Check data directory and file naming
2. **ExactExtract Errors**: Verify tool installation and permissions
3. **Memory Issues**: Reduce parallel processing or increase system memory
4. **Database Errors**: Check DuckDB installation and spatial extension

### Debug Tips
- Check file paths and naming conventions
- Validate input data formats
- Monitor system resources during processing
- Test with small datasets first

### Performance Issues
- **Slow Processing**: Increase parallel processing cores
- **Memory Problems**: Reduce batch size or increase system memory
- **Disk Space**: Monitor temporary file creation
- **I/O Bottlenecks**: Use SSD storage for better performance

---

## 14. Future Enhancements

### Potential Improvements
- **Dynamic Normal Calculation**: Rolling window normal calculation
- **Additional Statistics**: Standard deviation, percentiles, extremes
- **Spatial Interpolation**: Fill gaps in sparse data
- **Quality Metrics**: Data completeness and accuracy scores

### Integration Opportunities
- **Climate Analysis**: Integration with temperature and humidity data
- **Forest Health**: Correlation with NDVI and canopy density
- **Risk Assessment**: Automated drought/flood risk analysis
- **Reporting**: Automated normal rainfall reports
- **Visualization**: Normal rainfall maps and charts

---

## 15. Statistical Methodology

### Normal Calculation
- **Method**: Long-term average for each day of year
- **Period**: Last 30 years of data
- **Aggregation**: Sum of daily values for fortnightly periods
- **Quality Control**: Filters null and invalid values

### Fortnightly Aggregation
- **fn1**: Sum of days 1-15 for each month
- **fn2**: Sum of days 16-end for each month
- **Labeling**: Month name and fortnight identifier

### Data Validation
- **Minimum Observations**: Ensures sufficient data for reliable normals
- **Temporal Consistency**: Validates date sequences
- **Spatial Coverage**: Checks all regions have data
- **Value Ranges**: Validates reasonable rainfall values
