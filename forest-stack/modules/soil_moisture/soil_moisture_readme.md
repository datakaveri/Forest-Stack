# Soil Moisture Processing – Detailed Documentation

This module processes soil moisture data from annual TIFF stacks and generates area-weighted statistics for forest management regions. It handles multi-band raster data, implements intelligent caching, and provides parallel processing for optimal performance.

---

## 1. What the script does

### Data Processing Pipeline

1. **Data Preparation**:
   - Exports region polygons to GeoPackage for exactextract
   - Implements intelligent caching system with hash validation
   - Sets up parallel processing configuration

2. **Raster Processing**:
   - Processes annual TIFF stacks with multiple bands (fortnights)
   - Creates temporary VRT files for individual band processing
   - Performs area-weighted extraction using exactextract

3. **Caching System**:
   - Implements file hash-based caching
   - Stores results in Parquet format for efficiency
   - Validates cache integrity before reuse

4. **Data Aggregation**:
   - Combines results from all bands and years
   - Generates fortnightly labels and timestamps
   - Creates SQL insert statements for database integration

---

## 2. Data Sources

### Input Data
- **Soil Moisture TIFFs**: Annual stacked TIFF files
  - Format: `sm_{year}_annual_stack.tif`
  - Bands: Each band represents a fortnight (24 bands per year)
  - Coverage: Forest management regions

### Spatial Data
- **Region Boundaries**: `regions2` table in DuckDB
  - Range-type regions for forest management
  - Administrative codes and geometries
  - Exported to GeoPackage for exactextract

---

## 3. Core Functions

### `get_regions_gpkg()`
Exports region polygons to GeoPackage with caching.

### `process_raster_band(args)`
Processes individual bands from TIFF files with caching.

### `process_annual_tiff(tiff_path, poly_path, year)`
Processes all bands in an annual TIFF file.

---

## 4. Configuration

### Directory Structure
```
data/
├── soilmoisture_tiffs/     # Input TIFF files
├── soilmoisture_output/    # Output files
└── soilmoisture_cache/     # Cache directory
    ├── parquet/           # Cached results
    └── range_regions.gpkg # Region boundaries
```

### Performance Settings
```python
N_CORES = multiprocessing.cpu_count()
MEMORY_LIMIT = int(psutil.virtual_memory().total * 0.75)
```

---

## 5. Dependencies

### Core Libraries
- `duckdb` - Database operations and spatial queries
- `pandas` - Data manipulation and analysis
- `geopandas` - Geospatial data handling
- `rasterio` - Raster data processing
- `exactextract` - Area-weighted raster extraction

### Additional Packages
- `hashlib` - File hash calculation
- `multiprocessing` - Parallel processing
- `concurrent.futures` - Advanced parallel processing
- `psutil` - System resource monitoring
- `tqdm` - Progress tracking

---

## 6. Usage Instructions

### Prerequisites
1. Install all dependencies
2. Set up DuckDB with spatial extension
3. Install exactextract tool
4. Ensure soil moisture TIFF files are available

### Running the Script
```bash
cd forest-stack/modules/soil_moisture/
python soil_moisture.py
```

---

## 7. Output Data Structure

### Final Aggregated Data
| code | snapshot_label | value |
|------|----------------|-------|
| RJ001 | soil_moisture_2020_january_fn1 | 15.5 |
| RJ001 | soil_moisture_2020_january_fn2 | 18.2 |

### SQL Insert Generation
Generates SQL insert statements for PostgreSQL/PostGIS integration.

---

## 8. Performance Optimization

### Parallel Processing
- Processes bands in parallel using ProcessPoolExecutor
- Configurable number of cores
- Memory-efficient processing

### Caching System
- File hash-based validation
- Parquet format for efficient storage
- Incremental processing for large datasets

---

## 9. Integration with Forest-Stack

### Forest Management Applications
- **Drought Monitoring**: Soil moisture stress indicators
- **Ecosystem Health**: Water availability for forest ecosystems
- **Climate Analysis**: Soil moisture trend analysis
- **Management Planning**: Water-dependent forest areas

---

## 10. Troubleshooting

### Common Issues
1. **File Not Found**: Check data directory and file naming
2. **ExactExtract Errors**: Verify tool installation and permissions
3. **Memory Issues**: Reduce parallel processing or increase system memory
4. **Cache Problems**: Clear cache directory and restart processing

### Debug Tips
- Check file paths and naming conventions
- Validate input data formats
- Monitor system resources during processing
- Test with small datasets first