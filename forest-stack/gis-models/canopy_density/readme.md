# Forest Canopy Density (FCD) Module

This module computes Forest Canopy Density (FCD) classes using multi-band Sentinel-2 data and a previously computed Forest Mask. It processes satellite imagery to generate density classifications across an Area of Interest (AOI) using advanced vegetation indices and parallel processing.

---

## Algorithm Overview

The Forest Canopy Density computation follows a systematic workflow that processes satellite imagery to generate classified canopy density maps:

### 1. **Area of Interest (AOI) Preparation**
The module begins by reading the input AOI geometry and dividing it into manageable processing chunks:

```python
# Read AOI geometry
df_rajasthan = gpd.read_file(aoi_file_path)
polygon = df_rajasthan["geometry"][0]

# Divide into processing grids for scalability
polygons = compute_grids(polygon, grid_size)
df_rajasthan_chunked = gpd.GeoDataFrame([{"geometry": polygon} for polygon in polygons])
```

**Purpose**: Large areas are divided into smaller grids to enable memory-efficient processing and parallel computation.

### 2. **Satellite Data Acquisition**
For each grid chunk, the module queries Sentinel-2 L2A imagery using STAC API:

```python
daterange = f"{year - 1}-10-01/{year - 1}-12-01"  # Post-monsoon season
query = catalog_element84.search(
    collections=["sentinel-2-l2a"], 
    datetime=daterange, 
    limit=100, 
    bbox=bbox
)
```

**Downloaded Bands**:
- **Red, Green, Blue**: Visible spectrum bands for vegetation analysis
- **NIR (Near-Infrared)**: Critical for vegetation health assessment
- **SWIR16**: Short-wave infrared for moisture and canopy structure
- **SCL (Scene Classification Layer)**: For cloud and quality masking

### 3. **Cloud Filtering and Scene Selection**
The module applies sophisticated cloud filtering using the SCL band:

```python
# Apply quality mask based on SCL classification
scl = ds.scl.astype("uint8").compute(scheduler="threads")
cloud_index = ~scl.isin([0, 3, 8, 9, 10, 0])  # Exclude clouds, shadows, water

# Find minimum set of scenes for maximum coverage
indices = find_minimum_mask_set(cloud_index.data)
```

**Quality Control**:
- **SCL Values Excluded**: 0 (No Data), 3 (Cloud Shadows), 8 (Cloud Medium Probability), 9 (Cloud High Probability), 10 (Thin Cirrus)
- **Optimal Scene Selection**: Algorithm selects minimum number of scenes to achieve maximum cloud-free pixel coverage

### 4. **Multi-temporal Composite Generation**
Clean pixels from selected scenes are merged into a composite:

```python
ds = fill_with_indices(ds, range(len(indices)), cloud_index.isel(time=indices))
```

**Approach**: Prioritizes cloud-free pixels from the best available scenes, creating a seamless composite image for each grid.

### 5. **Forest Canopy Density Calculation**
This is the core algorithm that computes FCD using advanced vegetation indices:

#### **Vegetation Index Computation**
```python
@njit(cache=True, parallel=True)
def compute_fcd_array(red, green, blue, nir, swir):
    # Normalized Difference Vegetation Index
    ndvi = (nir - red) / (nir + red + 1e-8)
    
    # Bare Soil Index
    bsi = ((swir + blue) - (nir + red)) / ((swir + blue) + (nir + red) + 1e-8)
    
    # Crown Shadow Index (brightness component)
    max_green = np.max(green)
    max_red = np.max(red)
    csi = np.sqrt((max_green - green) * (max_red - red))
    
    # Vegetation Density (normalized NDVI)
    max_ndvi = np.max(ndvi)
    min_ndvi = np.min(ndvi)
    vd = (ndvi - min_ndvi) / (max_ndvi - min_ndvi + 1e-8)
    
    # Shadow Soil Index
    ssi = np.sqrt(csi * np.abs(bsi))
    
    # Final Forest Canopy Density
    fcd = (100 * np.sqrt(vd * ssi)).astype("uint8")
    
    return fcd
```

#### **Index Explanations**:
- **NDVI**: Measures vegetation greenness and health
- **BSI**: Identifies bare soil areas to distinguish from vegetated areas  
- **CSI**: Captures shadow patterns beneath tree canopies
- **VD**: Normalizes NDVI to 0-1 range for consistent scaling
- **SSI**: Combines shadow and soil information for canopy structure assessment
- **FCD**: Final index (0-100) representing canopy density percentage

**Performance Optimization**: Uses Numba JIT compilation with parallel processing for fast computation across large raster arrays.

### 6. **FCD Classification with Configurable Thresholds**
The continuous FCD values are classified into discrete density classes:

```python
def classify_fcd(fcd: xr.DataArray, state_code: str = "default", custom_thresholds: dict = None):
    # Get appropriate thresholds (national default or state-specific)
    if custom_thresholds is not None:
        thresholds = custom_thresholds
    else:
        config = Config()
        thresholds = config.get_fcd_thresholds(state_code)
    
    # Apply classification based on thresholds
    classes = np.zeros_like(fcd, dtype=np.uint8)
    classes[(fcd > 0) & (fcd <= open_max)] = 1      # Open forest
    classes[(fcd > low_min) & (fcd <= low_max)] = 2  # Low density
    classes[(fcd > med_min) & (fcd <= med_max)] = 3  # Medium density
    classes[fcd > high_min] = 4                      # High density
```

#### **Classification Threshold System**:

**Rajasthan Thresholds**:
| Class | Range (%) | Description |
|-------|-----------|-------------|
| 1 | 0-25 | Open Forest - Sparse canopy cover |
| 2 | 25-50 | Low Density - Moderate canopy gaps |
| 3 | 50-75 | Medium Density - Good canopy coverage |
| 4 | 75-100 | High Density - Dense canopy cover |

**State-Specific Adaptations**:
States can define custom thresholds in `forest-stack/common/config/forest_config.py`:

```python
FCD_THRESHOLDS = {
    "RJ": {              # Rajasthan
        "open_forest": {"min": 0, "max": 25},
        "low_density": {"min": 25, "max": 50},
        "medium_density": {"min": 50, "max": 75},
        "high_density": {"min": 75, "max": 100}
    }
}
```

### 7. **Grid Mosaicking and Reprojection**
Individual grid results are combined into a seamless output:

```python
# Reproject all grid tiles to consistent CRS
grid_datasets = [
    rioxarray.open_rasterio(path, chunks={})
    .rio.reproject("epsg:4326")
    .astype("uint8")
    for path in paths
]

# Merge into single mosaic
ds_mosaiced_density = merge_arrays(grid_datasets, nodata=0)
```

### 8. **Forest Mask Integration**
The final step applies the pre-computed forest mask to retain classifications only within forest areas:

```python
# Load forest mask
ds_forest = rxr.open_rasterio(forrest_mask_file_path)

# Apply mask: retain FCD values only where forest exists
ds_mosaiced_density = xr.where(ds_forest == 1, ds_mosaiced_density, 0)
```

**Output Values**:
- **0**: Non-forest areas (masked out)
- **1**: Open forest
- **2**: Low density forest  
- **3**: Medium density forest
- **4**: High density forest

---

## Usage Examples

### Basic Usage (National Thresholds):
```python
compute_forrest_density(
    aoi_file_path="study_area.geojson",
    year=2023,
    grid_size=0.4,
    output_file_path="fcd_output.tif",
    forrest_mask_file_path="forest_mask.tif"
)
```

### State-Specific Thresholds:
```python
compute_forrest_density(
    aoi_file_path="rajasthan_boundary.geojson",
    year=2023,
    grid_size=0.4,
    output_file_path="rajasthan_fcd.tif",
    forrest_mask_file_path="rajasthan_forest_mask.tif",
    state_code="RJ"  # Uses Rajasthan-optimized thresholds
)
```

### Custom Research Thresholds:
```python
custom_thresholds = {
    "open_forest": {"min": 0, "max": 20},
    "low_density": {"min": 20, "max": 45},
    "medium_density": {"min": 45, "max": 70},
    "high_density": {"min": 70, "max": 100}
}

compute_forrest_density(
    ...,
    custom_thresholds=custom_thresholds
)
```

---

## Performance and Scalability

### Memory Management:
- **Tiled Processing**: Divides large areas into manageable chunks
- **Garbage Collection**: Explicit memory cleanup after each grid
- **Chunked Arrays**: Uses Dask/Xarray for memory-efficient operations

### Processing Optimization:
- **Parallel Computing**: Numba JIT compilation with parallel loops
- **Asynchronous I/O**: Efficient data downloading and processing
- **Optimal Scene Selection**: Minimizes data transfer while maximizing coverage
- **Quality Filtering**: Reduces processing overhead by excluding poor-quality pixels

### Scalability Features:
- **Grid-based Architecture**: Scales to any area size
- **Configurable Grid Size**: Balance between memory usage and processing efficiency  
- **Multi-threaded Computation**: Leverages multiple CPU cores
- **Cloud-optimized**: Designed for cloud computing environments

---

## Configuration Reference

**Add State-Specific Thresholds** in `forest-stack/common/config/forest_config.py`:

```python
# Common Indian State Codes:
# RJ-Rajasthan, KL-Kerala, MP-Madhya Pradesh, AS-Assam, 
# MH-Maharashtra, KA-Karnataka, TN-Tamil Nadu, etc.

FCD_THRESHOLDS = {
    "default": { ... },
    "<STATE_CODE>": {
        "open_forest": {"min": 0, "max": <value>},
        "low_density": {"min": <value>, "max": <value>},
        "medium_density": {"min": <value>, "max": <value>},
        "high_density": {"min": <value>, "max": 100}
    }
}
```
