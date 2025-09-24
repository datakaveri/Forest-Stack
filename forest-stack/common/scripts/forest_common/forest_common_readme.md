# Forest Common Utilities – Detailed Documentation

This module provides shared utility functions for geospatial processing and STAC API interactions used across the Forest-Stack library. It includes polygon gridding utilities and STAC client configuration for satellite imagery access.

---

## 1. What the module provides

### Core Functions

1. **Polygon Gridding (`compute_grids`)**:
   - Divides large polygons into smaller grid-based sub-polygons
   - Enables scalable processing of large areas of interest
   - Configurable grid size for optimal performance vs. granularity

2. **STAC Client Configuration**:
   - Pre-configured STAC client for Element84's Earth Search API
   - Centralized access point for satellite imagery queries
   - Optimized for Sentinel-2 and other Earth observation datasets

---

## 2. Function Details

### `compute_grids(polygon: Polygon, grid_size: float) -> List[Polygon]`

**Purpose**: Divides a polygon into smaller grid-based polygons for scalable processing.

**Parameters**:
- `polygon` (Polygon): The input polygon to be divided
- `grid_size` (float): Grid cell size in degrees. Higher values create larger sub-polygons

**Returns**:
- `List[Polygon]`: List of intersecting grid polygons

**Algorithm**:
1. Calculates bounding box of input polygon
2. Creates regular grid cells across the bounding box
3. Intersects each grid cell with the original polygon
4. Filters out empty intersections
5. Applies small buffer (1e-4) to handle precision issues

**Example Usage**:
```python
from forest_stack.common.scripts.forest_common import compute_grids
from shapely.geometry import Polygon

# Create a large polygon
large_polygon = Polygon([(70, 25), (80, 25), (80, 35), (70, 35)])

# Divide into 0.1 degree grid cells
grid_polygons = compute_grids(large_polygon, 0.1)
print(f"Created {len(grid_polygons)} grid cells")
```

### STAC Client Access

**Purpose**: Provides pre-configured access to Element84's STAC API.

**Configuration**:
- Uses `Config.ELEMENT84_STAC_URL` from forest configuration
- Optimized for Sentinel-2 L2A data access
- Supports various Earth observation datasets

**Example Usage**:
```python
from forest_stack.common.scripts.forest_common import catalog_element84

# Search for Sentinel-2 data
search = catalog_element84.search(
    collections=["sentinel-s2-l2a"],
    bbox=[70, 25, 80, 35],
    datetime="2023-01-01/2023-12-31"
)
```

---

## 3. Dependencies

- `shapely` - Geometric operations and polygon processing
- `pystac-client` - STAC API client for satellite imagery
- `numpy` - Numerical operations for grid generation
- `forest_stack.common.config.forest_config` - Configuration management

---

## 4. Performance Considerations

### Grid Size Selection
- **Small grid_size (0.01-0.05)**: Higher granularity, more processing overhead
- **Medium grid_size (0.1-0.5)**: Balanced performance and detail
- **Large grid_size (1.0+)**: Faster processing, less spatial detail

### Memory Usage
- Grid generation is memory-efficient for most polygon sizes
- Consider polygon complexity when choosing grid size
- Large polygons with small grid sizes may generate many sub-polygons

---

## 5. Integration with Forest-Stack

This module is used by:
- **NDVI Processing**: For tiling large areas of interest
- **Forest Mask**: For processing large state boundaries
- **Canopy Density**: For scalable analysis of forest areas
- **Climate Scripts**: For regional data processing

---

## 6. Error Handling

- **Empty intersections**: Automatically filtered out
- **Precision issues**: Handled with small buffer application
- **Invalid polygons**: Input validation should be performed before calling

---

## 7. Configuration Requirements

Ensure the following configuration is available:
```python
# In forest_config.py
ELEMENT84_STAC_URL = "https://earth-search.aws.element84.com/v1"
```

---

## 8. Example Integration

```python
from forest_stack.common.scripts.forest_common import compute_grids, catalog_element84
from shapely.geometry import Polygon
import geopandas as gpd

# Load area of interest
aoi = gpd.read_file("rajasthan_boundary.geojson")
polygon = aoi.geometry.iloc[0]

# Create processing grid
grid_polygons = compute_grids(polygon, 0.1)

# Process each grid cell
for i, grid_cell in enumerate(grid_polygons):
    # Search for data in this grid cell
    bbox = grid_cell.bounds
    search = catalog_element84.search(
        collections=["sentinel-s2-l2a"],
        bbox=bbox,
        datetime="2023-01-01/2023-12-31"
    )
    
    # Process results...
    print(f"Processing grid cell {i+1}/{len(grid_polygons)}")
```

---

## 9. Troubleshooting

### Common Issues

1. **Import Error**: Ensure forest_config.py is properly configured
2. **STAC Connection**: Check internet connectivity and API availability
3. **Memory Issues**: Reduce grid_size for very large polygons
4. **Empty Results**: Verify polygon geometry validity

### Debug Tips

- Test with small polygons first
- Monitor memory usage with large grid sizes
- Check STAC API status if queries fail
- Validate input polygon geometry before processing
