# Forest-Stack Core Modules

Open-source analytics modules for forestry and climate workflows. A comprehensive Python library for processing satellite imagery, analyzing forest metrics, and generating climate data products using Sentinel-2 imagery and other geospatial datasets.

## Features

### Core Forest Models
- **NDVI Processing**: Automated NDVI mosaic generation using Sentinel-2 imagery with cloud masking and seasonal composites
- **Forest Mask Classification**: Binary forest/non-forest classification using NDVI time-series and LULC data
- **Canopy Density Analysis**: Forest Canopy Density (FCD) classification with multi-band Sentinel-2 analysis

### Climate & Environmental Utilities
- **Groundwater Analysis**: Groundwater depth and trend analysis tools
- **Rainfall Processing**: Average and mean rainfall calculation utilities
- **Soil Moisture Processing**: Area-weighted soil moisture statistics for rangeland regions

### Data Processing Tools
- **TIFF to PMTiles**: Conversion utilities for web-ready raster formats
- **Tile Management**: Automated tile processing and coverage optimization
- **Geospatial Utilities**: Common functions for geometry processing and data management

## Installation

```bash
# Clone the repository
git clone https://github.com/datakaveri/forest-stack.git
cd forest-stack

# Create virtual environment and install dependencies
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Project Structure

```
forest-stack/
├── forest-stack/                    # Main package
│   ├── models/                      # Core forest analysis models
│   │   ├── ndvi/                   # NDVI processing and mosaics
│   │   ├── forest_mask/            # Forest/non-forest classification
│   │   └── canopy_density/         # Forest canopy density analysis
│   ├── common/                     # Shared utilities and scripts
│   │   ├── config/                 # Configuration management
│   │   ├── data/                   # Sample data and boundaries
│   │   └── scripts/                # Climate and utility scripts
│   │       ├── groundwater_depth/  # Groundwater analysis
│   │       ├── groundwater_trend/  # Groundwater trend analysis
│   │       ├── rainfall/           # Rainfall processing
│   │       ├── soil_moisture/      # Soil moisture analysis
│   │       └── tif-to-pmtimles/    # Format conversion utilities
│   └── __init__.py
├── requirements.txt                # Python dependencies
└── README.md
```

## Core Modules

### NDVI Processing (`forest-stack.models.ndvi`)
Automated NDVI mosaic generation using Sentinel-2 imagery:
- Cloud masking and quality filtering
- Seasonal composite generation
- Tile-based processing for large areas
- Web-ready output formats (GeoTIFF, PMTiles)

### Forest Mask Classification (`forest-stack.models.forest_mask`)
Binary forest/non-forest classification:
- NDVI time-series analysis
- LULC data integration
- Vegetation cyclicity evaluation
- Optimized for Rajasthan region

### Canopy Density Analysis (`forest-stack.models.canopy_density`)
Forest Canopy Density (FCD) classification:
- Multi-band Sentinel-2 analysis
- Brightness, greenness, and shadow indices
- Density class categorization (Open, Low, Medium, High)
- Parallelized processing with Dask/Xarray

## Climate & Environmental Utilities

### Groundwater Analysis
- **Depth Analysis**: Groundwater depth processing and statistics
- **Trend Analysis**: Groundwater trend calculation and monitoring

### Rainfall Processing
- **Average Rainfall**: Area-weighted rainfall averaging
- **Mean Rainfall**: Statistical rainfall analysis

### Soil Moisture Processing
- Area-weighted soil moisture statistics
- Rangeland region analysis
- PostgreSQL/PostGIS integration

## Data Processing Tools

### Format Conversion
- **TIFF to PMTiles**: Convert raster data to web-optimized PMTiles format
- **Tile Management**: Automated tile processing and coverage optimization

### Geospatial Utilities
- Common geometry processing functions
- Spatial data management tools
- Configuration management for API keys and settings

## Dependencies

The library requires Python 3.8+ and includes these key dependencies:
- `rasterio` - Geospatial raster I/O
- `pystac-client` - STAC API client for satellite imagery
- `xarray` - Multi-dimensional data arrays
- `numpy` - Numerical computing
- `shapely` - Geometric operations
- `aiohttp` - Asynchronous HTTP client

## Quick Start Example

```python
from forest_stack.models.ndvi import NDVIProcessor
from forest_stack.models.forest_mask import ForestMaskProcessor
from forest_stack.models.canopy_density import CanopyDensityProcessor

# Initialize processors
ndvi_processor = NDVIProcessor()
forest_mask_processor = ForestMaskProcessor()
canopy_processor = CanopyDensityProcessor()

# Process NDVI mosaic
ndvi_processor.process_ndvi_mosaic(
    aoi_geometry="path/to/aoi.geojson",
    year=2023,
    season="spring"
)

# Generate forest mask
forest_mask_processor.generate_forest_mask(
    aoi_geometry="path/to/aoi.geojson",
    daterange="2023-01-01/2023-12-31"
)

# Calculate canopy density
canopy_processor.process_canopy_density(
    aoi_geometry="path/to/aoi.geojson",
    forest_mask_path="path/to/forest_mask.tif"
)
```

## Contributing

This library is part of the DataKaveri Forest-Stack project. Contributions are welcome! Please refer to individual module documentation for detailed usage instructions and examples.

## License

See [LICENSE](LICENSE) file for details.
