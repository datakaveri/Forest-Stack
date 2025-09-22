# Forest-Stack

Open-source geospatial analytics modules for forestry, climate, and environmental monitoring workflows. A comprehensive Python library for processing satellite imagery, analyzing forest metrics, and generating climate data products using Sentinel-2 imagery and other geospatial datasets.

## Features

### Core Forest Analysis Models
- **NDVI Processing**: Automated NDVI mosaic generation using Sentinel-2 imagery with cloud masking, seasonal composites, and parallel processing
- **Forest Mask Classification**: Binary forest/non-forest classification using NDVI time-series, LULC data integration, and vegetation cyclicity evaluation
- **Canopy Density Analysis**: Forest Canopy Density (FCD) classification with multi-band Sentinel-2 analysis using brightness, greenness, and shadow indices

### Climate & Environmental Utilities
- **Groundwater Analysis**: Comprehensive groundwater depth processing and trend analysis tools
- **Rainfall Processing**: Statistical rainfall analysis with average and mean calculations
- **Soil Moisture Processing**: Area-weighted soil moisture statistics for rangeland regions with PostgreSQL/PostGIS integration

### Carbon & Wildlife Management
- **Carbon Calculator**: TypeScript-based carbon sequestration calculations with species-specific biomass models
- **Wildlife Management**: Digital forms and tools for animal rescue operations and wildlife census management

### Data Processing & Web Tools
- **TIFF to PMTiles**: Automated conversion utilities for web-ready raster formats
- **Tile Management**: Tree cover and canopy tile processing with coverage optimization
- **Geospatial Utilities**: Common functions for geometry processing, spatial data management, and configuration handling

## Installation

```bash
# Clone the repository
git clone https://github.com/datakaveri/forest-stack.git
cd Forest-Stack

# Create virtual environment and install dependencies
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Project Structure

```
Forest-Stack/
├── forest-stack/                    # Main package
│   ├── models/                      # Core forest analysis models
│   │   ├── ndvi/                   # NDVI processing and mosaics
│   │   ├── forest_mask/            # Forest/non-forest classification
│   │   └── canopy_density/         # Forest canopy density analysis
│   ├── common/                     # Shared utilities and configuration
│   │   ├── config/                 # Configuration management
│   │   ├── data/                   # Sample data and boundary files
│   │   │   ├── rajasthan_state_simp.geojson
│   │   │   └── rajasthan-convex-hull.geojson
│   │   └── scripts/                # Climate, utility, and management scripts
│   │       ├── carbon_calculator/  # Carbon sequestration calculations (TypeScript)
│   │       ├── forest_common/      # Common forestry utilities
│   │       ├── forms/              # Digital forms and management tools
│   │       │   └── wildlife/       # Wildlife management forms
│   │       ├── groundwater_depth/  # Groundwater depth analysis
│   │       ├── groundwater_trend/  # Groundwater trend monitoring
│   │       ├── rainfall/           # Rainfall processing utilities
│   │       ├── soil_moisture/      # Soil moisture analysis
│   │       ├── tif-to-pmtimles/    # Format conversion utilities
│   │       └── treecover_and_canopy_tile_commands/ # Tile processing
│   └── __init__.py
├── requirements.txt                # Python dependencies
├── LICENSE                        # License information
└── README.md
```

## Core Modules

### NDVI Processing (`forest_stack.models.ndvi`)
Advanced NDVI mosaic generation using Sentinel-2 imagery with:
- Automated cloud masking and quality filtering using SCL (Scene Classification Layer)
- Seasonal and temporal composite generation
- Parallel processing with concurrent futures and async I/O
- Tile-based processing for memory-efficient handling of large areas
- Web-ready output formats (GeoTIFF, PMTiles)
- Configurable STAC catalog integration (Planetary Computer, Element84)

### Forest Mask Classification (`forest_stack.models.forest_mask`)
Sophisticated binary forest/non-forest classification featuring:
- Multi-temporal NDVI time-series analysis with seasonal patterns
- LULC (Land Use Land Cover) data integration for enhanced accuracy
- Vegetation cyclicity evaluation using statistical thresholds
- Dask-powered distributed computing for large-scale processing
- Optimized for Rajasthan region with customizable parameters
- Integration with ODC (Open Data Cube) STAC workflows

### Canopy Density Analysis (`forest_stack.models.canopy_density`)
Comprehensive Forest Canopy Density (FCD) classification with:
- Multi-spectral Sentinel-2 band analysis (Red, NIR, SWIR)
- Advanced brightness, greenness, and shadow index calculations
- Four-tier density classification (Open, Low, Medium, High canopy)
- Parallelized processing using Dask and Xarray for efficient computation
- Integration with existing forest mask outputs for targeted analysis

## Climate & Environmental Utilities

### Groundwater Analysis
- **Depth Analysis**: Comprehensive groundwater depth processing with statistical analysis and spatial interpolation
- **Trend Analysis**: Long-term groundwater trend calculation, monitoring, and visualization tools
- Integration with hydrological datasets and time-series analysis

### Rainfall Processing
- **Average Rainfall**: Area-weighted rainfall averaging with spatial aggregation methods
- **Mean Rainfall**: Statistical rainfall analysis with temporal pattern recognition
- Support for multiple precipitation datasets and temporal resolutions

### Soil Moisture Processing
- Area-weighted soil moisture statistics optimized for rangeland regions
- Advanced spatial analysis with geometric processing
- PostgreSQL/PostGIS integration for efficient spatial queries
- Multi-temporal soil moisture trend analysis

## Carbon & Wildlife Management Tools

### Carbon Calculator (`forest_stack.common.scripts.carbon_calculator`)
TypeScript-based carbon sequestration calculations featuring:
- Species-specific biomass models and growth parameters
- Wood density and biomass expansion factor integration  
- Diameter-based carbon storage calculations
- Temporal carbon accumulation modeling
- Support for multiple tree species and forest types

### Wildlife Management (`forest_stack.common.scripts.forms.wildlife`)
Digital management tools including:
- **Animal Rescue Forms**: Structured data collection for wildlife rescue operations
- **Wildlife Census Tools**: Systematic census management and data recording
- Integration-ready TypeScript modules for web applications

## Data Processing & Web Tools

### Format Conversion
- **TIFF to PMTiles**: Automated conversion of raster data to web-optimized PMTiles format for efficient web mapping
- **Tile Management**: Comprehensive tree cover and canopy tile processing with coverage optimization
- Shell script automation for batch processing workflows

### Geospatial Utilities (`forest_stack.common`)
- Advanced geometry processing functions with Shapely integration
- Spatial data management tools with coordinate system handling
- Configuration management for API keys, STAC endpoints, and processing parameters
- Common forestry utilities for cross-module functionality

## Dependencies

The library requires **Python 3.8+** and includes these key dependencies:

### Core Geospatial Libraries
- `rasterio` (>=1.3) - Geospatial raster I/O and processing
- `pystac-client` (>=0.7) - STAC API client for satellite imagery access
- `shapely` (>=2.0) - Geometric operations and spatial analysis
- `pyproj` (>=3.6) - Coordinate reference system transformations
- `affine` (>=2.3) - Affine transformations for geospatial data

### Scientific Computing & Performance  
- `numpy` (>=1.23) - Numerical computing and array operations
- `aiohttp` (>=3.9) - Asynchronous HTTP client for API requests
- `psutil` (>=5.9) - System and process monitoring
- `tqdm` (>=4.66) - Progress bars and monitoring

### Additional Dependencies (Auto-installed)
- `xarray` - Multi-dimensional data arrays and NetCDF support
- `dask` - Parallel computing and distributed processing
- `geopandas` - Vector data processing with pandas integration
- `rioxarray` - Xarray integration with rasterio for enhanced raster processing
- `odc-stac` - Open Data Cube STAC loading utilities

## Quick Start Example

```python
from forest_stack.models.ndvi.ndvi import NDVIProcessor
from forest_stack.models.forest_mask.forest_mask import ForestMaskProcessor  
from forest_stack.models.canopy_density.canopy_density import CanopyDensityProcessor
from forest_stack.common.config.forest_config import Config

# Initialize configuration
config = Config()

# Process NDVI mosaic with advanced options
ndvi_processor = NDVIProcessor()
ndvi_result = ndvi_processor.process_ndvi_mosaic(
    aoi_geometry="path/to/rajasthan_boundary.geojson",
    year=2023,
    season="monsoon",  # Options: spring, monsoon, winter
    cloud_threshold=20,  # Maximum cloud coverage percentage
    output_format="pmtiles"  # Web-optimized output
)

# Generate forest mask with temporal analysis
forest_mask_processor = ForestMaskProcessor()
forest_mask = forest_mask_processor.generate_forest_mask(
    aoi_geometry="path/to/study_area.geojson",
    daterange="2023-01-01/2023-12-31",
    ndvi_threshold=0.32,  # Vegetation threshold
    use_lulc_integration=True
)

# Calculate canopy density with multi-spectral analysis
canopy_processor = CanopyDensityProcessor()
canopy_density = canopy_processor.process_canopy_density(
    aoi_geometry="path/to/forest_area.geojson",
    forest_mask_path="outputs/forest_mask_2023.tif",
    density_classes=4,  # Open, Low, Medium, High
    enable_dask_processing=True
)
```

## Configuration

Forest-Stack uses a centralized configuration system. Create a `config.json` file in your working directory:

```json
{
    "stac_catalog": "planetary-computer",
    "api_endpoints": {
        "planetary_computer": "https://planetarycomputer.microsoft.com/api/stac/v1",
        "element84": "https://earth-search.aws.element84.com/v1"
    },
    "processing": {
        "max_workers": 4,
        "memory_limit": "8GB",
        "tile_size": 2048
    },
    "output": {
        "default_format": "geotiff",
        "compression": "lzw",
        "pmtiles_zoom_levels": [0, 12]
    }
}
```

## Use Cases & Applications

### Forest Monitoring & Management
- **Deforestation Detection**: Multi-temporal analysis for identifying forest loss patterns
- **Reforestation Tracking**: Monitoring forest recovery and plantation success rates  
- **Canopy Health Assessment**: Seasonal vegetation health monitoring using NDVI time-series

### Climate & Environmental Assessment
- **Carbon Stock Estimation**: Integration of canopy density with carbon calculator for biomass assessment
- **Groundwater Impact Analysis**: Correlation of forest cover changes with groundwater trends
- **Ecosystem Service Mapping**: Comprehensive environmental service quantification

### Conservation & Wildlife Management
- **Habitat Mapping**: Species-specific habitat suitability analysis
- **Conservation Planning**: Protected area effectiveness assessment
- **Wildlife Corridor Analysis**: Connectivity mapping for wildlife movement

## Contributing

Forest-Stack is part of the **DataKaveri** ecosystem and welcomes community contributions! 

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/yourusername/Forest-Stack.git
cd Forest-Stack

# Create development environment
python -m venv dev-env && source dev-env/bin/activate
pip install -r requirements.txt -e .

# Run tests (when available)
pytest tests/
```

### Contribution Guidelines
- Follow PEP 8 style guidelines for Python code
- Use TypeScript best practices for .ts modules
- Include comprehensive docstrings and type hints
- Add unit tests for new functionality
- Update documentation and README for new features

### Module Documentation
Detailed usage instructions and examples are available in individual module README files:
- `forest-stack/models/ndvi/readme.md`
- `forest-stack/models/forest_mask/forest_mask_readme.md`  
- `forest-stack/models/canopy_density/canopy_density_readme.md`

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## Acknowledgments

Forest-Stack leverages several open-source projects and data sources:
- **Microsoft Planetary Computer** for STAC-based satellite data access
- **Element84 Earth Search** for alternative STAC catalog integration
- **Sentinel-2 Mission** (ESA) for high-resolution multispectral imagery
- **Open Data Cube** ecosystem for efficient geospatial data processing

---

**Maintained by**: [DataKaveri](https://github.com/datakaveri)  
**Project Status**: Active Development  
**Python Version**: 3.8+
