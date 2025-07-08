# NDVI Processor – Detailed Documentation

This module automates the end-to-end processing pipeline for generating Normalized Difference Vegetation Index (NDVI) mosaics over a region using Sentinel-2 imagery from the Element84 STAC API. It includes smart tile coverage management, cloud-masking, NDVI computation, and the generation of web-ready raster outputs.

---

## 1. What the script does

1. Prepares Directories & Loads Geometries:
   - Reads the simplified and convex-hull boundaries of Rajasthan.
   - Loads and filters Sentinel-2 MGRS tiles that intersect with the state boundary.
2. Searches the STAC API:
   - Queries STAC API (Earth Search via Element84) for each user-defined year and seasonal time windows (April–May & November–December).
   - Retrieves all qualifying Sentinel-2 L2A scenes for the region, applying a tiered cloud threshold and coverage scoring.
3. Optimizes Tile Coverage:
   - Scores individual scenes based on cloud cover and spatial overlap.
   - For tiles with partial coverage, performs a secondary search to fill in gaps using complementary geometries.
4. Downloads Scene Bands:
   - Downloads only the necessary bands (red, nir, scl) for NDVI computation.
   - Uses resumable, async downloads with caching for efficiency.
5. Calculates NDVI:
   - Processes all available scenes per tile.
   - Reprojects and aligns bands; masks invalid pixels using the SCL band.
6. Computes NDVI for valid pixels and merges them using a maximum-value strategy.
   - Generates NDVI Composites:
   - Merges all tile-level NDVI rasters into seasonal mosaics.
   - Reprojects and clips results to the Rajasthan state boundary.
   - Applies color-relief from a predefined colormap.
   - Prepares outputs as GeoTIFFs (optionally PMTiles/web-ready formats).
---

## 2. Folder expectations
forest-stack/
├── data/
│   └── ndvi/
│       ├── sentinel-2-rj-tiles.json        # Source tile geometries
│       ├── colormap-ndvi.txt               # Color-relief map for NDVI
│       ├── stac_cache/                     # Cached STAC responses
│       ├── sentinel-2-l2a/                 # Downloaded band images
│       ├── ndvi_tiles/                     # Tile-level NDVI outputs
│       ├── ndvi_state_composite/           # Final composite GeoTIFFs
│       └── tiles/                          # (Optional) Web map tiles
├── forest_stack/
│   └── models/
│       └── ndvi/
│           └── ndvi_processor.py           # This module
└── common/
    └── data/
        ├── rajasthan_state_simp.geojson    # Simplified boundary
        └── rajasthan-convex-hull.geojson   # Outer convex hull
---

## 3. Configuration Expectations

The following variables must be configured either in forest_config.py or passed explicitly:

- years	- List of years to compute NDVI for (e.g., [2017, ..., 2024])
- cache_dir	- Base directory for storing all outputs and cache (./data/ndvi)
- valid_tiles - Filtered tiles that intersect the Rajasthan state boundary (from tile JSON)
- time_ranges - Predefined date ranges: Apr–May & Nov–Dec for each year

Cloud thresholds (eo:cloud_cover < 20, 40, 60, 80, 100) and SCL pixel masks are hardcoded but can be configured if needed.
---

## 4. Inputs

- ./common/data/rajasthan_state_simp.geojson - Detailed boundary for clipping rasters
- ./common/data/rajasthan-convex-hull.geojson - Outer convex hull used to filter valid tiles
- ./data/ndvi/sentinel-2-rj-tiles.json - All available Sentinel-2 tiles (MGRS) in JSON format
---

## 5. Output

- ./data/ndvi/ndvi_tiles/MGRS-<tile>-ndvi.tif - NDVI raster for individual tiles, scaled 0–200
- ./data/ndvi/ndvi_state_composite/ndvi-*.tif - State-wide composite per season per year (clipped, EPSG:4326)
- ./data/ndvi/ndvi_state_composite/ndvi-*_colored.tif - Color-relief version using GDAL colormap
- ./data/ndvi/stac_cache/*.json - Cached responses of STAC queries per time range

NDVI raster values are scaled to uint8 range (0–200) to balance file size and interpretability.

