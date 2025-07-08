# Forest Canopy Density (FCD) Classifier – Detailed Documentation

This module computes Forest Canopy Density (FCD) classes using multi-band Sentinel-2 data and a previously computed Forest Mask. It uses brightness, greenness, shadow, and vegetation indices to derive density classes across an Area of Interest (AOI).

It is optimized for tiled, large-area processing and is designed to run efficiently using parallelized NumPy and Dask/Xarray operations.

---

## 1. What the script does

1. Grid the AOI Polygon:
   - Divides the input geometry into smaller chunks for scalable processing.
2. Download Sentinel-2 Imagery:
   - Queries Planetary Computer’s Sentinel-2 L2A collection using the STAC API.
   - Retrieves red, green, blue, nir, swir16, and scl bands.
   - Applies cloud and quality masks based on SCL classification.
3. Select Cleanest Time Indices:
   - Computes a minimal set of scenes to maximize usable (non-cloudy) pixels.
   - Merges clean pixels across selected time indices.
4. Compute Forest Canopy Density:
   - Calculates NDVI, BSI, CSI, and VD (vegetation difference) using a custom formula.
   - Combines these into a Forest Canopy Density (FCD) index using Numba-accelerated functions.
5. Classify FCD Values:
   - Categorizes the FCD into the following classes:
      - 1 → Open Forest (0–30)
      - 2 → Low Density (31–55)
      - 3 → Medium Density (56–80)
      - 4 → High Density (>80)
6. Merge Gridded Results:
   - Reprojects all tile rasters to EPSG:4326.
   - Mosaics them into a single raster.
7. Apply Forest Mask:
   - Only retains FCD values where forest exists (from precomputed mask).
   - Non-forest pixels are set to 0.
---

## 2. Folder expectations
forest-stack/
├── data/
│   ├── input/
│   │   └── tiny_aoi_in_rj.geojson         # AOI polygon
│   └── output/
│       ├── forest_mask.tif                # Binary forest mask input
│       └── forest_density.tif             # Final classified FCD output
├── forest_stack/
│   ├── common/
│   │   ├── forest_config.py               # Config variables
│   │   └── forest_common.py               # Utility functions (e.g., catalog search, gridding)
│   └── models/
│       └── forest_density/
│           ├── forest_density.py          # Main computation script
│           └── forest_density_readme.md   # This README
├── requirements.txt
└── README.md
---

## 3. Configuration Expectations

The following variables must be configured either in forest_config.py or passed explicitly:

- PLANETARY_SUB_KEY	- API subscription key for Planetary Computer
- PLANETARY_STAC_URL - Base URL for STAC API endpoint
---

## 4. Inputs

- AOI File: GeoJSON polygon (tiny_aoi_in_rj.geojson)
- Satellite Data Source: Sentinel-2 L2A collection
- Bands Used:
  - blue, green, red, nir, swir16, scl
- Forest Mask: Binary GeoTIFF previously computed by forest mask module
---

## 5. Output

A single-classified GeoTIFF (forest_density.tif) with the following values:
- 0	- Non-forest
- 1	- Open forest
- 2	- Low density
- 3	- Medium density
- 4	- High density


