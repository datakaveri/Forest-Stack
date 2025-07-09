# Soil Moisture Processor – Detailed Documentation

This module calculates area-weighted soil moisture statistics over rangeland regions using multi-band annual TIFFs. It performs spatial extraction with geometric fidelity, caches band-wise outputs for performance, and generates SQL-compatible inserts to feed a PostgreSQL/PostGIS database.

---

## 1. What the script does

1. Region Geometry Extraction:
   - Queries the DuckDB-based regions2 table to extract all range-type polygons.
   - Converts them to a GeoPackage (range_regions.gpkg) for vectorized spatial processing via exactextract.
2. Raster Band Processing:
   - Iterates over each band in annual soil moisture TIFFs (typically bi-weekly/fn-wise stacks).
3. For each band:
   - Creates a temporary VRT.
   - Runs exactextract with area-weighted mean extraction per region.
   - Caches results as Parquet files and stores raster hashes to avoid reprocessing.
4. Parallelized Execution:
   - Uses ProcessPoolExecutor to process all bands of each TIFF in parallel, leveraging all available cores and memory.
5. SQL Insert Generation:
   - Assigns a snapshot label to each band based on year, month, and fortnight (fn1 or fn2).
   - Combines all region values across all bands and formats them into a SQL insert statement compatible with the data_product_snapshots_data table.
---

## 2. Folder expectations

```
forest-stack/
├── data/
│   ├── gis/
│   │   ├── rajasthan_state_simp.geojson      # Boundary of Rajasthan state
│   │   └── rajasthan-convex-hull.geojson     # Optional coarse outer boundary
│   └── forest_mask/
│       └── forest_mask.tif                   # Output binary forest mask
├── forest_stack/
│   ├── common/
│   │   ├── forest_config.py                  # API keys and config
│   │   └── forest_common.py                  # Shared util functions
│   └── models/
│       └── forest_mask/
│           ├── forest_mask.py                # Main processing script
│           └── forest_mask_readme.md         # Documentation (this file)
├── LICENSE
├── README.md
└── requirements.txt
```
---

## 3. Configuration Expectations

The following variables must be configured either in forest_config.py or passed explicitly:

- PLANETARY_SUB_KEY — Subscription key for Microsoft Planetary Computer.
- PLANETARY_STAC_URL — STAC API endpoint.
- LULC_START_YEAR — Base year for LULC data retrieval.
---

## 4. Inputs

- AOI File (.geojson): Polygon of interest (e.g., Rajasthan state).
- NDVI Source: Sentinel-2 imagery from Planetary Computer (bands: B04, B08, SCL).
- LULC Source: io-lulc-annual-v02 collection.
---

## 5. Output

Binary Raster (.tif) where:
 - 1 = Pixel is forest.
 - 0 = Pixel is not forest.


