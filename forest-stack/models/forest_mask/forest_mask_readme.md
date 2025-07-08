# Forest Mask Classifier – Detailed Documentation

This model computes a binary forest mask (0: non-forest, 1: forest) using Sentinel-2 NDVI time-series data and land-use land-cover (LULC) classification. The logic is designed to evaluate vegetation cyclicity and sustained greenness patterns over a defined seasonal window.

The model is built for the Rajasthan and can be generalized to other geographies

---

## 1. What the script does

1. Grid the Area of Interest (AOI): 
   - Divides the AOI polygon into smaller chunks for scalable NDVI processing. 
2. Download Sentinel-2 Imagery:
   - Queries Planetary Computer’s STAC API for Sentinel-2 L2A assets
   - Retrieves bands needed to compute NDVI (nir, red) and scene classification (scl).
   - Filters cloudy or invalid pixels using SCL mask (values 8, 9, 10).  
3. Compute NDVI Time Series:
   - Applies a rolling max and mean to smooth NDVI values
   - Interpolates missing values (NaNs) and thresholds below 0.25.
   - Calculates maximum number of consecutive “green” days based on NDVI cyclicity.  
4. Download and Align LULC Map:
   - Fetches annual LULC raster from ESRI (via Planetary Computer).
   - Clips and reprojects LULC data to NDVI grid.
5. Generate Forest Mask:
   - Retains pixels classified as forest (based on max green days ≥ 18 and valid LULC types).
   - Merges all chunked raster tiles to produce a final GeoTIFF mask.
---

## 2. Folder expectations

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


