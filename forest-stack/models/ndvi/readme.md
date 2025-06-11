# NDVI Processor – Detailed Documentation

This module generates cloud-filtered, district-clipped NDVI composites from Sentinel-2 L2A imagery.  
It is designed for large‐area, multi-year processing and has been tuned for Rajasthan Forest-Stack but can be adapted to other regions.

---

## 1. What the script does

1. Query Sentinel-2 STAC for each user-supplied year and time window (Apr–May, Nov–Dec by default).  
2. Select best-coverage scenes per Sentinel MGRS tile (≥60 % coverage, prioritising ≤20 % cloud).  
3. Download red (B04), NIR (B08) and SCL bands to a local cache using async HTTP and resumable chunks.  
4. Compute per-scene NDVI and merge multiple scenes per tile, taking the max NDVI per pixel.  
5. Create a state-wide mosaic, reproject to EPSG:4326, clip to the Rajasthan boundary, and write an 8-bit GeoTIFF.  
6. (Optional) Apply colour-relief and convert to MBTiles/PMTiles for web-map serving.

---

## 2. Folder expectations

