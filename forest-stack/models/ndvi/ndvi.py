import os
import json
import hashlib
import asyncio
import traceback
from affine import Affine
import aiohttp
import logging
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

import pystac_client
import rasterio
import numpy as np
from tqdm import tqdm
import subprocess
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import json
from shapely.geometry import shape, box, mapping
from shapely.ops import unary_union
from typing import List, Dict, Any, Tuple, Optional
import psutil
from osgeo import gdal
import gc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TimeRange:
    start_date: str
    end_date: str
    name: str

class NDVIProcessor:
    def __init__(self, years: List[int], cache_dir: str = "./data/ndvi"):
        self.years = years
        self.cache_dir = Path(cache_dir)
        self.stac_cache_dir = self.cache_dir / "stac_cache"
        self.raw_data_dir = self.cache_dir / "sentinel-2-l2a"
        self.ndvi_tiles_dir = self.cache_dir / "ndvi_tiles"
        self.composite_dir = self.cache_dir / "ndvi_state_composite"
        self.tiles_dir = self.cache_dir / "tiles"
        
        # Create necessary directories
        for dir_path in [self.stac_cache_dir, self.raw_data_dir, self.ndvi_tiles_dir, 
                        self.composite_dir, self.tiles_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Load district geometry
        with open('../../common/data/rajasthan-convex-hull.geojson') as f:
            self.district_geom = json.load(f)['features'][0]['geometry']

        with open('../../common/data/rajasthan_state_simp.geojson') as f:
            self.district_geom_detailed = json.load(f)['features'][0]['geometry']
            self.district_shape = shape(self.district_geom_detailed)
        
        # Load source tile geometries and clip them by district boundary
        self.source_tile_geometries = {}
        with open('./data/ndvi/sentinel-2-rj-tiles.json') as f:
            for line in f:
                if line.strip():
                    feature = json.loads(line)
                    tile_id = feature['properties']['Name']
                    tile_geom = shape(feature['geometry'])
                    
                    # Clip tile geometry by district boundary
                    clipped_geom = tile_geom.intersection(self.district_shape)
                    
                    # Only include tiles that intersect with the district
                    if not clipped_geom.is_empty:
                        self.source_tile_geometries[tile_id] = clipped_geom
            
        self.valid_tiles = [
            '42RWQ', '42RWR', '42RXN', '42RXP', '42RXQ', '42RXR', 
            '42RXS', '42RYN', '42RYP', '42RYQ', '42RYR', '42RYS', 
            '43QCF', '43QCG', '43QDF', '43QDG', '43QEG', '43QFG', 
            '43RBH', '43RBJ', '43RBK', '43RBL', '43RBM', '43RBN', 
            '43RCH', '43RCJ', '43RCK', '43RCL', '43RCM', '43RCN', 
            '43RCP', '43RDH', '43RDJ', '43RDK', '43RDL', '43RDM', 
            '43RDN', '43RDP', '43REH', '43REJ', '43REK', '43REL', 
            '43REM', '43REN', '43RFH', '43RFJ', '43RFK', '43RFL', 
            '43RFM', '43RGH', '43RGJ', '43RGK', '43RGL', '44RKQ'
        ]
        
        # Update valid_tiles list to only include tiles that intersect with district
        self.valid_tiles = list(self.source_tile_geometries.keys())
        logger.info(f"Found {len(self.valid_tiles)} valid tiles intersecting with district boundary")
        
        # Fixed list comprehension syntax
        self.time_ranges = [
            period for year in years 
            for period in [
                TimeRange(f"{year}-04-01", f"{year}-05-31", f"{year}_apr-may"),
                TimeRange(f"{year}-11-01", f"{year}-12-31", f"{year}_nov-dec")
            ]
        ]

    def find_uncovered_areas(self, source_geom: shape, response_geom: shape) -> List[shape]:
        """
        Find areas of the source geometry not covered by the response geometry.
        Applies a buffer to response geometry to handle minor misalignments.
        Both geometries should already be clipped to district boundary.
        """
        try:
            # Get the centroid to determine UTM zone
            centroid = response_geom.centroid
            lon, lat = centroid.x, centroid.y
            
            # Calculate UTM zone
            utm_zone = int((lon + 180) / 6) + 1
            epsg_code = f'326{utm_zone}' if lat >= 0 else f'327{utm_zone}'  # North/South hemisphere
            
            # Project to UTM
            import pyproj
            from shapely.ops import transform
            
            proj_utm = pyproj.CRS(f'EPSG:{epsg_code}')
            proj_wgs84 = pyproj.CRS('EPSG:4326')
            project = pyproj.Transformer.from_crs(proj_wgs84, proj_utm, always_xy=True).transform
            project_back = pyproj.Transformer.from_crs(proj_utm, proj_wgs84, always_xy=True).transform
            
            # Convert geometries to UTM
            source_utm = transform(project, source_geom)
            response_utm = transform(project, response_geom)
            
            # Apply buffer in meters
            BUFFER_SIZE = 1000  # meters
            buffered_response_utm = response_utm.buffer(BUFFER_SIZE)
            
            # Get the difference
            uncovered_utm = source_utm.difference(buffered_response_utm)
            
            # Convert back to WGS84
            uncovered = transform(project_back, uncovered_utm)
            
            # Filter out small polygons that might be artifacts
            MIN_AREA = 1000  # square meters
            
            if uncovered.is_empty:
                return []
            elif uncovered.geom_type == 'Polygon':
                # Convert to UTM again to check area in square meters
                uncovered_utm = transform(project, uncovered)
                if uncovered_utm.area >= MIN_AREA:
                    return [uncovered]
                else:
                    logger.debug(f"Filtered out small polygon of area {uncovered_utm.area:.2f} m²")
                    return []
            elif uncovered.geom_type == 'MultiPolygon':
                # Filter individual polygons by area
                significant_polygons = []
                for poly in uncovered.geoms:
                    poly_utm = transform(project, poly)
                    if poly_utm.area >= MIN_AREA:
                        significant_polygons.append(poly)
                if significant_polygons:
                    logger.debug(f"Filtered out {len(uncovered.geoms) - len(significant_polygons)} small polygons")
                return significant_polygons
            else:
                logger.warning(f"Unexpected geometry type: {uncovered.geom_type}")
                return []
                
        except Exception as e:
            logger.error(f"Error in find_uncovered_areas: {str(e)}")
            return []

    async def search_additional_coverage(
        self,
        uncovered_area: shape,
        time_range: TimeRange,
        existing_items: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Search for additional tiles to cover an uncovered area with caching.
        The uncovered_area is already clipped to district boundary.
        
        Args:
            uncovered_area: The geometry of the uncovered area
            time_range: The time range to search within
            existing_items: Already selected items to avoid duplicates
            
        Returns:
            Best matching item or None if no suitable match found
        """
        async def perform_search(search_geom: dict, cloud_threshold: int) -> List[Dict[str, Any]]:
            """Helper function to perform search with caching"""
            # Create cache key based on search parameters
            cache_key = hashlib.md5(
                f"{json.dumps(search_geom)}_{time_range.start_date}_{time_range.end_date}_{cloud_threshold}".encode()
            ).hexdigest()
            
            cache_file = self.stac_cache_dir / f"additional_search_{cache_key}.json"
            
            # Check cache first
            if cache_file.exists():
                try:
                    with open(cache_file) as f:
                        cached_results = json.load(f)
                        logger.info(f"Using cached results for additional search with cloud threshold {cloud_threshold}%")
                        return cached_results
                except Exception as e:
                    logger.warning(f"Error reading cache file: {e}")
                    # Continue with actual search if cache read fails
            
            # Perform actual searchF
            try:
                catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
                search = catalog.search(
                    collections=["sentinel-2-l2a"],
                    intersects=search_geom,
                    datetime=f"{time_range.start_date}/{time_range.end_date}",
                    query={
                        "eo:cloud_cover": {"lt": cloud_threshold},
                        "s2:degraded_msi_data_percentage": {"lt": 2}
                    }
                )
                
                collection = search.item_collection()
                results = [item.to_dict() for item in collection.items]
                
                # Cache the results
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(results, f)
                except Exception as e:
                    logger.warning(f"Error writing to cache file: {e}")
                
                return results
            except Exception as e:
                logger.error(f"Error performing search: {e}")
                return []
        
        try:
            # Convert shapely geometry to GeoJSON for the search
            search_geom = mapping(uncovered_area)
            
            # First attempt with lower cloud cover threshold
            search_results = await perform_search(search_geom, 20)
            
            # Process results
            candidates = []
            existing_ids = {item['id'] for item in existing_items}
            
            for item_dict in search_results:
                if item_dict['id'] not in existing_ids:
                    item_geom = shape(item_dict['geometry'])
                    if item_geom.intersects(uncovered_area):
                        item_geom = item_geom.intersection(self.district_shape)
                        intersection = item_geom.intersection(uncovered_area)
                        coverage_ratio = intersection.area / uncovered_area.area
                        candidates.append({
                            'item': item_dict,
                            'cloud_cover': item_dict['properties'].get('eo:cloud_cover', 100),
                            'coverage_ratio': coverage_ratio,
                            'intersection': intersection
                        })
            
            if candidates:
                # First filter by geometry - find items that provide best coverage
                max_coverage = max(c['coverage_ratio'] for c in candidates)
                best_coverage_candidates = [
                    c for c in candidates 
                    if c['coverage_ratio'] >= 0.9 * max_coverage
                ]
                
                if best_coverage_candidates:
                    best_candidate = min(best_coverage_candidates, key=lambda x: x['cloud_cover'])
                    logger.info(f"Found candidate with coverage {best_candidate['coverage_ratio']:.2%} "
                              f"and cloud cover {best_candidate['cloud_cover']}%")
                    return best_candidate['item']
            
            # If no suitable candidates found, try with higher cloud cover threshold
            if not candidates:
                logger.info("No candidates found with initial cloud cover threshold, trying with higher threshold")
                search_results = await perform_search(search_geom, 50)
                candidates = []
                
                for item_dict in search_results:
                    if item_dict['id'] not in existing_ids:
                        item_geom = shape(item_dict['geometry'])
                        if item_geom.intersects(uncovered_area):
                            item_geom = item_geom.intersection(self.district_shape)
                            intersection = item_geom.intersection(uncovered_area)
                            coverage_ratio = intersection.area / uncovered_area.area
                            candidates.append({
                                'item': item_dict,
                                'cloud_cover': item_dict['properties'].get('eo:cloud_cover', 100),
                                'coverage_ratio': coverage_ratio,
                                'intersection': intersection
                            })
                
                if candidates:
                    max_coverage = max(c['coverage_ratio'] for c in candidates)
                    best_coverage_candidates = [
                        c for c in candidates 
                        if c['coverage_ratio'] >= 0.9 * max_coverage
                    ]
                    
                    if best_coverage_candidates:
                        best_candidate = min(best_coverage_candidates, key=lambda x: x['cloud_cover'])
                        logger.info(f"Found candidate with higher cloud cover: coverage {best_candidate['coverage_ratio']:.2%}, "
                                  f"cloud cover {best_candidate['cloud_cover']}%")
                        return best_candidate['item']
            
            return None
            
        except Exception as e:
            logger.error(f"Error searching additional coverage: {e}")
            return None

    def validate_tile_coverage(self, tile_id: str, response_geometry: dict, accepting_best_available: bool = False) -> Tuple[bool, float]:
        """
        Validate tile coverage with option to accept best available coverage.
        Returns (is_valid, coverage_ratio)
        """
        if tile_id not in self.source_tile_geometries:
            return False, 0.0
            
        source_geom = self.source_tile_geometries[tile_id]
        response_geom = shape(response_geometry)
        
        # Clip response geometry by district boundary
        response_geom = response_geom.intersection(self.district_shape)
        
        # Calculate coverage ratio
        coverage_ratio = source_geom.intersection(response_geom).area / source_geom.area
        
        # Use lower threshold and accept best available if specified
        COVERAGE_THRESHOLD = 0.60  # Accept 60% coverage
        
        if coverage_ratio < COVERAGE_THRESHOLD and not accepting_best_available:
            logger.debug(f"Tile {tile_id} coverage: {coverage_ratio:.2%}")
            return False, coverage_ratio
            
        return True, coverage_ratio

    async def download_bands(self, search_results: Dict[str, List[Dict[str, Any]]]) -> bool:
        """Download required bands for each tile including SCL band for quality masking"""
        async def download_file(session: aiohttp.ClientSession, url: str, output_path: Path, progress_bar) -> bool:
            temp_path = output_path.with_suffix('.tmp')
            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"Failed to download {url}: HTTP {response.status}")
                        return False
                    
                    total_size = int(response.headers.get('content-length', 0))
                    progress_bar.total = total_size
                    progress_bar.refresh()
                    
                    with open(temp_path, 'wb') as f:
                        downloaded = 0
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                            downloaded += len(chunk)
                            progress_bar.update(len(chunk))
                    
                    temp_path.rename(output_path)
                    return True
                    
            except Exception as e:
                logger.error(f"Error downloading {url}: {str(e)}")
                return False
            finally:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)

        async with aiohttp.ClientSession() as session:
            for time_range_name, items in tqdm(search_results.items(), 
                                            desc="Processing time periods", 
                                            position=0):
                output_dir = self.raw_data_dir / time_range_name
                output_dir.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"Processing {len(items)} items for {time_range_name}")
                
                # Track all downloads for this time period
                pending_downloads = []
                for item in items:
                    item_id = item['id']
                    id_parts = item_id.split('_')
                    if len(id_parts) < 3:
                        continue
                    
                    tile_id = id_parts[1]
                    date = datetime.strptime(id_parts[2], '%Y%m%d').strftime('%Y%m%d')
                    
                    # Check for required bands including SCL
                    for band in ['red', 'nir', 'scl']:
                        if band not in item['assets']:
                            logger.warning(f"Missing {band} band for tile {tile_id}")
                            continue
                        
                        output_path = output_dir / f"MGRS-{tile_id}-{date}-{band}.tif"
                        if not output_path.exists():
                            href = item['assets'][band]['href']
                            pending_downloads.append((href, output_path, f"{tile_id}-{band}"))
                
                if pending_downloads:
                    logger.info(f"Starting {len(pending_downloads)} downloads for {time_range_name}")
                    
                    # Create progress bars for each download
                    progress_bars = {
                        name: tqdm(
                            total=0,  # Will be updated when download starts
                            desc=f"Downloading {name}",
                            unit='B',
                            unit_scale=True,
                            position=i+1  # Position below the main progress bar
                        )
                        for i, (_, _, name) in enumerate(pending_downloads)
                    }
                    
                    # Create limited size download pool
                    semaphore = asyncio.Semaphore(5)
                    
                    async def download_with_semaphore(url, path, name):
                        async with semaphore:
                            return await download_file(session, url, path, progress_bars[name])
                    
                    # Start all downloads with semaphore control
                    download_tasks = [
                        download_with_semaphore(url, path, name) 
                        for url, path, name in pending_downloads
                    ]
                    
                    try:
                        # Wait for all downloads to complete
                        results = await asyncio.gather(*download_tasks, return_exceptions=True)
                        
                        # Check results
                        successful = sum(1 for r in results if r is True)
                        failed = len(results) - successful
                        
                        logger.info(f"Downloads complete for {time_range_name}:")
                        logger.info(f"  - Successful: {successful}")
                        logger.info(f"  - Failed: {failed}")
                    
                    finally:
                        # Close all progress bars
                        for pbar in progress_bars.values():
                            pbar.close()
                else:
                    logger.info(f"No new downloads needed for {time_range_name}")
                
        # Verify we have complete pairs after all downloads
        all_pairs_complete = True
        for time_range_name in search_results.keys():
            output_dir = self.raw_data_dir / time_range_name
            file_pairs = defaultdict(set)
            
            for file in output_dir.glob("MGRS-*-*-*.tif"):
                try:
                    parts = file.stem.split('-')
                    if len(parts) != 4:
                        continue
                    base = "-".join(parts[:-1])
                    band = parts[-1]
                    file_pairs[base].add(band)
                except Exception as e:
                    logger.error(f"Error processing filename {file}: {str(e)}")
                    continue
            
            complete_pairs = sum(1 for bands in file_pairs.values() if 'red' in bands and 'nir' in bands)
            logger.info(f"Found {complete_pairs} complete band pairs in {time_range_name}")
            if complete_pairs == 0:
                all_pairs_complete = False
        
        if not all_pairs_complete:
            logger.error("No complete pairs of bands found after downloads")
            return False
            
        return True

    def create_composite(self) -> bool:
        """Create mosaic and clip to district boundary using parallel processing"""
        try:
            # Get available CPU cores and memory
            cpu_count = os.cpu_count()
            total_memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
            memory_limit = int(total_memory_gb * 0.75)
            
            logger.info(f"Using {cpu_count} CPU cores and {memory_limit}GB memory limit")
            
            gdal.SetCacheMax(memory_limit * 1024 * 1024 * 1024)
            
            for time_range_name in os.listdir(self.ndvi_tiles_dir):
                input_dir = self.ndvi_tiles_dir / time_range_name
                output_path = self.composite_dir / f"ndvi-{time_range_name}.tif"
                
                if output_path.exists():
                    logger.info(f"Composite already exists for {time_range_name}, skipping...")
                    continue
                
                source_files = list(input_dir.glob("*.tif"))
                if not source_files:
                    logger.warning(f"No source files found in {input_dir}, skipping...")
                    continue
                    
                logger.info(f"Found {len(source_files)} source files for {time_range_name}")
                
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count) as executor:
                        src_files = [rasterio.open(p) for p in source_files]
                        dst_crs = 'EPSG:4326'
                        
                        def reproject_file(src):
                            try:
                                if src.crs != dst_crs:
                                    logger.info(f"Reprojecting {src.name} from {src.crs} to {dst_crs}")
                                    temp_path = Path(str(src.name) + '_reprojected.tif')
                                    
                                    transform, width, height = calculate_default_transform(
                                        src.crs, dst_crs, src.width, src.height, *src.bounds)
                                    
                                    profile = src.profile.copy()
                                    profile.update({
                                        'crs': dst_crs,
                                        'transform': transform,
                                        'width': width,
                                        'height': height,
                                        'tiled': True,
                                        'blockxsize': 512,
                                        'blockysize': 512,
                                        'compress': 'lzw',
                                        'num_threads': 'ALL_CPUS'
                                    })
                                    
                                    with rasterio.Env(GDAL_NUM_THREADS='ALL_CPUS',
                                                    RASTERIO_NUM_THREADS='ALL_CPUS'):
                                        with rasterio.open(temp_path, 'w', **profile) as dst:
                                            reproject(
                                                source=rasterio.band(src, 1),
                                                destination=rasterio.band(dst, 1),
                                                src_transform=src.transform,
                                                src_crs=src.crs,
                                                dst_transform=transform,
                                                dst_crs=dst_crs,
                                                resampling=Resampling.nearest,
                                                num_threads=cpu_count
                                            )
                                    
                                    src.close()
                                    return rasterio.open(temp_path)
                                return src
                            except Exception as e:
                                logger.error(f"Error reprojecting {src.name}: {e}")
                                return None
                        
                        future_to_file = {executor.submit(reproject_file, src): src for src in src_files}
                        reprojected_files = []
                        
                        for future in concurrent.futures.as_completed(future_to_file):
                            src = future_to_file[future]
                            try:
                                result = future.result()
                                if result is not None:
                                    reprojected_files.append(result)
                            except Exception as e:
                                logger.error(f"Error processing {src.name}: {e}")
                        
                        logger.info(f"Successfully reprojected {len(reprojected_files)} files")
                        
                        # Modified merge operation without tile_size parameter
                        with rasterio.Env(GDAL_NUM_THREADS='ALL_CPUS',
                                        RASTERIO_NUM_THREADS='ALL_CPUS'):
                            mosaic, transform = merge(reprojected_files, method='max', nodata=0, dtype='uint8')
                        
                        meta = reprojected_files[0].meta.copy()
                        meta.update({
                            "height": mosaic.shape[1],
                            "width": mosaic.shape[2],
                            "transform": transform,
                            "compress": "lzw",
                            "tiled": True,
                            "blockxsize": 512,
                            "blockysize": 512
                        })
                        
                        with rasterio.open(output_path, "w", **meta) as dest:
                            dest.write(mosaic)
                        
                        # Close all files
                        for src in reprojected_files:
                            src.close()
                            # Remove temporary reprojected files
                            if '_reprojected.tif' in src.name:
                                Path(src.name).unlink()
                        
                        # Clip to district boundary with optimized settings
                        logger.info(f"Clipping to district boundary")
                        with rasterio.Env(GDAL_NUM_THREADS='ALL_CPUS',
                                        RASTERIO_NUM_THREADS='ALL_CPUS'):
                            with rasterio.open(output_path) as src:
                                out_image, out_transform = mask(src, [self.district_geom_detailed], crop=True)
                                out_meta = src.meta.copy()
                                out_meta.update({
                                    "height": out_image.shape[1],
                                    "width": out_image.shape[2],
                                    "transform": out_transform,
                                    "tiled": True,
                                    "blockxsize": 512,
                                    "blockysize": 512,
                                    "compress": "lzw"
                                })
                                
                                clipped_path = output_path.parent / f"{output_path.stem}_clipped.tif"
                                with rasterio.open(clipped_path, "w", **out_meta) as dest:
                                    dest.write(out_image)
                        
                        # Apply colormap with optimized GDAL settings
                        colored_path = output_path.parent / f"{output_path.stem}_colored.tif"
                        gdal_options = [
                            "-co", "TILED=YES",
                            "-co", "BLOCKXSIZE=512",
                            "-co", "BLOCKYSIZE=512",
                            "-co", "COMPRESS=LZW",
                            "-co", "NUM_THREADS=ALL_CPUS"
                        ]
                        
                        subprocess.run([
                            "gdaldem",
                            "color-relief",
                            "-alpha",
                            "-of", "GTiff"] + 
                            gdal_options + [
                            str(clipped_path),
                            "./data/ndvi/colormap-ndvi.txt",
                            str(colored_path)
                        ], check=True)
                        
                        # Create mbtiles with optimized settings
                        # mbtiles_path = self.tiles_dir / f"ndvi-{time_range_name}.mbtiles"
                        # subprocess.run([
                        #     "rio", "mbtiles",
                        #     str(colored_path),
                        #     str(mbtiles_path),
                        #     "--format", "WEBP",
                        #     "--co", "LOSSLESS=TRUE",
                        #     "--co", "QUALITY=60",
                        #     "--progress-bar",
                        #     "--zoom-levels", "0..10",
                        #     "--tile-size", "512"
                        # ], check=True)
                        
                        # # Convert to pmtiles
                        # pmtiles_path = self.tiles_dir / f"ndvi-{time_range_name}.pmtiles"
                        # subprocess.run([
                        #     "./pmtiles", "convert",
                        #     str(mbtiles_path),
                        #     str(pmtiles_path)
                        # ], check=True)
                        
                except Exception as e:
                    logger.error(f"Error in composite creation: {str(e)}")
                    return False
                
        except Exception as e:
            logger.error(f"Error in composite creation: {str(e)}")
            return False

    def get_valid_pixel_mask(self, scl_data):
        """
        Create a mask for valid pixels based on the Scene Classification Layer (SCL).
        Valid pixels are typically classified as vegetation, bare soil, or water.
        
        Args:
            scl_data: numpy array of SCL band values.

        Returns:
            numpy array: Boolean mask where True indicates valid pixels.
        """
        # Define the valid pixel classes according to Sentinel-2 Scene Classification
        # 4 = Vegetation, 5 = Not Vegetated, 6 = Water
        valid_classes = [4, 5, 6]
        return np.isin(scl_data, valid_classes)

    def reproject_to_reference(self, ds, profile, ref_transform, ref_crs):
        data = ds.read(1).astype(np.float32)
        if profile is None:
            profile = ds.profile.copy()
            ref_transform = ds.transform
            ref_crs = ds.crs
        else:
            if ds.crs != ref_crs or not np.array_equal(ds.transform, ref_transform):
                data = self.reproject_band(data, ds.transform, ds.crs, ref_transform, ref_crs)
        return data

    def merge_ndvi_arrays(self, ndvi_arrays, valid_masks):
        ndvi_stack = np.stack(ndvi_arrays)
        mask_stack = np.stack(valid_masks)
        final_ndvi = np.zeros_like(ndvi_arrays[0])
        final_mask = np.zeros_like(valid_masks[0], dtype=bool)

        valid_pixels = mask_stack.any(axis=0)
        if valid_pixels.any():
            masked_ndvi = np.ma.masked_array(ndvi_stack, ~mask_stack)
            final_ndvi[valid_pixels] = np.ma.max(masked_ndvi[:, valid_pixels], axis=0).data
            final_mask[valid_pixels] = True

        return final_ndvi, final_mask

    def save_ndvi_to_file(self, output_path, ndvi, mask, profile, transform, crs):
        """
        Save the NDVI array to a file as uint8 with scaled values (0–200).
        
        Args:
            output_path: Path to save the file.
            ndvi: The NDVI array.
            mask: Boolean mask indicating valid pixels.
            profile: Metadata profile.
            transform: Affine transform for the dataset.
            crs: Coordinate Reference System.
        """
        if profile is None:
            logger.error(f"Cannot save NDVI to {output_path}: Missing metadata profile.")
            return

        # Scale NDVI to range [0, 200] and convert to uint8
        ndvi_scaled = np.zeros_like(ndvi, dtype=np.uint8)
        ndvi_scaled[mask] = np.clip((ndvi[mask] + 1) * 100, 0, 200).astype(np.uint8)

        # Update profile for uint8 data type
        profile.update({
            'dtype': 'uint8',
            'count': 1,
            'compress': 'lzw',
            'nodata': 0,
            'transform': transform,
            'crs': crs
        })

        # Save the scaled NDVI to a file
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(ndvi_scaled, 1)
        logger.info(f"Saved NDVI to {output_path} as uint8 with values scaled to 0–200.")

    def resample_to_match(self, source_data, source_transform, source_crs, target_shape, target_transform, target_crs):
        """
        Resample a source array to match the target shape and georeferencing.

        Args:
            source_data: numpy array of the source data.
            source_transform: Affine transform of the source data.
            source_crs: CRS of the source data.
            target_shape: Shape (height, width) of the target array.
            target_transform: Affine transform of the target array.
            target_crs: CRS of the target array.

        Returns:
            numpy array: Resampled source array.
        """
        resampled_data = np.zeros(target_shape, dtype=source_data.dtype)
        reproject(
            source=source_data,
            destination=resampled_data,
            src_transform=source_transform,
            src_crs=source_crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear
        )
        return resampled_data

    def check_georeferencing(self, file_path):
        """
        Check if a file has georeferencing metadata.

        Args:
            file_path (str): Path to the file to check.

        Returns:
            bool: True if the file has georeferencing, False otherwise.
        """
        try:
            with rasterio.open(file_path) as src:
                if src.crs is None or src.transform == Affine.identity():
                    logger.warning(f"File missing georeferencing: {file_path}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Error checking georeferencing for {file_path}: {e}")
            return False

    def process_tile_items(self, tile_id: str, time_range_name: str) -> bool:
        """
        Process and merge all available files for a single tile, ignoring file dates.
        This ensures all files in the folder are used to fill gaps in the NDVI output.
        """
        try:
            input_dir = self.raw_data_dir / time_range_name
            output_dir = self.ndvi_tiles_dir / time_range_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"MGRS-{tile_id}-ndvi.tif"

            if output_path.exists():
                logger.info(f"NDVI already processed for tile {tile_id}, skipping...")
                return True

            # Find all files for the given tile, regardless of date
            tile_files = list(input_dir.glob(f"MGRS-{tile_id}-*.tif"))
            if not tile_files:
                logger.warning(f"No files found for tile {tile_id} in {input_dir}")
                return False

            logger.info(f"Found {len(tile_files)} files for tile {tile_id}")

            # Group files by band (red, nir, scl)
            bands = defaultdict(list)
            for file in tile_files:
                if "-red" in file.name:
                    bands['red'].append(file)
                elif "-nir" in file.name:
                    bands['nir'].append(file)
                elif "-scl" in file.name:
                    bands['scl'].append(file)

            # Ensure all bands are available for processing
            if not all(bands.values()):
                logger.warning(f"Missing required bands for tile {tile_id}, skipping...")
                return False

            ndvi_arrays = []
            valid_masks = []
            profile = None
            ref_transform = None
            ref_crs = None

            # Process each set of files for the tile
            for red_path, nir_path, scl_path in zip(bands['red'], bands['nir'], bands['scl']):
                try:
                    with rasterio.open(red_path) as red_ds, \
                        rasterio.open(nir_path) as nir_ds, \
                        rasterio.open(scl_path) as scl_ds:

                        # Check metadata for each file
                        if red_ds.profile is None or nir_ds.profile is None or scl_ds.profile is None:
                            logger.error(f"Missing metadata for one of the files: {red_path}, {nir_path}, {scl_path}")
                            continue

                        # Read and reproject bands to a common grid
                        red = red_ds.read(1).astype(np.float32)
                        nir = nir_ds.read(1).astype(np.float32)
                        scl = scl_ds.read(1).astype(np.float32)

                        if profile is None:
                            profile = red_ds.profile.copy()
                            ref_transform = red_ds.transform
                            ref_crs = red_ds.crs

                        # Resample bands to the same resolution and alignment
                        target_shape = red.shape
                        nir = self.resample_to_match(nir, nir_ds.transform, nir_ds.crs, target_shape, ref_transform, ref_crs)
                        scl = self.resample_to_match(scl, scl_ds.transform, scl_ds.crs, target_shape, ref_transform, ref_crs)

                        # Validate shapes before NDVI calculation
                        if red.shape != nir.shape or red.shape != scl.shape:
                            logger.error(f"Shape mismatch for tile {tile_id}: Red {red.shape}, NIR {nir.shape}, SCL {scl.shape}")
                            continue

                        # Mask invalid pixels based on the Scene Classification Layer (SCL)
                        valid_mask = self.get_valid_pixel_mask(scl)
                        ndvi = np.zeros_like(red, dtype=np.float32)
                        calc_mask = valid_mask & (red + nir > 0)

                        if calc_mask.any():
                            ndvi[calc_mask] = (nir[calc_mask] - red[calc_mask]) / (nir[calc_mask] + red[calc_mask])
                            ndvi_arrays.append(ndvi)
                            valid_masks.append(calc_mask)

                except Exception as e:
                    logger.error(f"Error processing files {red_path}, {nir_path}, {scl_path}: {str(e)}")
                    continue

            if not ndvi_arrays:
                logger.error(f"No valid NDVI calculations for tile {tile_id}")
                return False

            # Merge NDVI arrays using the maximum value approach
            final_ndvi, final_mask = self.merge_ndvi_arrays(ndvi_arrays, valid_masks)
        
            # Save the final NDVI to the output file
            if profile is not None:
                self.save_ndvi_to_file(output_path, final_ndvi, final_mask, profile, ref_transform, ref_crs)
                logger.info(f"NDVI successfully processed for tile {tile_id}")
                return True
            else:
                logger.error(f"Cannot save NDVI for tile {tile_id}: Missing metadata")
                return False

        except Exception as e:
            print(traceback.format_exc())
            logger.error(f"Error processing tile {tile_id}: {str(e)}")
            return False

    def calculate_ndvi(self, search_results: Dict[str, List[Dict[str, Any]]]) -> bool:
        """Calculate NDVI with memory optimization and EPSG:4326 output"""

        
        try:
            # Get available system memory and calculate workers
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024 * 1024 * 1024)

            # Limit max workers to 12
            max_workers = min(12, max(1, int(available_gb / 2)), os.cpu_count())
            logger.info(f"Using {max_workers} workers based on {available_gb:.1f}GB available memory")

            for time_range_name, items in search_results.items():
                logger.info(f"Processing {time_range_name}")

                # Group items by tile ID
                tile_groups = defaultdict(list)
                for item in items:
                    id_parts = item['id'].split('_')
                    if len(id_parts) >= 2:
                        tile_id = id_parts[1]
                        tile_groups[tile_id].append(item)

                logger.info(f"Found {len(tile_groups)} tiles to process")

                # Process tiles in smaller batches
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    batch_size = max_workers * 2  # Process 2 batches per worker

                    for batch_start in range(0, len(tile_groups), batch_size):
                        batch_items = list(tile_groups.items())[batch_start:batch_start + batch_size]

                        # Submit tasks with the correct arguments
                        for tile_id, tile_items in batch_items:
                            futures.append(
                                executor.submit(self.process_tile_items, tile_id, time_range_name)
                            )

                        # Wait for this batch to complete
                        successful = 0
                        for future in tqdm(
                            concurrent.futures.as_completed(futures),
                            total=len(futures),
                            desc=f"Calculating NDVI batch {batch_start // batch_size + 1}"
                        ):
                            if future.result():
                                successful += 1

                        # Clear futures list for next batch
                        futures = []

                        # Force garbage collection
                        gc.collect()

                logger.info(f"Processing summary for {time_range_name}:")
                logger.info(f"  - Total tiles: {len(tile_groups)}")
                logger.info(f"  - Successfully processed: {successful}")

            return True

        except Exception as e:
            logger.error(f"Error in NDVI calculation: {str(e)}")
            return False


    async def search_stac(self) -> Dict[str, List[Dict[str, Any]]]:
        """Search STAC API with geometric complement handling for partial coverage"""
        catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
        results = {}

        CLOUD_THRESHOLDS = [20, 40, 60, 80, 100]
        MIN_ACCEPTABLE_COVERAGE = 0.60
        GOOD_COVERAGE_THRESHOLD = 0.95

        for time_range in tqdm(self.time_ranges, desc="Searching STAC API"):
            cache_file = self.stac_cache_dir / f"search_{time_range.name}.json"

            if cache_file.exists():
                with open(cache_file) as f:
                    cached_results = json.load(f)
                    logger.info(f"Using cached results for {time_range.name} ({len(cached_results)} items)")
                    results[time_range.name] = cached_results
                    continue

            logger.info(f"\nProcessing time period: {time_range.name}")
            logger.info(f"Target: {len(self.valid_tiles)} tiles need coverage")

            tile_items = defaultdict(list)
            best_coverage = defaultdict(float)
            remaining_tiles = set(self.valid_tiles)
            tiles_needing_additional = set()

            # First pass: Get primary coverage
            for cloud_threshold in CLOUD_THRESHOLDS:
                if not remaining_tiles:
                    break

                logger.info(f"\nAttempting search with {cloud_threshold}% cloud threshold:")
                logger.info(f"- Searching for {len(remaining_tiles)} tiles with no coverage yet")

                try:
                    search = catalog.search(
                        collections=["sentinel-2-l2a"],
                        intersects=self.district_geom,
                        datetime=f"{time_range.start_date}/{time_range.end_date}",
                        query={
                            "eo:cloud_cover": {"lt": cloud_threshold},
                            "s2:degraded_msi_data_percentage": {"lt": 5}
                        }
                    )

                    # Using  items() instead of async operations
                    items = [item.to_dict() for item in search. items()]
                    initial_count = len(remaining_tiles)

                    for item in items:
                        item_id = item.get('id', '')
                        id_parts = item_id.split('_')
                        if len(id_parts) >= 2:
                            tile_id = id_parts[1]
                            if tile_id in remaining_tiles:
                                item_geom = shape(item['geometry'])
                                source_geom = self.source_tile_geometries[tile_id]

                                intersection = item_geom.intersection(source_geom)
                                coverage = intersection.area / source_geom.area

                                if coverage > best_coverage[tile_id]:
                                    best_coverage[tile_id] = coverage
                                    tile_items[tile_id] = [item]

                                    if coverage > GOOD_COVERAGE_THRESHOLD:
                                        remaining_tiles.remove(tile_id)
                                    elif coverage > MIN_ACCEPTABLE_COVERAGE:
                                        tiles_needing_additional.add(tile_id)
                                        if tile_id in remaining_tiles:
                                            remaining_tiles.remove(tile_id)

                    tiles_found = initial_count - len(remaining_tiles)
                    good_coverage = sum(1 for cov in best_coverage.values() if cov > GOOD_COVERAGE_THRESHOLD)
                    partial_coverage = sum(1 for cov in best_coverage.values()
                                        if MIN_ACCEPTABLE_COVERAGE < cov <= GOOD_COVERAGE_THRESHOLD)

                    logger.info(f"Results at {cloud_threshold}% threshold:")
                    logger.info(f"- Found new coverage for {tiles_found} tiles")
                    logger.info(f"- Tiles with >95% coverage: {good_coverage}")
                    logger.info(f"- Tiles with 60-95% coverage: {partial_coverage}")
                    logger.info(f"- Tiles still with no coverage: {len(remaining_tiles)}")

                except Exception as e:
                    logger.error(f"Error in primary search at {cloud_threshold}% threshold: {e}")

            # Second pass: Search for complementary coverage
            if tiles_needing_additional:
                logger.info(f"\nStarting complementary coverage search:")
                logger.info(f"- {len(tiles_needing_additional)} tiles need additional coverage to reach 95%")

                for tile_id in tiles_needing_additional:
                    try:
                        source_geom = self.source_tile_geometries[tile_id]
                        covered_geom = unary_union([shape(item['geometry'])
                                                    for item in tile_items[tile_id]])
                        uncovered_geom = source_geom.difference(covered_geom)

                        if not uncovered_geom.is_empty:
                            uncovered_area_pct = (uncovered_geom.area / source_geom.area) * 100
                            logger.info(f"Tile {tile_id}: Searching for {uncovered_area_pct:.1f}% missing coverage")

                            search = catalog.search(
                                collections=["sentinel-2-l2a"],
                                intersects=mapping(uncovered_geom),
                                datetime=f"{time_range.start_date}/{time_range.end_date}"
                            )

                            # Using  items() for complementary search as well
                            complement_items = [item.to_dict() for item in search. items()]

                            if complement_items:
                                initial_coverage = best_coverage[tile_id]
                                items_added = 0
                                final_coverage = initial_coverage

                                for item in complement_items:
                                    item_geom = shape(item['geometry'])
                                    new_geom = unary_union([covered_geom, item_geom])
                                    new_coverage = new_geom.intersection(source_geom).area / source_geom.area

                                    if new_coverage > final_coverage + 0.05:
                                        tile_items[tile_id].append(item)
                                        items_added += 1
                                        final_coverage = new_coverage
                                        best_coverage[tile_id] = new_coverage

                                    if final_coverage > GOOD_COVERAGE_THRESHOLD:
                                        break

                                if items_added > 0:
                                    logger.info(f"- Improved {tile_id} coverage: {initial_coverage:.1%} → {final_coverage:.1%} "
                                                f"using {items_added} additional items")

                    except Exception as e:
                        logger.error(f"Error searching complementary coverage for tile {tile_id}: {e}")

            # Compile and cache results
            search_results = []
            for items in tile_items.values():
                search_results.extend(items)

            with open(cache_file, 'w') as f:
                json.dump(search_results, f)

            results[time_range.name] = search_results

            # Final summary
            logger.info(f"\nFinal coverage summary for {time_range.name}:")
            logger.info(f"Total tiles processed: {len(self.valid_tiles)}")
            logger.info(f"Coverage statistics:")

            coverage_ranges = {
                '95-100%': 0,
                '80-95%': 0,
                '60-80%': 0,
                '<60%': 0
            }

            multi_item_tiles = sum(1 for items in tile_items.values() if len(items) > 1)

            for tile_id, items in tile_items.items():
                coverage = best_coverage[tile_id]
                if coverage >= 0.95:
                    coverage_ranges['95-100%'] += 1
                elif coverage >= 0.80:
                    coverage_ranges['80-95%'] += 1
                elif coverage >= 0.60:
                    coverage_ranges['60-80%'] += 1
                else:
                    coverage_ranges['<60%'] += 1

            for range_name, count in coverage_ranges.items():
                logger.info(f"- {range_name}: {count} tiles")

            logger.info(f"Tiles using multiple images: {multi_item_tiles}")
            logger.info(f"Total images selected: {len(search_results)}\n")

        return results

    async def process(self):
        """Run the complete processing pipeline"""
        try:
            # Step 1: Search STAC
            logger.info("Step 1: Searching STAC API")
            search_results = await self.search_stac()
            
            # Step 2: Download bands
            logger.info("Step 2: Downloading bands")
            download_success = await self.download_bands(search_results)
            if not download_success:
                logger.error("Download step failed, stopping process")
                return
            
            # Step 3: Calculate NDVI
            logger.info("Step 3: Calculating NDVI")
            ndvi_success = self.calculate_ndvi(search_results)
            if not ndvi_success:
                logger.error("NDVI calculation failed, stopping process")
                return
            
            # Step 4: Create composites
            logger.info("Step 4: Creating composites and tiles")
            composite_success = self.create_composite()
            if not composite_success:
                logger.error("Composite creation failed")
                return
            
            logger.info("Processing complete!")
            
        except Exception as e:
            logger.error(f"Process failed: {str(e)}")
            raise

def main():
    """Synchronous main function"""
    try:
        # Create processor instance
        processor = NDVIProcessor(years=[2017,2018,2019,2020,2021,2022,2023,2024])
        # processor = NDVIProcessor(years=[2017])
        
        # Run the async process in the event loop
        asyncio.run(processor.process())
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
