import copy
import gc
import glob
import os
import tempfile
from typing import List, Tuple

import dask
import geopandas as gpd
import numpy as np
import odc.geo
import pandas as pd
import planetary_computer
import rioxarray
import rioxarray as rxr
import xarray as xr
from dask import distributed
from dask.diagnostics import ProgressBar
from numba import njit, prange
from odc.stac import load
from pystac_client import Client
from rioxarray.merge import merge_arrays
from shapely.geometry import MultiPolygon, Polygon
from tqdm.notebook import tqdm

from forest_stack.common.config.forest_config import Config
from forest_stack.common.scripts.forest_common import compute_grids, catalog_element84

# # # # # # # #
# INPUTS      #
# # # # # # # #

#check these too (can change the input to data folder and create input and output folders inside data folder)
YEAR = 2023  # Latest year to compute forrest density for
AOI_FILE_PATH = "./tiny_aoi_in_rj.geojson"
GRID_SIZE = 0.4
OUTPUT_FILE_PATH = "./forrest_density.tif"
FOREST_MASK_FILE_PATH = "./forrest_mask.tif"
############################################################################


def find_minimum_mask_set(masks):
    N, H, W = masks.shape
    flattened_masks = masks.reshape(N, -1)
    covered, selected_masks = np.zeros(flattened_masks.shape[1], dtype=bool), []
    while not np.all(covered):
        best_mask_idx = np.argmax(np.sum(flattened_masks[:, ~covered], axis=1))
        if not np.any(flattened_masks[best_mask_idx, ~covered]):
            break
        covered |= flattened_masks[best_mask_idx]
        selected_masks.append(best_mask_idx)
    selected_masks = sorted(selected_masks, key=lambda x: x.sum(), reverse=True)
    return selected_masks


def fill_with_indices(ds, indices, cloud_index):
    combined = xr.Dataset()
    mask = xr.zeros_like(cloud_index[0], dtype=bool)
    for idx in indices:
        layer_mask = cloud_index[idx] & ~mask
        combined = combined.combine_first(ds.isel(time=idx).where(layer_mask))
        mask |= layer_mask
    return combined


@njit(cache=True, parallel=True)
def compute_fcd_array(red, green, blue, nir, swir):
    ndvi = (nir - red) / (nir + red + 1e-8)
    bsi = ((swir + blue) - (nir + red)) / ((swir + blue) + (nir + red) + 1e-8)

    max_green = np.max(green)
    max_red = np.max(red)
    max_ndvi = np.max(ndvi)
    min_ndvi = np.min(ndvi)

    csi = np.sqrt((max_green - green) * (max_red - red))
    vd = (ndvi - min_ndvi) / (
        max_ndvi - min_ndvi + 1e-8
    )  # Add small constant to avoid divide-by-zero
    ssi = np.sqrt(csi * np.abs(bsi))
    fcd = (100 * np.sqrt(vd * ssi)).astype("uint8")

    return fcd


def classify_fcd(fcd: xr.DataArray, state_code: str) -> xr.DataArray:
    """Apply FCD classification over the array with configurable thresholds

    Args:
        fcd (xr.DataArray): FCD Array to classify
        state_code (str, optional): State code for predefined thresholds. 
                                   Options: RJ (Rajasthan)

    Returns:
        xr.DataArray: Classified array with values:
                     0 = Non-forest
                     1 = Open forest
                     2 = Low density forest
                     3 = Medium density forest  
                     4 = High density forest
    """
    # Get thresholds from configuration
    config = Config()
    thresholds = config.get_fcd_thresholds(state_code)
    
    # Extract threshold values
    open_max = thresholds["open_forest"]["max"]
    low_min = thresholds["low_density"]["min"] 
    low_max = thresholds["low_density"]["max"]
    med_min = thresholds["medium_density"]["min"]
    med_max = thresholds["medium_density"]["max"]
    high_min = thresholds["high_density"]["min"]
    
    # Apply classification
    classes = np.zeros_like(fcd, dtype=np.uint8)
    classes[(fcd > 0) & (fcd <= open_max)] = 1  # Open forest
    classes[(fcd > low_min) & (fcd <= low_max)] = 2  # Low density
    classes[(fcd > med_min) & (fcd <= med_max)] = 3  # Medium density
    classes[fcd > high_min] = 4  # High density

    classes_da = xr.DataArray(
        classes,
        dims=fcd.dims,
        coords=fcd.coords,
        attrs={
            "description": "Forest canopy density classes",
            "state_code": state_code
        },
    )

    return classes_da


def compute_forrest_density(
    aoi_file_path: str,
    year: int,
    grid_size: float,
    output_file_path: str,
    forrest_mask_file_path: str,
    state_code: str
):
    """Compute Forest Canopy Density with configurable classification thresholds

    Args:
        aoi_file_path (str): Path to the Area Of Interest vector file
        year (int): Year to compute FCD of
        grid_size (float): Grid size for polygon chunking
        output_file_path (str): Path to store the computed FCD file. Path must contain .tif equivalent suffix
        forrest_mask_file_path (str): Path to the forest mask .tif file
        state_code (str): State code for predefined thresholds.
                                   Options: RJ (Rajasthan).
    """
    # READ AOI
    df_rajasthan = gpd.read_file(aoi_file_path)
    polygon = df_rajasthan["geometry"][0]

    # COMPUTE POLYGON GRIDS
    polygons = compute_grids(polygon, grid_size)
    polygons = polygons[:1]
    df_rajasthan_chunked = gpd.GeoDataFrame(
        [{"geometry": polygon} for polygon in polygons]
    )

    daterange = f"{year - 1}-10-01/{year - 1}-12-01"
    with tempfile.TemporaryDirectory() as temp_dir:
        density_layers_dir = f"{temp_dir}/density-layers"
        os.makedirs(density_layers_dir)
        print(f"Starting process for {len(df_rajasthan_chunked)} polygon chunks.")
        for polygon_index in range(len(df_rajasthan_chunked)):
            polygon = df_rajasthan_chunked.iloc[polygon_index, -1]
            if type(polygon) == MultiPolygon:
                polygon = list(polygon.geoms)[
                    np.argmax([geom.area for geom in polygon.geoms])
                ]
            bbox = polygon.bounds
            print(f"Polygon Index : {polygon_index} SCL Downloading Started...")
            query = catalog_element84.search(
                collections=["sentinel-2-l2a"], datetime=daterange, limit=100, bbox=bbox
            )
            query = list(query.items())
            ds = (
                load(
                    query,
                    geopolygon=polygon,
                    groupby="solar_day",
                    bands=["red", "green", "blue", "nir", "swir16", "scl"],
                    chunks={},
                )
                .astype("uint16")
                .transpose("y", "x", "time")
            )
            crs = ds.rio.crs
            with ProgressBar():
                scl = ds.scl.astype("uint8").compute(scheduler="threads")

            cloud_index = ~scl.isin([0, 3, 8, 9, 10, 0])
            cloud_index = cloud_index.transpose("time", "y", "x")
            indices = find_minimum_mask_set(cloud_index.data)

            print(f"Polygon Index : {polygon_index} FCC Downloading Started.....")
            with ProgressBar():
                ds = ds.isel(time=indices).compute(scheduler="threads")
            ds = fill_with_indices(
                ds, range(len(indices)), cloud_index.isel(time=indices)
            ).astype("float32")
            print(f"Polygon Index : {polygon_index} FCC Downloaded")

            print(
                f"Polygon Index : {polygon_index} Forest Mask Computation Started....."
            )
            ds = ds / 10000.0
            blue, green, red, nir, swir = (
                ds.blue.data,
                ds.green.data,
                ds.red.data,
                ds.nir.data,
                ds.swir16.data,
            )
            ds = xr.DataArray(
                data=compute_fcd_array(red, green, blue, nir, swir),
                dims=["y", "x"],
                coords={"y": ds.y.values, "x": ds.x.values},
            )
            ds_density = ds
            ds_density.rio.write_crs(crs).rio.to_raster(
                f"{density_layers_dir}/{polygon_index}.tif", compress="lzw"
            )
            print(f"Polygon Index : {polygon_index} Forest Mask Computation Computed")

            del ds_density
            del ds
            del cloud_index
            del scl
            gc.collect()
        
        # MERGE ALL GRID FORREST LAYERS
        paths = glob.glob(f"{density_layers_dir}/*.tif")
        print(f"Count of gridded files found for merge: {len(paths)}")
        grid_datasets = [
            rioxarray.open_rasterio(path, chunks={})
            .rio.reproject("epsg:4326")
            .astype("uint8")
            for path in paths
        ]
        ds_mosaiced_density = merge_arrays(grid_datasets, nodata=0)

        # CLASSIFY FCD
        ds_mosaiced_density = classify_fcd(ds_mosaiced_density, state_code)

        # ADD FORREST MASK
        ds_forest = rxr.open_rasterio(forrest_mask_file_path)
        ds_mosaiced_density = xr.where(ds_forest == 1, ds_mosaiced_density, 0)

        # SAVE
        ds_mosaiced_density = ds_mosaiced_density.astype("uint8")
        ds_mosaiced_density = ds_mosaiced_density.rio.write_nodata(0)
        ds_mosaiced_density.rio.to_raster(output_file_path, compress="lzw")


if __name__ == "__main__":
    # Example usage with Rajasthan state code
    compute_forrest_density(
        AOI_FILE_PATH, YEAR, GRID_SIZE, OUTPUT_FILE_PATH, FOREST_MASK_FILE_PATH,
        state_code="RJ"  # Use Rajasthan state code for FCD thresholds
    )
