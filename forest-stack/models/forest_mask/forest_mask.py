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
import xarray as xr
from dask import distributed
from dask.diagnostics import ProgressBar
from numba import njit, prange
from odc.stac import load
from rioxarray.merge import merge_arrays
from shapely.geometry import MultiPolygon, Polygon
from tqdm.notebook import tqdm
from pystac_client import Client
from forest_stack.common.forest_config import Config
from forest_stack.common.forest_common import compute_grids, catalog_element84

#values to check in this file
#0.123. 0.25, 0.32

# # # # # # # #
# INPUTS      #
# # # # # # # #
YEAR = 2023  # Latest year to compute forrest mask for
AOI_FILE_PATH = "../../common/data/rajasthan_state_simp.geojson"
GRID_SIZE = 0.07
OUTPUT_FILE_PATH = "./data/forest_mask/forrest_mask.tif"
############################################################################

planetary_computer.settings.set_subscription_key(Config.PLANETARY_SUB_KEY)

catalog_mpc = Client.open(
    Config.PLANETARY_STAC_URL,
    modifier=planetary_computer.sign_inplace,
)


@njit(parallel=True)
def max_rolling(x, window):
    pad_arr = np.zeros(shape=x.shape) + np.nan
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1] - window + 1):
            pad_arr[i, j + window - 1] = np.max(x[i, j : j + window])
    return pad_arr


@njit(parallel=True)
def mean_rolling(x, window):
    pad_arr = np.zeros(shape=x.shape) + np.nan
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1] - window + 1):
            pad_arr[i, j + window - 1] = np.mean(x[i, j : j + window])
    return pad_arr


@njit(parallel=True)
def interpolate_nan_along_axis(arr, axis=1):
    # Function to interpolate NaN values along a single axis
    def interpolate_axis(a):
        non_nan_indices = np.arange(len(a))[~np.isnan(a)]
        interp_func = np.interp(np.arange(len(a)), non_nan_indices, a[non_nan_indices])
        a[np.isnan(a)] = interp_func[np.isnan(a)]
        return a

    # Apply interpolation along the specified axis
    if axis == 0:
        for i in prange(arr.shape[1]):
            arr[:, i] = interpolate_axis(arr[:, i])
    elif axis == 1:
        for i in prange(arr.shape[0]):
            arr[i, :] = interpolate_axis(arr[i, :])
    return arr


@njit(parallel=True, cache=True)
def median_along_axis(arr):
    rows, cols = arr.shape
    result = np.empty(rows, dtype=np.float32)

    for i in prange(rows):
        valid_values = []
        for j in range(cols):
            if not np.isnan(arr[i, j]):  # Exclude NaN values
                valid_values.append(arr[i, j])

        valid_values = np.array(valid_values, dtype=np.float32)
        valid_values.sort()  # Sort valid (non-NaN) values
        n = valid_values.size

        if n == 0:  # Handle case where all values are NaN
            result[i] = np.nan
        else:
            mid = n // 2
            if n % 2 == 0:
                result[i] = (valid_values[mid - 1] + valid_values[mid]) / 2
            else:
                result[i] = valid_values[mid]

    return result


@njit(parallel=True, cache=False)
def get_max_consecutive_green_days(
    x, min_thr=0.25, max_roll_window=2, mean_roll_window=5
):
    """
    Compute Maximum Consecutive Green Days in NDVI Time Series.

    Processes a 2D NDVI time series (`x`) to calculate the maximum number of
    consecutive "green" days (NDVI > 0.25) per row after applying interpolation,
    thresholding, and smoothing.

    Parameters:
    -----------
    x : ndarray
        2D array of NDVI values (rows: spatial units, columns: time steps).
    min_thr : float, default=0.25
        Minimum NDVI threshold; values below are adjusted to min_thr.
    max_roll_window : int, default=2
        Window size for maximum rolling operation.
    mean_roll_window : int, default=5
        Window size for mean rolling operation.
    """
    xt = interpolate_nan_along_axis(x)
    for i in prange(len(x)):
        if (xt[i] > 0.20).sum() <= 3:
            x[i] = 0
    x = np.where(x <= min_thr, min_thr, x)
    x = np.where(np.isnan(x), 0, x)
    x = max_rolling(x, max_roll_window)
    x = np.where(x == 0, np.nan, x)
    x = interpolate_nan_along_axis(x)
    x = mean_rolling(x, mean_roll_window)
    x = interpolate_nan_along_axis(x[:, ::-1])
    left, center, right = (
        median_along_axis(x[:, 10:19]),
        median_along_axis(x[:, 19:24]),
        median_along_axis(x[:, 24:30]),
    )
    x = (x[:, ::-1] > 0.25).astype("uint8")
    n_consecutives = np.zeros(shape=(x.shape[0]))
    for n in prange(len(n_consecutives)):
        if (center[n] <= left[n]) and (center[n] <= right[n]) and (center[n] <= 0.32):
            n_consecutives[n] = 0
        else:
            arr = x[n]
            max_count = 0
            current_count = 0
            for i in arr:
                if i == 1:
                    current_count += 1
                    max_count = max(max_count, current_count)
                else:
                    current_count = 0
            n_consecutives[n] = max_count

    return n_consecutives


def get_max_consecutive_green_days_xarray(ds: xr.Dataset):
    """Compute Maximum Consecutive Green Days in NDVI Time Series and return it as an xarray Dataset

    Args:
        ds (xr.Dataset): Input dataset

    Returns:
        xr.Dataset: Ouput
    """
    out_coords = {"y": ds.y.values, "x": ds.x.values, "time": [ds.time.values[-1]]}
    for key in ds.coords:
        if key not in out_coords:
            out_coords[key] = ds.coords[key]

    dims = copy.deepcopy(ds.dims)
    out_coords = copy.deepcopy(out_coords)
    shape = copy.deepcopy(ds.data.shape)
    try:
        ci = get_max_consecutive_green_days(
            ds.data.reshape(-1, shape[2]),
            min_thr=0.123,
            max_roll_window=3,
            mean_roll_window=3,
        )
        return xr.DataArray(
            data=ci.reshape(shape[0], shape[1], 1), dims=dims, coords=out_coords
        )
    except:
        return xr.DataArray(
            data=np.zeros(shape=(shape[0], shape[1], 1)), dims=dims, coords=out_coords
        )


def download_ndvi_and_lulc(
    polygon: Polygon, daterange: str
) -> Tuple[xr.Dataset, xr.Dataset]:
    """Download and Process NDVI and LULC.
    It fetches LULC data from ESRI for configured LULC start year. If year is 2023 then we fetch LULC for 2023-05-01/2024-05-01

    Args:
        polygon (Polygon): The area of interest to download data for
        daterange (str): The date range to download data for

    Returns:
        Tuple[xr.Dataset, xr.Dataset]: NDVI, LULC
    """
    bbox = polygon.bounds
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
        .astype("float32")
        .transpose("y", "x", "time")
    )
    ds["NDVI"] = (ds.nir - ds.red) / (ds.red + ds.nir + 1e-3)
    ds_ndvi = ds.NDVI
    ds_scl = ds.scl
    ds_ndvi.data[
        (
            (ds_scl.data == 8)
            | (ds_scl.data == 9)
            | (ds_scl.data == 10)
            | (ds_ndvi.data == 0)
        )
    ] = np.nan

    for time_index in tqdm(range(len(ds_ndvi.time.values[:-1]))):
        curr_time = ds_ndvi.time.values[time_index]
        nearest_time = (
            ds_ndvi.time.values[time_index - 1]
            if (
                abs(ds_ndvi.time.values[time_index - 1] - curr_time)
                - abs(ds_ndvi.time.values[time_index + 1] - curr_time)
            )
            < 0
            else ds_ndvi.time.values[time_index + 1]
        )
        ds_ndvi.loc[:, :, curr_time] = xr.where(
            np.isnan(ds_ndvi.sel(time=curr_time)),
            ds_ndvi.sel(time=nearest_time),
            ds_ndvi.sel(time=curr_time),
        )

    times = pd.date_range(daterange.split("/")[0], daterange.split("/")[1], freq="10D")
    ds_ndvi = ds_ndvi.sel(time=times, method="nearest")
    ds_ndvi["time"] = times

    ds_lulc = None

    # LULC
    query = catalog_mpc.search(
        collections=["io-lulc-annual-v02"],
        datetime=f"{Config.LULC_START_YEAR}-05-01/{Config.LULC_START_YEAR + 1}-05-01",
        limit=100,
        bbox=bbox,
    )
    query = list(query.items())
    ds_lulc = (
        load(query, geopolygon=polygon, groupby="solar_day", bands=["data"], chunks={})
        .astype("float32")
        .transpose("y", "x", "time")["data"]
        .isel(time=0)
    )
    ds_lulc = ds_lulc.compute()
    ds_lulc = ds_lulc.rio.reproject(ds_ndvi.rio.crs)

    ds_lulc = ds_lulc.sel(x=ds_ndvi.x.values, y=ds_ndvi.y.values, method="nearest")
    ds_lulc["y"], ds_lulc["x"] = ds_ndvi.y.values, ds_ndvi.x.values

    with ProgressBar():
        ds_ndvi = ds_ndvi.compute(scheduler="threads")

    return ds_ndvi, ds_lulc

def compute_forrest_mask_dataset(
    ds_ndvi: xr.Dataset, ds_lulc: xr.Dataset
) -> xr.Dataset:
    """Compute forrest mask array, transpose to ("time", "y", "x") that is projected to input CRS

    Args:
        ds_ndvi (xr.Dataset): NDVI for the region
        ds_lulc (xr.Dataset): LULC of the region

    Returns:
        xr.Dataset: Forrest mask array
    """
    with ProgressBar():
        ds_forrest = ds_ndvi.chunk({"x": 1000, "y": 1000, "time": -1}).map_blocks(
            get_max_consecutive_green_days_xarray,
            template=xr.zeros_like(ds_ndvi.isel(time=[-1]), dtype="uint8").chunk(
                {"x": 1000, "y": 1000, "time": -1}
            ),
        )
        ds_forrest = ds_forrest.compute(scheduler="processes", num_workers=24)

    ds_forrest = xr.where(
        ds_lulc.isin([5, 6, 7, 8, 1]) & (ds_forrest <= 35), 0, ds_forrest
    )
    ds_forrest = xr.where(ds_forrest >= 18, 1, 0)
    ds_forrest = ds_forrest.transpose("time", "y", "x").rio.write_crs(ds_ndvi.rio.crs)

    return ds_forrest


def compute_forrest_mask(
    aoi_file_path: str, year: int, grid_size: float, output_file_path: str
):
    """Main method for Forrest Mask Computation

    Args:
        aoi_file_path (str): Path to the Area of Intereset vector file
        year (str): Latest year to compute forrest mask for
        grid_size (float): The grid size factor to divide the polygon into
        output_file_path (str): Path to the forrest mask `.tif` file. Must contain `.tif` equivalent suffix
    """
    # READ AOI
    df_rajasthan = gpd.read_file(aoi_file_path)
    polygon = df_rajasthan["geometry"][0]

    # COMPUTE POLYGON GRIDS
    polygons = compute_grids(polygon, grid_size)
    df_rajasthan_chunked = gpd.GeoDataFrame(
        [{"geometry": polygon} for polygon in polygons]
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        # COMPUTE FORREST LAYERS FOR GRIDS
        forrest_layers_dir = f"{temp_dir}/forrest-layers"
        os.makedirs(forrest_layers_dir)
        daterange = f"{year-1}-05-01/{year}-05-01"
        print(f"Starting process for {len(df_rajasthan_chunked)} polygon chunks.")
        for polygon_index in range(len(df_rajasthan_chunked)):
            polygon = df_rajasthan_chunked.iloc[polygon_index, -1]
            if type(polygon) == MultiPolygon:
                polygon = list(polygon.geoms)[
                    np.argmax([geom.area for geom in polygon.geoms])
                ]
            try:
                # DOWNLOAD NDVI AND LULC
                print(
                    f"Polygon Index : {polygon_index}  Year : {year} NDVI Downloading Started....."
                )
                ds_ndvi, ds_lulc = download_ndvi_and_lulc(polygon, daterange)
                print(f"Polygon Index : {polygon_index}  Year : {year} NDVI Downloaded")

                print(
                    f"Polygon Index : {polygon_index}  Year : {year} Forest Mask Computation Started....."
                )

                ds_forrest = compute_forrest_mask_dataset(ds_ndvi, ds_lulc)

                ds_forrest.rio.to_raster(
                    f"{forrest_layers_dir}/{polygon_index}.tif", compress="lzw"
                )
                print(
                    f"Polygon Index : {polygon_index}  Year : {year} Forest Mask Computation Computed"
                )

                del ds_forrest
                del ds_ndvi
                gc.collect()
            except Exception as e:
                print(f"Error computing forrest mask for polygon: {polygon.wkt}")
                print(f"Error >> {e}")
                print("Ignoring and continuing.")

        # MERGE ALL GRID FORREST LAYERS
        paths = glob.glob(f"{forrest_layers_dir}/*.tif")
        print(f"Count of gridded files found for merge: {len(paths)}")
        grid_datasets = [
            rioxarray.open_rasterio(path, chunks={})
            .rio.reproject("epsg:4326")
            .astype("uint8")
            for path in paths
        ]
        mosaic = merge_arrays(grid_datasets, nodata=0)
        mosaic = mosaic.astype("uint8")
        mosaic.rio.to_raster(output_file_path, compress="lzw")


if __name__ == "__main__":
    compute_forrest_mask(AOI_FILE_PATH, YEAR, GRID_SIZE, OUTPUT_FILE_PATH)
