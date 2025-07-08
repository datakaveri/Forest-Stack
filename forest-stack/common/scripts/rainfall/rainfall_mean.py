import duckdb
import rasterio
import exactextract
import pandas as pd
import geopandas as gpd
from shapely import wkb
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import os
import warnings
from osgeo import gdal, osr

# Configure GDAL to use exceptions
gdal.UseExceptions()
osr.UseExceptions()

# Suppress specific GDAL warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='osgeo')

# Configure paths
DATA_DIR = Path("data/rainfall_data")

# Calculate cutoff date (30 years ago from today)
today = datetime.now()
cutoff_date = today - timedelta(days=30*365)

def process_tif_file(tif_file, subdistricts):
    """Process a single TIF file and return its rainfall data"""
    date = datetime.strptime(tif_file.name.split('_')[0], '%Y%m%d')
    rainfall_data = []
    
    try:
        with rasterio.open(tif_file) as src:
            means = exactextract.exact_extract(src, subdistricts, ['mean'])
            
            for j, feature in enumerate(means):
                mean_value = feature['properties']['mean']
                if not pd.isna(mean_value):
                    rainfall_data.append({
                        'code': subdistricts.iloc[j]['code'],
                        'date': date,
                        'value': mean_value,
                        'month': date.month,
                        'day': date.day
                    })
    except Exception as e:
        print(f"Error processing {tif_file}: {str(e)}")
        return []
    # Explicitly Free up ram
    del means
    return rainfall_data

def main():
    # Connect to the database
    conn = duckdb.connect('dfhms.db')
    conn.execute("INSTALL spatial;")
    conn.execute("LOAD spatial;")

    # Get subdistricts data
    print("Loading subdistricts data...")
    subdistricts_df = conn.execute("""
        SELECT code, name, ST_AsWKB(geom) as geom_wkb 
        FROM regions2 
        WHERE geom IS NOT NULL AND type = 'range'
    """).fetchdf()

    subdistricts_df['geom_wkb'] = subdistricts_df['geom_wkb'].apply(bytes)
    subdistricts_df['geometry'] = gpd.GeoSeries.from_wkb(subdistricts_df['geom_wkb'])
    subdistricts = gpd.GeoDataFrame(
        subdistricts_df.drop('geom_wkb', axis=1), 
        geometry='geometry', 
        crs="EPSG:4326"
    )

    # List and filter TIF files
    tif_files = [f for f in sorted(DATA_DIR.glob('*.tif')) 
                 if datetime.strptime(f.name.split('_')[0], '%Y%m%d') >= cutoff_date]
    total_files = len(tif_files)
    print(f"Found {total_files} files to process from {cutoff_date.year} onwards")

    # Set up multiprocessing
    num_cores = 1 #mp.cpu_count()
    pool = mp.Pool(num_cores)
    process_func = partial(process_tif_file, subdistricts=subdistricts)

    # Process files with progress bar
    print(f"Processing files using {num_cores} cores...")
    results = []
    with tqdm(total=total_files, desc="Processing rainfall files") as pbar:
        for result in pool.imap_unordered(process_func, tif_files):
            results.extend(result)
            pbar.update()

    pool.close()
    pool.join()

    print(f"Processed {len(results)} measurements")

    # Convert to DataFrame
    print("Calculating normal rainfall values...")
    df = pd.DataFrame(results)
    normal_rainfall = df.groupby(['code', 'month', 'day'])['value'].mean().reset_index()

    # Create SQL template
    sql_template = """-- Copy normal rainfall data to the database
    WITH region_codes AS (
        SELECT DISTINCT 
            id as region_id,
            code
        FROM public.regions 
        WHERE code IN ({codes})
        AND deleted_at IS NULL
    ),
    dataset_snapshots AS (
        SELECT 
            dps.id as snapshot_id,
            dps.label
        FROM public.data_product_snapshots dps
        JOIN public.data_products dp ON dp.id = dps.data_product_id
        WHERE dp.name = 'Normal rainfall'
        AND dps.deleted_at IS NULL
        AND dp.deleted_at IS NULL
    ),
    raw_data(region_code, snapshot_label, value) AS (
        VALUES
    {values}
    )
    INSERT INTO public.data_product_snapshots_data
    (data_product_snapshot_id, region_id, value, created_at, updated_at)
    SELECT 
        gs.snapshot_id,
        rc.region_id,
        to_jsonb(rd.value),
        NOW(),
        NOW()
    FROM raw_data rd
    JOIN region_codes rc ON rc.code = rd.region_code
    JOIN dataset_snapshots gs ON gs.label = rd.snapshot_label;
    """

    # Format values for SQL
    print("Generating SQL file...")
    codes = "'" + "', '".join(normal_rainfall['code'].unique()) + "'"
    values_list = []

    for code in tqdm(normal_rainfall['code'].unique(), desc="Processing regions"):
        code_data = normal_rainfall[normal_rainfall['code'] == code]
        for month in range(1, 13):
            month_data = code_data[code_data['month'] == month]
            
            # First fortnight (days 1-15)
            fn1_mean = month_data[month_data['day'] <= 15]['value'].sum()
            if not pd.isna(fn1_mean):
                month_name = datetime(2000, month, 1).strftime('%b').lower()
                values_list.append(
                    f"        ('{code}', 'normal_rainfall_{month_name}_fn1', {fn1_mean:.2f})"
                )
            
            # Second fortnight (days 16-end)
            fn2_mean = month_data[month_data['day'] > 15]['value'].sum()
            if not pd.isna(fn2_mean):
                month_name = datetime(2000, month, 1).strftime('%b').lower()
                values_list.append(
                    f"        ('{code}', 'normal_rainfall_{month_name}_fn2', {fn2_mean:.2f})"
                )

    values = ",\n".join(values_list)
    sql = sql_template.format(codes=codes, values=values)

    with open('normal_rainfall_insert.sql', 'w') as f:
        f.write(sql)

    print("SQL file generated successfully!")

if __name__ == '__main__':
    main()