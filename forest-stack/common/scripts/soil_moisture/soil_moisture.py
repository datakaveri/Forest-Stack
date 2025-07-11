import duckdb
import geopandas as gpd
from shapely import wkb
from datetime import datetime
import pandas as pd
import os
from pathlib import Path
import multiprocessing
import psutil
from tqdm import tqdm
import concurrent.futures
import rasterio
import hashlib

# Configuration
DB_PATH = "dfhms.db"
DATA_DIR = Path("./data")
TIFF_DIR = DATA_DIR / "soilmoisture_tiffs"
OUTPUT_DIR = DATA_DIR / "soilmoisture_output"
CACHE_DIR = DATA_DIR / "soilmoisture_cache"
PARQUET_CACHE_DIR = CACHE_DIR / "parquet"
OUTPUT_SQL = "soil_moisture_insert.sql"

# Performance configuration
N_CORES = multiprocessing.cpu_count()
MEMORY_LIMIT = int(psutil.virtual_memory().total * 0.75)

# Create directories
for dir_ in [OUTPUT_DIR, CACHE_DIR, PARQUET_CACHE_DIR]:
    dir_.mkdir(parents=True, exist_ok=True)

# Initialize database
con = duckdb.connect(DB_PATH)
con.execute(f"SET memory_limit='{MEMORY_LIMIT}B'")
con.execute(f"SET threads={N_CORES}")
con.execute(f"INSTALL spatial;")
con.execute(f"LOAD spatial;")

def get_file_hash(filepath):
    """Calculate file hash to detect changes"""
    BUF_SIZE = 65536
    sha1 = hashlib.sha1()
    
    with open(filepath, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha1.update(data)
    
    return sha1.hexdigest()

def get_regions_gpkg():
    """Export regions to GeoPackage for exactextract"""
    cache_path = CACHE_DIR / "range_regions.gpkg"
    
    # Check if cached regions file exists and is up to date
    regions_hash_file = CACHE_DIR / "regions_hash.txt"
    current_regions = con.execute("""
        SELECT code, ST_AsText(geom) as geom_text 
        FROM regions2 
        WHERE type = 'range'
        ORDER BY code
    """).fetchall()
    current_hash = hashlib.sha1(str(current_regions).encode()).hexdigest()
    
    cached_hash = None
    if regions_hash_file.exists():
        with open(regions_hash_file, 'r') as f:
            cached_hash = f.read().strip()
    
    if not cache_path.exists() or cached_hash != current_hash:
        print("Generating new regions GeoPackage...")
        df = con.execute("""
            SELECT code, ST_AsWKB(geom) as geom_wkb 
            FROM regions2 
            WHERE type = 'range'
        """).fetchdf()
        
        df['geom_wkb'] = df['geom_wkb'].apply(bytes)
        df['geometry'] = gpd.GeoSeries.from_wkb(df['geom_wkb'])
        df.drop(columns=['geom_wkb'], inplace=True)
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        
        # Export to GeoPackage
        gdf.to_file(cache_path, driver="GPKG")
        
        # Save hash
        with open(regions_hash_file, 'w') as f:
            f.write(current_hash)
    
    return cache_path

def get_cache_path(tiff_path, band_num):
    """Generate cache path for band results"""
    return PARQUET_CACHE_DIR / f"{tiff_path.stem}_band_{band_num}.parquet"

def process_raster_band(args):
    """Process a single band from a TIFF file with caching"""
    tiff_path, band_num, poly_path = args
    cache_path = get_cache_path(tiff_path, band_num)
    
    # Check cache as before...
    if cache_path.exists():
        tiff_hash = get_file_hash(tiff_path)
        cache_meta_path = cache_path.with_suffix('.meta')
        
        if cache_meta_path.exists():
            with open(cache_meta_path, 'r') as f:
                cached_hash = f.read().strip()
                if cached_hash == tiff_hash:
                    return pd.read_parquet(cache_path)
    
    # Create temporary VRT for the specific band
    vrt_path = CACHE_DIR / f"temp_band_{band_num}.vrt"
    os.system(f'gdalbuildvrt -b {band_num} "{vrt_path}" "{tiff_path}"')
    
    # Run exactextract with area weights
    out_csv = CACHE_DIR / f"{tiff_path.stem}_band_{band_num}.csv"
    cmd = (f'./exactextract '
           f'-r sm:"{vrt_path}" '  # Main raster
           f'-r weight:"{vrt_path}" '  # Weight raster (same as input for area weighting)
           f'-p "{poly_path}" '
           f'-f "code" '
           f'-s "weighted_mean(sm,weight)" '  # Use weighted mean with area weights
           f'-o "{out_csv}"')
    os.system(cmd)
    
    # Clean up VRT
    if vrt_path.exists():
        os.remove(vrt_path)
    
    # Read and process results
    if out_csv.exists():
        df = pd.read_csv(out_csv)
        df.columns = ['code', 'value']
        
        # Cache results
        df.to_parquet(cache_path)
        
        # Save tiff hash
        with open(cache_path.with_suffix('.meta'), 'w') as f:
            f.write(get_file_hash(tiff_path))
        
        # Clean up CSV
        os.remove(out_csv)
        
        return df
    
    return None

def process_annual_tiff(tiff_path, poly_path, year):
    """Process all bands in an annual TIFF file"""
    print(f"Processing {tiff_path.name}")
    
    # Get number of bands (fortnights) in the TIFF
    with rasterio.open(tiff_path) as src:
        n_bands = src.count
    
    # Check if all bands are cached and valid
    all_cached = True
    cached_results = []
    tiff_hash = get_file_hash(tiff_path)
    
    for band in range(1, n_bands + 1):
        cache_path = get_cache_path(tiff_path, band)
        cache_meta_path = cache_path.with_suffix('.meta')
        
        if not cache_path.exists() or not cache_meta_path.exists():
            all_cached = False
            break
        
        with open(cache_meta_path, 'r') as f:
            cached_hash = f.read().strip()
            if cached_hash != tiff_hash:
                all_cached = False
                break
        
        cached_results.append(pd.read_parquet(cache_path))
    
    if all_cached:
        print(f"Using cached results for {tiff_path.name}")
        return cached_results
    
    # Process each band in parallel
    results = []
    with tqdm(total=n_bands, desc=f"Processing bands for {year}") as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=N_CORES) as executor:
            futures = []
            for band in range(1, n_bands + 1):
                futures.append(executor.submit(process_raster_band, (tiff_path, band, poly_path)))
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
                pbar.update(1)
    
    return results

def generate_sql_insert(all_data, output_file):
    """Generate SQL insert statement for soil moisture """
    
    # Create SQL template
    sql_template = """-- Copy soil moisture  data to the database
    WITH region_codes AS (
        SELECT DISTINCT 
            id as region_id,
            code
        FROM public.regions 
        WHERE code IN ({codes})
        AND type = 'range'
        AND deleted_at IS NULL
    ),
    dataset_snapshots AS (
        SELECT 
            dps.id as snapshot_id,
            dps.label
        FROM public.data_product_snapshots dps
        JOIN public.data_products dp ON dp.id = dps.data_product_id
        WHERE dp.name = 'Soil moisture'
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
     # Filter out rows with NaN values
    valid_data = all_data.dropna(subset=['value'])

    # Get unique region codes
    codes = "'" + "', '".join(valid_data['code'].unique()) + "'"
    
    # Format values for SQL
    values_list = []
    for _, row in valid_data.iterrows():
        percentage_value = row['value'] * 100
        values_list.append(
            f"        ('{row['code']}', '{row['snapshot_label']}', {percentage_value:.2f})"
        )
    
    values = ",\n".join(values_list)
    sql = sql_template.format(codes=codes, values=values)
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(sql)
    
    print(f"SQL file generated successfully at {output_file}!")

def main():
    # Export regions to GeoPackage
    poly_path = get_regions_gpkg()
    
    # Process each annual TIFF
    all_results = []
    tiff_files = sorted(TIFF_DIR.glob("sm_*_annual_stack.tif"))
    
    for tiff_path in tiff_files:
        year = int(tiff_path.stem.split('_')[1])
        fortnight_results = process_annual_tiff(tiff_path, poly_path, year)
        
        # Add year and fortnight information
        for fn_num, result in enumerate(fortnight_results, 1):
            month = ((fn_num - 1) // 2) + 1
            is_second_fortnight = fn_num % 2 == 0
            month_name = datetime(year, month, 1).strftime("%B").lower()
            fn_label = "fn2" if is_second_fortnight else "fn1"
            
            result['snapshot_label'] = f"soil_moisture_{year}_{month_name}_{fn_label}"
            all_results.append(result)
    
    # Combine all results
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Generate SQL insert statement
    generate_sql_insert(final_df, OUTPUT_SQL)

if __name__ == "__main__":
    main()