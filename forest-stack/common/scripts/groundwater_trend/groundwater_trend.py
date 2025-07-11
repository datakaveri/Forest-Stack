import duckdb
import geopandas as gpd
from shapely import wkb
from datetime import datetime, timedelta
import pandas as pd
import os
import subprocess
from pathlib import Path
from tqdm import tqdm
import multiprocessing
import psutil
import concurrent.futures
import numpy as np
from functools import partial
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
DB_PATH = "dfhms.db"
REGIONS_TABLE = "regions2"
USERNAME = os.getenv('USERNAME')
PASSWORD = os.getenv('PASSWORD')
DATA_DIR = Path("./data")
NC_DIR = DATA_DIR / "groundwater_trends_nc"
TIFF_DIR = DATA_DIR / "groundwater_trends_tiff"
CSV_DIR = DATA_DIR / "groundwater_trends_csv"
CACHE_DIR = DATA_DIR / "cache/groundwater_trends"
OUTPUT_CSV = DATA_DIR / "groundwater_trends_out.csv"
FINAL_TABLE = "groundwater_trends"

# Performance configuration
N_CORES = multiprocessing.cpu_count()
# Use 75% of available RAM (in bytes)
MEMORY_LIMIT = int(psutil.virtual_memory().total * 0.75)
# Calculate chunk size based on available memory (assuming 1GB per chunk as baseline)
CHUNK_SIZE = max(1, MEMORY_LIMIT // (1024**3))

# Create all necessary directories
for dir_ in [NC_DIR, TIFF_DIR, CSV_DIR, CACHE_DIR]:
    dir_.mkdir(parents=True, exist_ok=True)

# Configure DuckDB to use maximum available memory
con = duckdb.connect(DB_PATH)
con.execute(f"SET memory_limit='{MEMORY_LIMIT}B'")
con.execute(f"SET threads={N_CORES}")
df = con.execute(f"INSTALL spatial;")
df = con.execute(f"LOAD spatial;")

# Load polygons from DuckDB into a DataFrame
df = con.execute(f"SELECT code, ST_AsWKB(geom) as geom_wkb FROM {REGIONS_TABLE} where type='range'").fetchdf()
df['geom_wkb'] = df['geom_wkb'].apply(bytes)
df['geometry'] = gpd.GeoSeries.from_wkb(df['geom_wkb'])
df.drop(columns=['geom_wkb'], inplace=True)
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

# Export polygons to a GeoPackage for exactextract
POLY_PATH = DATA_DIR / "regions2.gpkg"
gdf.to_file(POLY_PATH, driver="GPKG")

def get_year_range(start_date: str, end_date: str):
    """Get the range of years between two dates."""
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    return range(start_year, end_year + 1)

def get_cache_path(year: int) -> Path:
    """Generate cache file path for a given year."""
    return CACHE_DIR / f"measurements_{year}.parquet"

def process_tiff_file(args):
    """Process a single TIFF file and return the results DataFrame."""
    tiff_path, poly_path = args
    fname = tiff_path.name
    date_str = fname.split(".A")[1].split(".022")[0]
    out_csv = CSV_DIR / f"{fname}.csv"
    
    if not out_csv.exists():
        cmd = (f'./exactextract -r INDEX:{tiff_path} '
               f'-p {poly_path} -f "code" -s "mean(INDEX)" '
               f'-o {out_csv}')
        os.system(cmd)
        
        df = pd.read_csv(out_csv)
        df['DATE'] = date_str
        df['INDEX_mean'] = df['INDEX_mean'] * 0.1
        df.to_csv(out_csv, index=False)
    
    return pd.read_csv(out_csv)

def calculate_daily_means_for_year(year: int) -> pd.DataFrame:
    """Calculate daily means for a specific year and cache the results."""
    cache_file = get_cache_path(year)
    
    if cache_file.exists():
        print(f"Loading cached data for year {year}")
        return pd.read_parquet(cache_file)
    
    year_str = str(year)
    tiffs = sorted([t for t in TIFF_DIR.glob("*.tif") if year_str in t.name])
    
    # Process files in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=N_CORES) as executor:
        args = [(tiff, POLY_PATH) for tiff in tiffs]
        results = list(tqdm(
            executor.map(process_tiff_file, args, chunksize=CHUNK_SIZE),
            total=len(tiffs),
            desc=f"Processing {year} data"
        ))
    
    if results:
        year_df = pd.concat(results, ignore_index=True)
        # Save to parquet with compression
        year_df.to_parquet(cache_file, compression='snappy')
        return year_df
    return pd.DataFrame()

def parallel_download(date_list):
    """Download multiple files in parallel."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_CORES) as executor:
        futures = []
        for current_date in date_list:
            year_ = current_date.strftime("%Y")
            month_ = current_date.strftime("%m")
            day_str = current_date.strftime("%Y%m%d")
            
            url = (f"https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/"
                  f"GLDAS_CLSM025_DA1_D_EP.2.2/{year_}/{month_}/"
                  f"GLDAS_CLSM025_DA1_D_EP.A{day_str}.022.nc4")
            
            nc_file = NC_DIR / f"GLDAS_CLSM025_DA1_D_EP.A{day_str}.022.nc4"
            if not nc_file.exists():
                download_cmd = f'wget --no-verbose -N --user {USERNAME} --password {PASSWORD} "{url}" -P {NC_DIR}'
                futures.append(executor.submit(os.system, download_cmd))
        
        for future in concurrent.futures.as_completed(futures):
            future.result()

def download_data(start_date: str, end_date: str):
    start = datetime.strptime(start_date, '%Y%m%d').date()
    end = datetime.strptime(end_date, '%Y%m%d').date()
    date_list = [start + timedelta(days=x) for x in range((end-start).days + 1)]
    
    # Download in parallel
    parallel_download(date_list)
    
    # Convert to TIFF in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_CORES) as executor:
        futures = []
        for current_date in date_list:
            day_str = current_date.strftime("%Y%m%d")
            nc_file = NC_DIR / f"GLDAS_CLSM025_DA1_D_EP.A{day_str}.022.nc4"
            tiff_file = TIFF_DIR / f"GLDAS_CLSM025_DA1_D.A{day_str}.022.tif"
            
            if not tiff_file.exists() and nc_file.exists():
                convert_cmd = f'gdal_translate -of GTiff NETCDF:"{nc_file}":GWS_tavg "{tiff_file}"'
                futures.append(executor.submit(subprocess.run, convert_cmd, shell=True, check=True))
        
        for future in concurrent.futures.as_completed(futures):
            future.result()

def parallel_process_fortnight(group_data):
    """Process a fortnight group in parallel."""
    def fortnight_label(row):
        month_start = datetime(row['year'], row['month'], 1)
        if row['day'] <= 15:
            start_date = month_start
            end_date = month_start + timedelta(days=14)
            fn = "fn1"
        else:
            start_date = month_start + timedelta(days=15)
            next_month = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)
            month_end = next_month - timedelta(days=1)
            end_date = month_end
            fn = "fn2"

        month_name = month_start.strftime("%B").lower()
        label = f"groundwater_trend_{row['year']}_{month_name}_{fn}"
        return pd.Series({'fn_label': fn, 'start_date': start_date, 'end_date': end_date, 'label': label})
    
    return group_data.apply(fortnight_label, axis=1)

def aggregate_fortnightly(start_date: str, end_date: str):
    df_list = []
    for year in get_year_range(start_date, end_date):
        year_df = calculate_daily_means_for_year(year)
        if not year_df.empty:
            df_list.append(year_df)
    
    if not df_list:
        print("No data to process")
        return
    
    # Process data in chunks to manage memory
    chunk_size = max(1, len(df_list) // N_CORES)
    full_df_chunks = []
    
    for i in range(0, len(df_list), chunk_size):
        chunk_df = pd.concat(df_list[i:i+chunk_size], ignore_index=True)
        chunk_df['date_dt'] = pd.to_datetime(chunk_df['DATE'], format='%Y%m%d')
        chunk_df['year'] = chunk_df['date_dt'].dt.year
        chunk_df['month'] = chunk_df['date_dt'].dt.month
        chunk_df['day'] = chunk_df['date_dt'].dt.day
        full_df_chunks.append(chunk_df)
    
    full_df = pd.concat(full_df_chunks, ignore_index=True)
    
    # Process fortnights in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=N_CORES) as executor:
        chunks = np.array_split(full_df, N_CORES)
        fortnight_results = list(executor.map(parallel_process_fortnight, chunks))
    
    fortnight_info = pd.concat(fortnight_results, ignore_index=True)
    full_df = pd.concat([full_df, fortnight_info], axis=1)
    
    # Aggregate using DuckDB for better performance
    temp_parquet = CACHE_DIR / "temp_full.parquet"
    full_df.to_parquet(temp_parquet)
    
    agg_query = f"""
    SELECT 
        code as range_code,
        AVG(INDEX_mean) as value,
        MIN(start_date) as start_date,
        MAX(end_date) as end_date,
        label
    FROM '{temp_parquet}'
    GROUP BY code, year, month, fn_label, label
    """
    
    final_df = con.execute(agg_query).fetchdf()
    final_df.to_csv(OUTPUT_CSV, index=False)
    
    # Generate SQL insert statement
    generate_sql_insert(final_df, 'groundwater_trends_insert.sql')
    
    # Update database
    con.execute(f"""
    CREATE TABLE IF NOT EXISTS {FINAL_TABLE} (
        range_code VARCHAR,
        value DOUBLE,
        start_date TIMESTAMP,
        end_date TIMESTAMP,
        label VARCHAR
    )
    """)
    con.execute(f"DELETE FROM {FINAL_TABLE}")
    con.execute(f"COPY {FINAL_TABLE} FROM '{OUTPUT_CSV}' (AUTO_DETECT TRUE)")
    con.commit()
    
    # Cleanup
    if temp_parquet.exists():
        temp_parquet.unlink()

def generate_sql_insert(df: pd.DataFrame, output_file: str = 'groundwater_trends_insert.sql'):
    """Generate SQL insert statement for groundwater trends data."""
    
    # Create SQL template
    sql_template = """-- Copy groundwater trends data to the database
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
        WHERE dp.name = 'Groundwater trend'
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
    codes = "'" + "', '".join(df['range_code'].unique()) + "'"
    values_list = []

    # Process each unique combination of range_code and label
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing records"):
        values_list.append(
            f"        ('{row['range_code']}', '{row['label']}', {row['value']:.2f})"
        )

    values = ",\n".join(values_list)
    sql = sql_template.format(codes=codes, values=values)

    # Write to file
    with open(output_file, 'w') as f:
        f.write(sql)

    print(f"SQL file generated successfully at {output_file}!")


if __name__ == "__main__":
    # Example execution
    start_date = "20130101"
    end_date = "20241130"
    
    # Set pandas to use all cores for operations
    pd.set_option('compute.use_numexpr', True)
    
    # download_data(start_date, end_date)
    aggregate_fortnightly(start_date, end_date)
    print("All done!")