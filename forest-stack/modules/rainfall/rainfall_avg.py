import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import requests
import duckdb
from calendar import month_abbr
from tqdm import tqdm
from requests.exceptions import Timeout, RequestException
import os
import logging
import json
from pathlib import Path
import xarray as xr
import polars as pl  # Import Polars
import concurrent.futures

class IMDDataProcessor:
    def __init__(self, cache_dir="./data/IMD", log_dir="logs"):
        """Initialize the IMD data processor"""
        self.cache_dir = self._setup_dir(cache_dir)
        self.log_dir = self._setup_dir(log_dir)
        self._setup_logging()
        self.grid_coords = self._setup_grid_coordinates()
    
    def _setup_dir(self, dir_path):
        """Create directory if it doesn't exist"""
        os.makedirs(dir_path, exist_ok=True)
        return dir_path
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(self.log_dir, f"imd_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _setup_grid_coordinates(self):
        """Setup grid coordinates for rainfall data"""
        return {
            'rainfall': {
                'lon': np.linspace(66.5, 100, num=135),
                'lat': np.linspace(6.5, 38.5, num=129),
                'shape': (129, 135)
            }
        }
    
    def process_rainfall_data(self, data, date_val):
        """Process rainfall grid data"""
        data = data.reshape(1, *self.grid_coords['rainfall']['shape'])
        data[data < 0] = np.nan
        
        return xr.Dataset(
            {"prcp": xr.DataArray(
                data=data,
                dims=["time", "lat", "lon"],
                coords={
                    "lat": xr.IndexVariable("lat", self.grid_coords['rainfall']['lat'],
                                            attrs=dict(units='degrees_north', long_name='Latitude')),
                    "lon": xr.IndexVariable("lon", self.grid_coords['rainfall']['lon'],
                                            attrs=dict(units='degrees_east', long_name='Longitude')),
                    "time": [date_val]
                },
                attrs=dict(
                    units='mm',
                    standard_name='precipitation',
                    long_name='precipitation',
                    description='IMD 25X25 Daily Gridded Data'
                )
            )}
        )

    def download_daily_data(self, date_val, timeout=15):
        """Download daily rainfall data from IMD website or load from .nc file"""

        # 1. Check if data exists in the .nc files
        nc_folder = "./data/IMD"  # Path to your .nc folder
        year = date_val.year
        nc_file_pattern = f"RF25_ind{year}*.nc"
        nc_files = list(Path(nc_folder).glob(nc_file_pattern))

        if nc_files:
            try:
                print(f"Loading rainfall data for {year} from .nc file...\r", end="", flush=True)
                ds = xr.open_mfdataset(nc_files, combine='by_coords')['RAINFALL']
                
                # Extract data for the specific date
                ds_date = ds.sel(TIME=date_val.strftime('%Y-%m-%d'))

                # Convert to numpy array and reshape
                data = ds_date.values.astype(np.float32)
                data = data.reshape(-1)  # Flatten the array

                logging.info(f"Loaded rainfall data for {date_val.strftime('%Y-%m-%d')} from .nc file \r")
                return self.process_rainfall_data(data, date_val)

            except Exception as e:
                logging.warning(f"Error loading from .nc file for {date_val.strftime('%Y-%m-%d')}: {e} \r")

        # 2. If not found in .nc, proceed with the download
        base_url = "https://www.imdpune.gov.in/cmpg/Realtimedata/Rainfall"
        filename = f"rain_ind0.25_{date_val.strftime('%y_%m_%d')}.grd"
        
        cache_file = os.path.join(self.cache_dir, filename)
        
        # Check cache first
        if os.path.exists(cache_file):
            try:
                data = np.fromfile(cache_file, dtype=np.float32)
                logging.info(f"Loaded {filename} from cache")
                return self.process_rainfall_data(data, date_val)
            except Exception as e:
                logging.warning(f"Error loading cached file {filename}: {str(e)}")
                os.remove(cache_file)
        
        # Download if not in cache
        try:
            url = f"{base_url}/{filename}"
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            
            with open(cache_file, 'wb') as f:
                f.write(response.content)
            
            data = np.fromfile(cache_file, dtype=np.float32)
            return self.process_rainfall_data(data, date_val)
            
        except Exception as e:
            logging.error(f"Error downloading {filename}: {str(e)}")
            return None

    def convert_to_duckdb_format(self, dataset):
        """Convert xarray dataset to format suitable for DuckDB using Polars"""
        
        # Convert to Polars DataFrame
        df = dataset['prcp'].to_dataframe().reset_index()
        df = pl.from_pandas(df)  # Convert to Polars

        # Ensure proper types (Polars usually infers these correctly)
        df = df.with_columns([
            pl.col("lat").cast(pl.Float64),
            pl.col("lon").cast(pl.Float64),
            pl.col("prcp").cast(pl.Float64).alias("rainfall")  # Rename to rainfall
        ])

        # Add geometry and fortnight columns
        df = df.with_columns([
            pl.format('{}-{}', pl.col("time").dt.strftime('%Y-%m'), 
                      pl.when(pl.col("time").dt.day() <= 15).then(1).otherwise(2)).alias("fortnight")
        ])

        return df.to_pandas()  # Convert back to Pandas for DuckDB

    def create_duckdb_tables(self, conn, dataset):
        """Create and populate DuckDB tables using Parquet"""
        try:
            print("Converting rainfall data to DataFrame...")
            df = self.convert_to_duckdb_format(dataset)

            print("Saving DataFrame to Parquet...")
            parquet_file = "rainfall_data.parquet"
            df.to_parquet(parquet_file)  # Save DataFrame to Parquet

            print("Creating rainfall table from Parquet...")
            # Create the table and load data from Parquet
            conn.execute(f"""
                DROP TABLE IF EXISTS rainfall_data;
                CREATE TABLE rainfall_data AS 
                SELECT * FROM '{parquet_file}';  -- Load from Parquet
            """)

            print("Creating final rainfall table...")
            conn.execute("""
                DROP TABLE IF EXISTS rainfall_data_agg;

                CREATE TABLE rainfall_data_agg AS
                (WITH spatial_join AS (
                    SELECT
                        rainfall as value,
                        fortnight,
                        time as date,
                        s.code as range_code
                    FROM rainfall_data r
                    JOIN regions2 s
                    ON ST_Contains(s.geom::GEOMETRY, ST_Point(lon, lat))  -- Use lon and lat directly
                    WHERE s.type = 'range'
                ),
                aggregated_data AS (
                    SELECT
                        range_code,
                        fortnight,
                        date,
                        AVG(value) as avg_value,
                        COUNT(*) as point_count
                    FROM spatial_join
                    GROUP BY range_code, fortnight, date
                )
                SELECT
                    range_code,
                    avg_value as value,
                    date as measurement_date,
                    CAST(
                        DATE_TRUNC('month', date) +
                        CASE WHEN RIGHT(fortnight, 1) = '1'
                            THEN INTERVAL '0 days'
                            ELSE INTERVAL '15 days'
                        END AS TIMESTAMP
                    ) as start_date,
                    CAST(
                        CASE WHEN RIGHT(fortnight, 1) = '1'
                            THEN DATE_TRUNC('month', date) + INTERVAL '14 days'
                            ELSE DATE_TRUNC('month', date) + INTERVAL '1 month' - INTERVAL '1 day'
                        END AS TIMESTAMP
                    ) as end_date,
                    'average_rainfall_' ||
                    EXTRACT(year FROM date) || '_' ||
                    CASE
                        WHEN EXTRACT(month FROM date) = 1 THEN 'jan'
                        WHEN EXTRACT(month FROM date) = 2 THEN 'feb'
                        WHEN EXTRACT(month FROM date) = 3 THEN 'mar'
                        WHEN EXTRACT(month FROM date) = 4 THEN 'apr'
                        WHEN EXTRACT(month FROM date) = 5 THEN 'may'
                        WHEN EXTRACT(month FROM date) = 6 THEN 'jun'
                        WHEN EXTRACT(month FROM date) = 7 THEN 'jul'
                        WHEN EXTRACT(month FROM date) = 8 THEN 'aug'
                        WHEN EXTRACT(month FROM date) = 9 THEN 'sep'
                        WHEN EXTRACT(month FROM date) = 10 THEN 'oct'
                        WHEN EXTRACT(month FROM date) = 11 THEN 'nov'
                        WHEN EXTRACT(month FROM date) = 12 THEN 'dec'
                    END || '_fn' ||
                    RIGHT(fortnight, 1) as label,
                    point_count
                FROM aggregated_data
                ORDER BY range_code, date);
            """)

            print("Creating indexes for rainfall table...")
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rainfall_code ON rainfall_data_agg(range_code);
                CREATE INDEX IF NOT EXISTS idx_rainfall_date ON rainfall_data_agg(measurement_date);
                CREATE INDEX IF NOT EXISTS idx_rainfall_start ON rainfall_data_agg(start_date);
                CREATE INDEX IF NOT EXISTS idx_rainfall_end ON rainfall_data_agg(end_date);
            """)

            count = conn.execute("SELECT COUNT(*) FROM rainfall_data_agg").fetchone()[0]
            print(f"Created rainfall table with {count} records")
            return count

        except Exception as e:
            print(f"Error creating rainfall table: {str(e)}")
            logging.error(f"Error creating rainfall table: {str(e)}")
            raise

def process_all_data(db_path, start_date, end_date):
    """Process rainfall data and save to DuckDB"""
    processor = IMDDataProcessor()

    print("\nInitializing processing...")
    conn = duckdb.connect(db_path)

    try:
        print("Loading spatial extension...")
        conn.execute("INSTALL spatial;")
        conn.execute("LOAD spatial;")
        
        print("\nProcessing RAINFALL data:")
        date_range = pd.date_range(start_date, end_date)

        # Use ThreadPoolExecutor for parallel downloads
        with concurrent.futures.ThreadPoolExecutor() as executor:
            datasets = list(tqdm(executor.map(processor.download_daily_data, date_range), 
                                 total=len(date_range), desc="Downloading rainfall data"))

        # Filter out any None values (failed downloads)
        datasets = [ds for ds in datasets if ds is not None]

        if datasets:
            print(f"Combining {len(datasets)} rainfall datasets...")
            combined_ds = xr.concat(datasets, dim="time")
            records = processor.create_duckdb_tables(conn, combined_ds)
            print(f"Created rainfall table with {records} records")
            return True

        return False

    except Exception as e:
        print(f"\nError processing data: {str(e)}")
        logging.error(f"Error processing data: {str(e)}")
        return False

    finally:
        print("\nClosing database connection...")
        conn.close()

def validate_data_quality(conn):
    """Validate rainfall data quality and log issues"""
    logging.info("Validating data quality...")
    
    try:
        # Check for missing values
        missing_check = conn.execute("""
            SELECT 
                range_code,
                COUNT(*) as total_records,
                SUM(CASE WHEN value IS NULL THEN 1 ELSE 0 END) as missing_rainfall
            FROM rainfall_data_agg
            GROUP BY range_code
            HAVING SUM(CASE WHEN value IS NULL THEN 1 ELSE 0 END) > 0
        """).fetchdf()
        
        if not missing_check.empty:
            logging.warning("Found missing values in the following ranges:")
            logging.warning(missing_check.to_string())
        
        # Check for anomalous values
        anomaly_check = conn.execute("""
            SELECT 
                range_code,
                measurement_date,
                value as rainfall
            FROM rainfall_data_agg
            WHERE value < 0 OR value > 1000  -- Suspicious rainfall values
        """).fetchdf()
        
        if not anomaly_check.empty:
            logging.warning("Found anomalous values:")
            logging.warning(anomaly_check.to_string())
        
        return True
        
    except Exception as e:
        logging.error(f"Error during data validation: {str(e)}")
        return False

def generate_sql_insert_file(conn, output_file="rainfall_insert.sql"):
    """Generate SQL insert statements for the rainfall data"""
    try:
        print(f"\nGenerating SQL insert file: {output_file}")

        # Get all the data from the rainfall_data table
        # No change here, your rounding and aggregation seem fine
        rainfall_data = conn.execute("""
            SELECT 
                range_code,
                label,
                round(sum(value),2) as value
            FROM rainfall_data_agg 
            GROUP BY range_code, label 
            ORDER BY range_code, label
        """).fetchdf()

        # Create the SQL file
        with open(output_file, 'w') as f:
            # Write the CTE setup
            f.write("""-- Copy rainfall data to the database
WITH region_codes AS (
    -- Get region IDs from the regions table based on codes in the data
    SELECT DISTINCT 
        id as region_id,
        code
    FROM public.regions 
    WHERE code IN (
""")

            # Write unique range codes
            # Using a set comprehension for a more concise way to get unique codes
            unique_codes = sorted({row['range_code'] for _, row in rainfall_data.iterrows()})  
            f.write(", ".join([f"'{code}'" for code in unique_codes]))  # Using join for simpler formatting

            f.write("""
    )
    AND deleted_at IS NULL  -- Removed extra indentation here
),
dataset_snapshots AS (  -- Removed extra indentation here
    -- Get snapshot IDs based on labels and data product ID
    SELECT 
        dps.id as snapshot_id,
        dps.label
    FROM public.data_product_snapshots dps
    JOIN public.data_products dp ON dp.id = dps.data_product_id
    WHERE dp.name = 'Average rainfall'
    AND dps.deleted_at IS NULL
    AND dp.deleted_at IS NULL
),
raw_data(region_code, snapshot_label, value) AS (
    VALUES
""")
            for i, row in rainfall_data.iterrows():
                comma = "," if i < len(rainfall_data) - 1 else ""
                f.write(f"        ('{row['range_code']}', '{row['label']}', {row['value']}){comma}\n")

            f.write(""")
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
""")

        print(f"Successfully generated SQL insert file: {output_file}")
        return True

    except Exception as e:
        print(f"Error generating SQL insert file: {str(e)}")
        logging.error(f"Error generating SQL insert file: {str(e)}")
        return False

# Update the main function to include the SQL file generation
def main():
    """Main execution function"""
    # Configuration
    start_date = datetime(2013, 1, 1)
    end_date = datetime(2024, 11, 30)
    db_path = 'dfhms.db'  # Corrected db_path

    logging.info("Starting IMD rainfall data processing")
    logging.info(f"Processing period: {start_date} to {end_date}")
    logging.info(f"Using database: {db_path}")

    # Process data
    success = process_all_data(db_path, start_date, end_date)

    if success:
        logging.info("Data processing completed successfully")

        # Connect to database for validation and SQL generation
        conn = duckdb.connect(db_path)
        try:
            # Validate data
            if validate_data_quality(conn):
                logging.info("Data validation completed")

                # Show sample of processed data
                sample = conn.execute("""
                    SELECT * FROM rainfall_data_agg 
                    LIMIT 5
                """).fetchdf()
                logging.info("\nSample of processed data:")
                logging.info("\n" + str(sample))

                # Show summary statistics
                summary = conn.execute("""
                    SELECT 
                        COUNT(DISTINCT range_code) as num_ranges,
                        COUNT(*) as total_records,
                        MIN(measurement_date) as start_date,
                        MAX(measurement_date) as end_date
                    FROM rainfall_data_agg
                """).fetchdf()
                logging.info("\nSummary statistics:")
                logging.info("\n" + str(summary))

                # Generate SQL insert file
                if generate_sql_insert_file(conn):
                    logging.info("SQL insert file generated successfully")
                else:
                    logging.error("Failed to generate SQL insert file")
            else:
                logging.error("Data validation failed")
        finally:
            conn.close()
    else:
        logging.error("Data processing failed")

if __name__ == "__main__":
    main()