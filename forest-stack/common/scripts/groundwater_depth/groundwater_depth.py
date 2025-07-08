import zipfile
from datetime import datetime
import requests
import geopandas as gpd
import duckdb
import pandas as pd
import os
import subprocess
import json


groundwater_data_path ='data/groundwater_depth/Atal_Jal_Disclosed_Ground_Water_Level-2015-2022-utf8.csv'
subdistricts_path = 'data/gis/ranges-withcode-geom.shp'
# Parameters
db_path = "dfhms.db"
table_name = "regions"
output_table = "canopy_density_data"

# Load subdistricts GeoJSON using DuckDB directly
con = duckdb.connect(db_path, read_only=False)
con.execute("INSTALL spatial;")
con.execute("LOAD spatial;")

# Load the groundwater CSV and subdistricts GeoJSON into DuckDB
# con.execute(f"""
#     CREATE TABLE groundwater AS 
#         SELECT
#             sr_no,state_name_with_lgd_code,district_name_with_lgd_code,block_name_with_lgd_code,gp_name_with_lgd_code,
#             village,site_name,type,source,well_id,latitude,longitude,well_depth_meters,aquifer,
#             CAST(REGEXP_EXTRACT(column_name, '\d{{4}}') AS INTEGER) as measurement_year,
#             CASE
#                 WHEN column_name ILIKE 'premonsoon_%' THEN 'premonsoon'
#                 WHEN column_name ILIKE 'postmonsoon_%' THEN 'postmonsoon'
#             END AS monsoon_period,
#             CASE
#                 WHEN ground_level ILIKE 'dry' THEN 0
#                 WHEN ground_level ILIKE 'na' THEN NULL
#                 ELSE try_cast(ground_level AS FLOAT)
#             END AS groundwater_depth,
#             ST_Point(longitude, latitude) as geom 
#         FROM
#             (
#                 SELECT * FROM
#                     read_csv (
#                         'Atal_Jal_Disclosed_Ground_Water_Level-2015-2022-utf8.csv',
#                         header = True,
#                         normalize_names = True
#                     )
#             ) UNPIVOT (
#                 ground_level FOR column_name IN (
#                     "premonsoon_2015_meters_below_ground_level",
#                     "postmonsoon_2015_meters_below_ground_level",
#                     "premonsoon_2016_meters_below_ground_level",
#                     "postmonsoon_2016_meters_below_ground_level",
#                     "premonsoon_2017_meters_below_ground_level",
#                     "postmonsoon_2017_meters_below_ground_level",
#                     "premonsoon_2018_meters_below_ground_level",
#                     "postmonsoon_2018_meters_below_ground_level",
#                     "premonsoon_2019_meters_below_ground_level",
#                     "postmonsoon_2019_meters_below_ground_level",
#                     "premonsoon_2020_meters_below_ground_level",
#                     "postmonsoon_2020_meters_below_ground_level",
#                     "premonsoon_2021_meters_below_ground_level",
#                     "postmonsoon_2021_meters_below_ground_level",
#                     "premonsoon_2022_meters_below_ground_level",
#                     "postmonsoon_2022_meters_below_ground_level"
#                 )
#             )
#         WHERE state_name_with_lgd_code ILIKE 'rajasthan%'
#         ;
# """)

# con.execute(f"""
#     CREATE TABLE subdistricts AS 
#     SELECT * FROM 'ranges-withcode-geom.shp';
# """)

periods = [
    (2015, 'premonsoon'),
    (2015, 'postmonsoon'),
    (2016, 'premonsoon'),
    (2016, 'postmonsoon'),
    (2017, 'premonsoon'),
    (2017, 'postmonsoon'),
    (2018, 'premonsoon'),
    (2018, 'postmonsoon'),
    (2019, 'premonsoon'),
    (2019, 'postmonsoon'),
    (2020, 'premonsoon'),
    (2020, 'postmonsoon'),
    (2021, 'premonsoon'),
    (2021, 'postmonsoon'),
    (2022, 'premonsoon'),
    (2022, 'postmonsoon')
]

output_paths = []

for measurement_year, period in periods:
    # Aggregate data at subdistrict level
    query = f"""
        CREATE TABLE yearly_period_agg_{measurement_year}_{period} AS 
        SELECT name, {measurement_year} as year, '{period}' as period,  AVG(groundwater_depth) AS AverageGroundwaterDepth , any_value(geom) as geom
        FROM 
            (
            SELECT s.*, g.* 
            FROM groundwater AS g JOIN {table_name} AS s ON ST_Within(g.geom, s.geom) 
            WHERE 
                s.type = 'range' AND
                g.groundwater_depth is NOT NULL AND 
                g.measurement_year = {measurement_year} AND 
                g.monsoon_period = '{period}'
            )
        GROUP BY name;
    """
    con.execute(query)

    # Export the aggregated data to geojson
    output_path = f'./groundwater_aggregated_{measurement_year}_{period}.geojson'
    output_path2 = f'./groundwater_aggregated_{measurement_year}_{period}.csv'
    con.execute(f"COPY yearly_period_agg_{measurement_year}_{period} TO '{output_path}' WITH (FORMAT GDAL, DRIVER 'GeoJSON', LAYER_CREATION_OPTIONS 'WRITE_BBOX=YES');")
    con.execute(f"COPY yearly_period_agg_{measurement_year}_{period} TO '{output_path2}';")
    output_paths.append(output_path)


