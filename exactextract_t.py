# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:42:16 2024

@author: pmoghaddasi
"""


from exactextract import exact_extract
import xarray as xr
import pandas as pd
import geopandas as gpd
from pyproj import CRS
import rioxarray
import time

# Load the raster dataset from the NetCDF file
nc_path = "C:\\Users\\pmoghaddasi\\Desktop\\Snow\\test\\daymet_v4_daily_na_swe_2020.nc"
ds  = xr.open_dataset(nc_path)

rast = ds['swe'][:3]
time_labels = ds.time[:3].dt.strftime('%Y-%m-%d').values  # Extracting timestamps

s = time.time()

# =============================================================================
# shapefile = "D:\\Fatemeh\\camels-20240418T1048Z\\basin_set_full_res\\HCDN_nhru_final_671.shp"
# 
# =============================================================================
output_shapefile = "C:\\Users\\pmoghaddasi\\Desktop\\Snow\\test\\HCDN_nhru_final_671_daymet.shp"
# =============================================================================
# 
# gdf = gpd.read_file(shapefile)
# 
# daymet_crs = CRS.from_proj4(
#     "+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 "
#     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
# )
# 
# 
# gdf_daymet = gdf.to_crs(daymet_crs)
# 
# shapefile = gdf_daymet.to_file(output_shapefile)
# 
# =============================================================================

gdf = gpd.read_file(output_shapefile)

# Perform the exact extraction
stats = exact_extract(rast, output_shapefile, ['mean'])

# Convert results to a DataFrame
stats_df = pd.DataFrame(stats)

# Flatten the properties dictionaries into separate columns
properties_df = pd.json_normalize(stats_df['properties'])

# Rename the columns using extracted time labels
properties_df.columns = [f"{time_label}" for time_label in time_labels]

# Adding 'hru_id' from the GeoDataFrame as the index of properties_df
properties_df['hru_id'] = gdf['hru_id']
properties_df.set_index('hru_id', inplace=True)

t = time.time()

print(t-s)
print(properties_df)
