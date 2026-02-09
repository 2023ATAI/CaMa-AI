import os
import xarray as xr
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree


nc_folder = 'E:\LZH_py_fortran_comprision\python'      
excel_path = 'D:\camels\cattributes_other_camels.csv'   

nc_files = [os.path.join(nc_folder, f) for f in os.listdir(nc_folder) if f.endswith('.nc4')]
df_station = pd.read_excel(excel_path)
station_coords = df_station[['gauge_lon', 'gauge_lat']].values


lats = ds_sim['lat'].values
lons = ds_sim['lon'].values

lon2d, lat2d = np.meshgrid(lons, lats)
grid_points = np.column_stack([lon2d.ravel(), lat2d.ravel()])
tree = cKDTree(grid_points)

#read obs
df_obs = pd.read_excel(excel_file)

obs_points = df_obs[['lon', 'lat']].values
_, indices = tree.query(obs_points)


df_obs['grid_index'] = indices
df_obs['grid_lon'] = grid_points[indices][:, 0]
df_obs['grid_lat'] = grid_points[indices][:, 1]


df_obs['time'] = pd.to_datetime(df_obs['time'])  
grouped = df_obs.groupby(['time', 'grid_lat', 'grid_lon'])['runoff'].median().reset_index()


obs_grid = np.full(ds_sim['runoff'].shape, np.nan)  

time_index = {pd.to_datetime(str(t.values)): i for i, t in enumerate(ds_sim['time'])}

for _, row in grouped.iterrows():
    try:
        t_idx = time_index[pd.to_datetime(row['time'])]
        lat_idx = np.argmin(np.abs(ds_sim['lat'].values - row['grid_lat']))
        lon_idx = np.argmin(np.abs(ds_sim['lon'].values - row['grid_lon']))
        obs_grid[t_idx, lat_idx, lon_idx] = row['runoff']
    except KeyError:
        continue

da_obs = xr.DataArray(
    data=obs_grid,
    coords=dict(
        time=ds_sim['time'],
        lat=ds_sim['lat'],
        lon=ds_sim['lon']
    ),
    dims=["time", "lat", "lon"],
    name='obs_runoff'
)

da_obs.to_netcdf("processed_obs_runoff.nc")

ds_combined = ds_sim.copy()
ds_combined['obs_runoff'] = da_obs
ds_combined.to_netcdf("combined_runoff_data.nc")
