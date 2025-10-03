import os
import xarray as xr
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree


# ---------------------
# 路径配置
# ---------------------
nc_folder = 'E:\LZH_py_fortran_comprision\python'         # 模拟结果的 NetCDF 文件夹
excel_path = 'D:\camels\cattributes_other_camels.csv'      # 包含站点观测数据的 Excel 文件

# ---------------------
# 读取所有 NetCDF 文件并合并
# ---------------------
nc_files = [os.path.join(nc_folder, f) for f in os.listdir(nc_folder) if f.endswith('.nc4')]
df_station = pd.read_excel(excel_path)
station_coords = df_station[['gauge_lon', 'gauge_lat']].values

# 提取模拟网格的经纬度
lats = ds_sim['lat'].values
lons = ds_sim['lon'].values

lon2d, lat2d = np.meshgrid(lons, lats)
grid_points = np.column_stack([lon2d.ravel(), lat2d.ravel()])
tree = cKDTree(grid_points)

# ---------------------
# 读取观测数据
# ---------------------
df_obs = pd.read_excel(excel_file)

# ---------------------
# 将观测站点匹配到最近网格点
# ---------------------
obs_points = df_obs[['lon', 'lat']].values
_, indices = tree.query(obs_points)

# 将观测点匹配的网格点索引映射到 lat-lon 坐标
df_obs['grid_index'] = indices
df_obs['grid_lon'] = grid_points[indices][:, 0]
df_obs['grid_lat'] = grid_points[indices][:, 1]

# ---------------------
# 根据时间和格点，取中位数进行聚合
# ---------------------
df_obs['time'] = pd.to_datetime(df_obs['time'])  # 确保时间格式统一
grouped = df_obs.groupby(['time', 'grid_lat', 'grid_lon'])['runoff'].median().reset_index()

# ---------------------
# 转换为 xarray DataArray（与模拟结果对齐）
# ---------------------
# 创建空数组填充 median runoff
obs_grid = np.full(ds_sim['runoff'].shape, np.nan)  # 假设 runoff 是 (time, lat, lon)

# 建立映射
time_index = {pd.to_datetime(str(t.values)): i for i, t in enumerate(ds_sim['time'])}

for _, row in grouped.iterrows():
    try:
        t_idx = time_index[pd.to_datetime(row['time'])]
        lat_idx = np.argmin(np.abs(ds_sim['lat'].values - row['grid_lat']))
        lon_idx = np.argmin(np.abs(ds_sim['lon'].values - row['grid_lon']))
        obs_grid[t_idx, lat_idx, lon_idx] = row['runoff']
    except KeyError:
        # 观测时间不在模拟结果中
        continue

# 转换为 DataArray
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

# ---------------------
# 保存或与模拟结果合并
# ---------------------
# 例如保存为 NetCDF
da_obs.to_netcdf("processed_obs_runoff.nc")

# 或与模拟数据合并进行对比
ds_combined = ds_sim.copy()
ds_combined['obs_runoff'] = da_obs
ds_combined.to_netcdf("combined_runoff_data.nc")