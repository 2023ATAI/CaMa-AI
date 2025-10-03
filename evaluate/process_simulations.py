import matplotlib
import numpy as np
import pandas as pd
import yaml
import argparse
import xarray as xr
import os, sys
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from matplotlib import colors, cm
from mpl_toolkits.basemap import Basemap

from evluation_utils import KGESS, RMSE, BIAS, CORRELATION
def process_simulations(config, OUTPUTFOLDER):
    SIMDIRs = config['SIMDIR']
    print(f'a total of simulation results is {len(SIMDIRs)}')
    for SIMDIR in SIMDIRs:
        print(f'processing simulation dir : {SIMDIR}')
        process_single_simulation(config, SIMDIR, OUTPUTFOLDER)
    return None

def process_station(station, REFDIR, TIMESCALE, STARTYEAR, ENDYEAR, min_lon, max_lon, min_lat, max_lat):
    result = {
        'Flag': False,
        'use_syear': -9999,
        'use_eyear': -9999,
        'obs_syear': -9999,
        'obs_eyear': -9999,
        'ref_dir': 'file'
    }
    file_path = f'{REFDIR}\\GRDC_{TIMESCALE}\\{int(station["ID"])}_Q_{TIMESCALE}.Cmd.nc'
    if os.path.exists(file_path):
        result['ref_dir'] = file_path
        with xr.open_dataset(file_path) as df:
            result['obs_syear'] = int(df["time.year"].values[0])
            result['obs_eyear'] = int(df["time.year"].values[-1])
            result['use_syear'] = max(result['obs_syear'], STARTYEAR)
            result['use_eyear'] = min(result['obs_eyear'], ENDYEAR)
            if ((result['use_eyear'] - result['use_syear'] >= 1) and
                (station['lon'] >= min_lon) and
                (station['lon'] <= max_lon) and
                (station['lat'] >= min_lat) and
                (station['lat'] <= max_lat) and
                (station['ix2'] == -9999)):
                result['Flag'] = True
    return result

def process_single_simulation(config, SIMDIR, OUTPUTFOLDER):
    STARTYEAR = config['STARTYEAR']
    ENDYEAR = config['ENDYEAR']
    SPATIALRESOLUTION = config['SPATIALRESOLUTION']
    TIMESCALE = config['TIMESCALE']
    REFDIR = config['REFDIR']
    min_lon = config['MIN_LON']
    max_lon = config['MAX_LON']
    min_lat = config['MIN_LAT']
    max_lat = config['MAX_LAT']
    # --------------------------- get ref location list -----------------------------------------
    if not os.path.exists(f"{OUTPUTFOLDER}\\stn_list.txt"):
        file_path = f'{REFDIR}\\list\\GRDC_alloc_{SPATIALRESOLUTION}Deg.txt'
        station_list = pd.read_csv(file_path, delimiter=r"\s+", header=0)
        print(file_path)

        results = Parallel(n_jobs=-1)(
            delayed(process_station)(row, REFDIR, TIMESCALE, STARTYEAR, ENDYEAR, min_lon, max_lon, min_lat, max_lat) for _, row in station_list.iterrows())
        for i, result in enumerate(results):
            for key, value in result.items():
                station_list.at[i, key] = value

        ind = station_list[station_list['Flag'] == True].index
        data_select = station_list.loc[ind]

        if SPATIALRESOLUTION == 0.25:
            lat0 = np.arange(89.875, -90, -0.25)
            lon0 = np.arange(-179.875, 180, 0.25)
        elif SPATIALRESOLUTION == 0.0167:  # 01min
            lat0 = np.arange(89.9916666666666600, -90, -0.0166666666666667)
            lon0 = np.arange(-179.9916666666666742, 180, 0.0166666666666667)
        elif SPATIALRESOLUTION == 0.0833:  # 05min
            lat0 = np.arange(89.9583333333333286, -90, -0.0833333333333333)
            lon0 = np.arange(-179.9583333333333428, 180, 0.0833333333333333)
        elif SPATIALRESOLUTION == 0.1:  # 06min
            lat0 = np.arange(89.95, -90, -0.1)
            lon0 = np.arange(-179.95, 180, 0.1)
        elif SPATIALRESOLUTION == 0.05:  # 03min
            lat0 = np.arange(89.975, -90, -0.05)
            lon0 = np.arange(-179.975, 180, 0.05)
        data_select['lon_cama'] = -9999.0
        data_select['lat_cama'] = -9999.0
        print(data_select.dtypes)
        for iii in range(len(data_select['ID'])):
            data_select['lon_cama'].values[iii] = np.float64(lon0[int(data_select['ix1'].values[iii]) - 1])
            data_select['lat_cama'].values[iii] = np.float64(lat0[int(data_select['iy1'].values[iii]) - 1])
            if abs(data_select['lat_cama'].values[iii] - data_select['lat'].values[iii]) > 1:
                print(f"Warning: ID {data_select['ID'][iii]} lat is not match")
            if abs(data_select['lon_cama'].values[iii] - data_select['lon'].values[iii]) > 1:
                print(f"Warning: ID {data_select['ID'][iii]} lon is not match")
        data_select['ref_lon'] = data_select['lon_cama']
        data_select['ref_lat'] = data_select['lat_cama']
        data_select['use_syear'] = data_select['use_syear'].astype(int)
        data_select['use_eyear'] = data_select['use_eyear'].astype(int)
        data_select['obs_syear'] = data_select['obs_syear'].astype(int)
        data_select['obs_eyear'] = data_select['obs_eyear'].astype(int)
        data_select['ID'] = data_select['ID'].astype(int)

        data_select.to_csv(f"{OUTPUTFOLDER}\\stn_list.txt", index=False)

    stn_list = pd.read_csv(f"{OUTPUTFOLDER}\\stn_list.txt", delimiter=r",", header=0)

    # --------------------------- process sim discharge data to station data-----------------------------------------
    years = list(range(STARTYEAR, ENDYEAR + 1))
    filenames = [f'{SIMDIR}o_outflw{year}.nc' for year in years]
    discharg_sim_dataset = xr.open_mfdataset(filenames, combine='nested', concat_dim='time')

    model_name = os.path.basename(os.path.normpath(SIMDIR))
    OUTPUTFOLDER1 = OUTPUTFOLDER + f"{model_name}\\"
    if not os.path.exists(OUTPUTFOLDER1):
        os.mkdir(OUTPUTFOLDER1)
    for row in stn_list.itertuples(index=False):
        station_ID = row.ID
        lon_cama = row.lon_cama
        lat_cama = row.lat_cama
        syear = row.use_syear
        eyear = row.use_eyear
        print(f"{station_ID}: ({lon_cama}, {lat_cama}), {syear}–{eyear}")
        da_raw = discharg_sim_dataset['outflw'].sel({'lon': lon_cama, 'lat': lat_cama}, method='nearest')

        station_OUTPUTFOLDER = OUTPUTFOLDER1 + "station\\"
        if not os.path.exists(station_OUTPUTFOLDER):
            os.mkdir(station_OUTPUTFOLDER)
        out_file = os.path.join(station_OUTPUTFOLDER, f"{station_ID}.nc")
        if not os.path.exists(out_file):
            start_day = f'{syear}-01-01'
            end_day = f'{eyear}-12-31'
            discharge_sim_station = da_raw.sel(time=slice(start_day, end_day))

            discharge_sim_station.to_netcdf(
                out_file,
                encoding={'outflw': {'dtype': 'float32', 'zlib': True, 'complevel': 4}}
            )

    # --------------------------- Calculate metrics all station-----------------------------------------
    metrics = stn_list.copy()
    for idx, row in metrics.iterrows():
        station_ID = row.ID
        print(f"{station_ID}")
        ref_path = f'{REFDIR}GRDC_{TIMESCALE}\\{station_ID}_Q_Day.Cmd.nc'
        sim_path = f'{OUTPUTFOLDER}\\{model_name}\\station\\{station_ID}.nc'
        with xr.open_dataset(sim_path) as df:
            sim_data = df['outflw'].values * 86400
            start_date = pd.to_datetime(df.time.min().values)
            end_date = pd.to_datetime(df.time.max().values)
        full_time = pd.date_range(start_date, end_date, freq='D')
        with xr.open_dataset(ref_path) as df1:
            ds = df1.reindex(time=full_time).ffill('time')
            ref_data = ds['discharge'].sel(time=slice(start_date, end_date)).squeeze(drop=True).values

        kgess = KGESS(sim_data, ref_data)
        correlation = CORRELATION(sim_data, ref_data)
        rmse = RMSE(sim_data, ref_data)
        bias = BIAS(sim_data, ref_data)
        metrics.loc[idx, 'KGESS'] = kgess
        metrics.loc[idx, 'CORRELATION'] = correlation
        metrics.loc[idx, 'RMSE'] = rmse
        metrics.loc[idx, 'BIAS'] = bias

    metrics.to_csv(f'{OUTPUTFOLDER}\\{model_name}\\metrics.csv', index=False, encoding='utf-8')

    return None

# ---------------------------------------------------timeseries----------------------------------------------------------
def plot_timeseries(sim_dirs, ref_dir, fig_dir, case_plot, station_ID, OUTPUTFOLDER, STARTYEAR, ENDYEAR):
    font = {'family': 'Times New Roman'}
    matplotlib.rc('font', **font)

    ref_path = f'{ref_dir}{station_ID}_Q_Day.Cmd.nc'
    print(f'station: {station_ID}, ref_path: {ref_path}')
    data = {}
    time_index = None
    for target in sim_dirs:
        sim_path = f'{OUTPUTFOLDER}\\{target}\\station\\{station_ID}.nc'
        with xr.open_dataset(sim_path) as df:
            sim_data = df['outflw'].values * 86400
            start_date = pd.to_datetime(df.time.min().values)
            end_date = pd.to_datetime(df.time.max().values)
        full_time = pd.date_range(start_date, end_date, freq='D')
        with xr.open_dataset(ref_path) as df1:
            ds = df1.reindex(time=full_time).ffill('time')
            ref_data = ds['discharge'].sel(time=slice(start_date, end_date)).squeeze(drop=True).values
        data[f'{target}'] = sim_data
        data['ref'] = ref_data
        time_index = full_time
        print(f'station: {station_ID}, sim_path: {sim_path}')

    plt.figure(figsize=(12, 8))

    if case_plot == 'value':
        plt.plot(time_index, data['ref'], label='Reference', color='#0000FF', linewidth=2, zorder=1)
        colors = ['#82afda', '#f79059', '#c2bdde']
        for i, target in enumerate(sim_dirs):
            if target == 'Fortran':
                plt.plot(time_index, data[target], label=target, color='#FF69B4', linewidth=4, zorder=2)
            elif target == 'Pytorch':
                plt.plot(time_index, data[target], label=target, color='#000000', linestyle=':', linewidth=4, zorder=3)
        plt.ylabel('Discharge (m³/s)', labelpad=20, fontsize=25)
        plt.title(f'Station {station_ID}: Discharge Time Series ({STARTYEAR}–{ENDYEAR})', pad=20, fontsize=25)

    elif case_plot == 'residual':
        colors = ['#82afda', '#f79059', '#c2bdde']
        for i, target in enumerate(sim_dirs):
            residual = (data[target] - data['ref']) / data['ref'] * 100
            if target == 'Fortran':
                plt.plot(time_index, residual, label=f'{target} Residual', color='#82c9ff',linewidth=4, zorder=1)
            elif target == 'Pytorch':
                plt.plot(time_index, residual, label=f'{target} Residual', color='#274753',linestyle=':', linewidth=4, zorder=2)
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.ylabel('Residual (%)', labelpad=20, fontsize=25)
        plt.title(f'Station {station_ID}: Residual Time Series ({STARTYEAR}–{ENDYEAR})', pad=20, fontsize=25)

    else:
        print(f"Invalid case_plot value: {case_plot}. Choose 'value' or 'residual'.")
        plt.close()
        return None

    plt.xlabel('Time', labelpad=25, fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(loc='upper left',fontsize=20)
    plt.grid(False)
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f'station_{station_ID}_{case_plot}.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Figure saved: {fig_path}')

    return None

#-----------------------------------------Discharge Spatial-------------------------------------------------------------
def plot_spatial_map(stn_lon, stn_lat, metric, indicator, fig_dir, selected_sim,
                     min_lon=-125.0, max_lon=-66.0, min_lat=22.0, max_lat=54.0):
    font = {'family': 'Times New Roman'}
    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(12, 8))

    # 设置 Basemap
    M = Basemap(projection='cyl', llcrnrlat=min_lat, urcrnrlat=max_lat,
                llcrnrlon=min_lon, urcrnrlon=max_lon, resolution='l')
    M.drawmapboundary(fill_color='white', zorder=-1)
    M.fillcontinents(color='0.8', lake_color='white', zorder=0)
    M.drawcoastlines(color='#BEBEBE', linewidth=0.1)

    # 添加经纬度网格
    parallels = np.arange(int(min_lat), int(max_lat) + 1, 5)
    meridians = np.arange(int(min_lon), int(max_lon) + 1, 10)
    M.drawparallels(parallels, labels=[1, 0, 0, 1], fontsize=20, linewidth=0.2)
    M.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=20, linewidth=0.2)

    # 设置不同指标的取值范围
    if indicator.upper() == 'KGESS':
        vmin, vmax = -1, 1
    elif indicator.upper() == 'CORRELATION':
        vmin, vmax = -1, 1
    elif indicator.upper() == 'RMSE':
        vmin, vmax = 0, max(100, np.max(metric) if len(metric) > 0 else 100)
    elif indicator.upper() == 'BIAS':
        vmin, vmax = -100, 100
    else:
        vmin, vmax = (np.min(metric) if len(metric) > 0 else -1,
                      np.max(metric) if len(metric) > 0 else 1)

    # 设置 colormap 对应关系
    cmap_dict = {
        'KGESS': 'coolwarm',
        'CORRELATION': 'RdPu',
        'RMSE': 'inferno',
        'BIAS': 'PRGn'
    }
    # 默认 colormap
    cmap = cmap_dict.get(indicator.upper(), 'inferno')

    # 绘制散点图
    loc_lon, loc_lat = M(stn_lon, stn_lat)
    sc = M.scatter(loc_lon, loc_lat, c=metric, s=150,
                   cmap=cmap, vmin=vmin, vmax=vmax,
                   marker='.', edgecolors='none', alpha=0.9)

    # 颜色条
    cbar = M.colorbar(sc, location='bottom', pad='12%')
    cbar.set_label(indicator, fontsize=25)
    cbar.ax.tick_params(labelsize=18)

    plt.title(f'Spatial Map of {indicator} for {selected_sim}', fontsize=25, pad=15)

    # 保存图像
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f'spatial_{selected_sim}_{indicator}.png')
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f'Spatial map saved: {fig_path}')

# def plot_spatial_map(stn_lon, stn_lat, metric, indicator, fig_dir, selected_sim, min_lon=-125.0, max_lon=-66.0,
#                      min_lat=22.0, max_lat=54.0):
#     font = {'family': 'Times New Roman'}
#     matplotlib.rc('font', **font)
#
#     fig = plt.figure(figsize=(12, 8))
#
#     # 设置 Basemap
#     M = Basemap(projection='cyl', llcrnrlat=min_lat, urcrnrlat=max_lat,
#                 llcrnrlon=min_lon, urcrnrlon=max_lon, resolution='l')
#     M.drawmapboundary(fill_color='white', zorder=-1)
#     # M.fillcontinents(color='#D3D3D3', lake_color='white', zorder=0)
#     M.fillcontinents(color='0.8', lake_color='white', zorder=0)
#     M.drawcoastlines(color='#BEBEBE', linewidth=0.1)
#
#     # 添加经纬度网格
#     parallels = np.arange(int(min_lat), int(max_lat) + 1, 5)
#     meridians = np.arange(int(min_lon), int(max_lon) + 1, 10)
#     M.drawparallels(parallels, labels=[1, 0, 0, 1], fontsize=20, linewidth=0.2)
#     M.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=20, linewidth=0.2)
#
#     if indicator.upper() == 'KGESS':
#         vmin, vmax = -1, 1
#     elif indicator.upper() == 'CORRELATION':
#         vmin, vmax = -1, 1
#     elif indicator.upper() == 'RMSE':
#         vmin, vmax = 0, max(100, np.max(metric) if len(metric) > 0 else 100)
#     elif indicator.upper() == 'BIAS':
#         vmin, vmax = -100, 100
#     else:
#         vmin, vmax = np.min(metric) if len(metric) > 0 else -1, np.max(metric) if len(metric) > 0 else 1
#
#     # 绘制散点图
#     loc_lon, loc_lat = M(stn_lon, stn_lat)
#     sc = M.scatter(loc_lon, loc_lat, c=metric, s=150,cmap='inferno', vmin=vmin, vmax=vmax,
#                    marker='.', edgecolors='none', alpha=0.9)
#
#     cbar = M.colorbar(sc, location='bottom', pad='12%')
#     cbar.set_label(indicator, fontsize=25)
#     cbar.ax.tick_params(labelsize=18)
#
#     plt.title(f'Spatial Map of {indicator} for {selected_sim}', fontsize=25, pad=15)
#
#     # 保存图像
#     os.makedirs(fig_dir, exist_ok=True)
#     fig_path = os.path.join(fig_dir, f'spatial_{selected_sim}_{indicator}.png')
#     plt.savefig(fig_path, dpi=300,bbox_inches="tight")
#     plt.close()
#     print(f'Spatial map saved: {fig_path}')
#



#----------------------------------------------maximum flood deepth timeseries-----------------------------
def plot_flood_depth_timeseries(SIMDIR,sim_dirs, fig_dir, case_plot, OUTPUTFOLDER, STARTYEAR, ENDYEAR):

    font = {'family': 'Times New Roman'}
    matplotlib.rc('font', **font)

    data = {}
    time_index = None

    for target in sim_dirs:
        files = []
        years = list(range(STARTYEAR, ENDYEAR + 1))
        filenames = [f'{SIMDIR}o_maxdph{year}.nc' for year in years]

        if os.path.exists(f):
                files.append(f)

        if not files:
            print(f"No maxdph files found for {target}")
            continue

        ds = xr.open_mfdataset(files, combine="by_coords")
        da = ds["maxdph"]

        # 取每一天的空间平均最大洪水深度
        mean_series = da.mean(dim=["lat", "lon"], skipna=True).to_pandas()

        data[target] = mean_series.values
        time_index = mean_series.index

    if not data:
        print("No flood depth data available for plotting.")
        return None

    plt.figure(figsize=(12, 8))
    if case_plot == 'value':
        for target in sim_dirs:
            plt.plot(time_index, data[target], label=target, linewidth=1.5)
        plt.ylabel("Max Flood Depth (m)", labelpad=20, fontsize=16)
        plt.title(f"Maximum Flood Depth Time Series ({STARTYEAR}–{ENDYEAR})", fontsize=18)

    plt.xlabel("Time", labelpad=20, fontsize=16)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f"flood_depth_{case_plot}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Flood depth time series saved: {fig_path}")
    return fig_path

#-----------------------------------------------------maximum flood deepth spatial---------------------------------------------
def plot_flood_deepth_map(data, title, filename, min_lon=-125.0, max_lon=-66.0, min_lat=22.0, max_lat=54.0):
    lon = np.linspace(min_lon, max_lon, data.shape[1])
    lat = np.linspace(min_lat, max_lat, data.shape[0])
    lon2d, lat2d = np.meshgrid(lon, lat)
    font = {'family': 'Times New Roman'}
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(12, 8))
    M = Basemap(projection='cyl', llcrnrlat=min_lat, urcrnrlat=max_lat,
                llcrnrlon=min_lon, urcrnrlon=max_lon, resolution='l')
    M.drawmapboundary(fill_color='white', zorder=-1)
    M.fillcontinents(color='0.8', lake_color='white', zorder=0)
    M.drawcoastlines(color='0.6', linewidth=0.1)

    parallels = np.arange(int(min_lat), int(max_lat) + 1, 5)
    meridians = np.arange(int(min_lon), int(max_lon) + 1, 10)
    M.drawparallels(parallels, labels=[1, 0, 0, 1], fontsize=20, linewidth=0.2)
    M.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=20, linewidth=0.2)

    cs = M.contourf(lon2d, lat2d, data, levels=50, cmap='jet', latlon=True)
    cbar = M.colorbar(cs, location='bottom', pad='12%')
    cbar.set_label(label='Max Depth (m)', fontsize=25)
    cbar.ax.tick_params(labelsize=18)
    plt.title(title, fontsize=25, pad=15)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f'save: {filename}')
    return filename

# ---------------------------------------------------flooded area timeseries----------------------------------------------------------
def plot_flooded_area_timeseries(SIMDIR,sim_dirs, fig_dir, case_plot, OUTPUTFOLDER, STARTYEAR, ENDYEAR):
    font = {'family': 'Times New Roman'}
    matplotlib.rc('font', **font)

    data = {}
    time_index = None

    for target in sim_dirs:
        files = []
        years = list(range(STARTYEAR, ENDYEAR + 1))
        f = [f'{SIMDIR}o_fldare{year}.nc' for year in years]
        if os.path.exists(f):
            files.append(f)

        if not files:
            print(f"No fldare files found for {target}")
            continue

        ds = xr.open_mfdataset(files, combine="by_coords")
        da = ds["fldare"]

        # 取每一天的空间总和flooded area（总淹没面积）
        sum_series = da.sum(dim=["lat", "lon"], skipna=True).to_pandas()  # 使用sum作为总面积；如果要平均，用mean()

        data[target] = sum_series.values
        time_index = sum_series.index

    if not data:
        print("No flooded area data available for plotting.")
        return None

    plt.figure(figsize=(12, 8))
    if case_plot == 'value':
        for target in sim_dirs:
            plt.plot(time_index, data[target], label=target, linewidth=1.5)
        plt.ylabel("Total Flooded Area (sq km)", labelpad=20, fontsize=16)
        plt.title(f"Flooded Area Time Series ({STARTYEAR}–{ENDYEAR})", fontsize=18)


    plt.xlabel("Time", labelpad=20, fontsize=16)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f"flooded_area_{case_plot}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Flooded area time series saved: {fig_path}")
    return fig_path

# ---------------------------------------------------flooded area spatial----------------------------------------------------------

def plot_fldaremap(data, title, filename, min_lon=-125.0, max_lon=-66.0, min_lat=22.0, max_lat=54.0):

    lon = np.linspace(min_lon, max_lon, data.shape[1])
    lat = np.linspace(min_lat, max_lat, data.shape[0])
    lon2d, lat2d = np.meshgrid(lon, lat)
    font = {'family': 'Times New Roman'}
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(12, 8))
    M = Basemap(projection='cyl', llcrnrlat=min_lat, urcrnrlat=max_lat,
                llcrnrlon=min_lon, urcrnrlon=max_lon, resolution='l')
    M.drawmapboundary(fill_color='white', zorder=-1)
    M.fillcontinents(color='0.8', lake_color='white', zorder=0)
    M.drawcoastlines(color='0.6', linewidth=0.1)

    parallels = np.arange(int(min_lat), int(max_lat) + 1, 5)
    meridians = np.arange(int(min_lon), int(max_lon) + 1, 10)
    M.drawparallels(parallels, labels=[1, 0, 0, 1], fontsize=20, linewidth=0.2)
    M.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=20, linewidth=0.2)

    cs = M.contourf(lon2d, lat2d, data, levels=50, cmap='jet', latlon=True)
    cbar = M.colorbar(cs, location='bottom', pad='12%')
    cbar.set_label('Max Depth (m)', fontsize=25)
    cbar.ax.tick_params(labelsize=18)

    plt.title(title, fontsize=25, pad=15)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f'Flooded area spatial map saved: {filename}')
    return filename
# ---------------------------------------------------river network spatial ----------------------------------------------------------
def plot_river_network_spatial(SIMDIR,STARTYEAR,ENDYEAR,sim_full_dir, fig_dir, start_year=2000, end_year=2020, min_lon=-180, max_lon=180, min_lat=-90, max_lat=90):
    files = []
    for year in range(start_year, end_year + 1):
        years = list(range(STARTYEAR, ENDYEAR + 1))
        f = [f'{SIMDIR}o_pthflw{year}.nc' for year in years]
        if os.path.exists(f):
            files.append(f)
            break  # 只取第一个文件作为静态河网

    if not files:
        raise FileNotFoundError(f"No pthout files found in {sim_full_dir}")

    ds = xr.open_dataset(files[0])  # 只打开一个文件
    da = ds["pthout"]

    # 取第一个时间步作为静态图
    da_plot = da.isel(time=0) if "time" in da.dims else da

    # 经纬度
    lat_name = "lat" if "lat" in da.dims else "y"
    lon_name = "lon" if "lon" in da.dims else "x"
    lats, lons = ds[lat_name].values, ds[lon_name].values
    if lats.ndim == 1 and lons.ndim == 1:
        lon2d, lat2d = np.meshgrid(lons, lats)
    else:
        lon2d, lat2d = lons, lats

    field = da_plot.values

    # 绘图
    fig = plt.figure(figsize=(12, 8))
    M = Basemap(projection="cyl",
                llcrnrlat=min_lat, urcrnrlat=max_lat,
                llcrnrlon=min_lon, urcrnrlon=max_lon,
                resolution="l")
    M.drawmapboundary(fill_color="white")
    M.fillcontinents(color="0.85", lake_color="white")
    M.drawcoastlines(color="0.4", linewidth=0.3)

    x, y = M(lon2d, lat2d)
    cs = M.pcolormesh(x, y, field, cmap="viridis", shading="auto")
    cbar = plt.colorbar(cs, orientation="horizontal", pad=0.07)
    cbar.set_label("River Network (pthout)")

    model_name = os.path.basename(os.path.normpath(sim_full_dir))
    plt.title(f"River Network Spatial Map — {model_name}")
    os.makedirs(fig_dir, exist_ok=True)
    out_png = os.path.join(fig_dir, f"river_network_spatial_{model_name}.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"River network spatial map saved: {out_png}")
    return out_png

# ---------------------------------------------------water level spatial ----------------------------------------------------------
def plot_water_level_spatial(SIMDIR,STARTYEAR,ENDYEAR,sim_full_dir, fig_dir, start_year=2000, end_year=2020, min_lon=-180, max_lon=180, min_lat=-90, max_lat=90):
    files = []
    for year in range(start_year, end_year + 1):
        years = list(range(STARTYEAR, ENDYEAR + 1))
        f = [f'{SIMDIR}o_sfcelv{year}.nc' for year in years]
        if os.path.exists(f):
            files.append(f)

    if not files:
        raise FileNotFoundError(f"No sfcelv files found in {sim_full_dir}")

    ds = xr.open_mfdataset(files, combine="by_coords")
    da = ds["sfcelv"]

    # 时间平均空间分布
    da_plot = da.mean(dim="time", skipna=True)

    # 经纬度
    lat_name = "lat" if "lat" in da.dims else "y"
    lon_name = "lon" if "lon" in da.dims else "x"
    lats, lons = ds[lat_name].values, ds[lon_name].values
    if lats.ndim == 1 and lons.ndim == 1:
        lon2d, lat2d = np.meshgrid(lons, lats)
    else:
        lon2d, lat2d = lons, lats

    field = da_plot.values

    # 绘图
    fig = plt.figure(figsize=(14, 10))
    M = Basemap(projection="cyl",
                llcrnrlat=min_lat, urcrnrlat=max_lat,
                llcrnrlon=min_lon, urcrnrlon=max_lon,
                resolution="l")
    M.drawmapboundary(fill_color="white")
    M.fillcontinents(color="0.85", lake_color="white")
    M.drawcoastlines(color="0.4", linewidth=0.3)

    x, y = M(lon2d, lat2d)
    cs = M.pcolormesh(x, y, field, cmap="viridis", shading="auto")
    cbar = plt.colorbar(cs, orientation="horizontal", pad=0.07)
    cbar.set_label("Average Water Level (m)")

    model_name = os.path.basename(os.path.normpath(sim_full_dir))
    plt.title(f"Average Water Level Spatial Map ({start_year}-{end_year}) — {model_name}")
    os.makedirs(fig_dir, exist_ok=True)
    out_png = os.path.join(fig_dir, f"water_level_spatial_{model_name}.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Water level spatial map saved: {out_png}")
    return out_png


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='D:/pythonProject/0619Cama-AI/evaluate/evl_config.yml', help='Path to config file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    OUTPUTFOLDER = 'D:\\LZH_CaMa\\'
    process_simulations(config, OUTPUTFOLDER)
