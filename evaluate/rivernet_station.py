import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, box
import numpy as np
import re
import cartopy.crs as ccrs 
import cartopy.feature as cfeature  

rivers_fp = r"E:\HydroRIVERS_v10_na_shp\HydroRIVERS_v10_na_shp\HydroRIVERS_v10_na.shp"
rivers_gdf = gpd.read_file(rivers_fp)

world_fp = r"E:\countries\ne_110m_admin_0_countries.shp"
world_gdf = gpd.read_file(world_fp)
usa_gdf = world_gdf[world_gdf["NAME"] == "United States of America"]

rivers_gdf = rivers_gdf.to_crs(epsg=4326)
usa_gdf = usa_gdf.to_crs(epsg=4326)

min_lon, max_lon = -125.0, -66.0
min_lat, max_lat = 22.0, 54.0
bbox = box(min_lon, min_lat, max_lon, max_lat)

filtered_rivers = gpd.clip(rivers_gdf, bbox)
clipped_usa = gpd.clip(usa_gdf, bbox)

main_rivers = filtered_rivers[filtered_rivers["DIS_AV_CMS"] >= 1000]
secondary_rivers = filtered_rivers[(filtered_rivers["DIS_AV_CMS"] >= 100) & (filtered_rivers["DIS_AV_CMS"] < 1000)]
# small_rivers = filtered_rivers[filtered_rivers["DIS_AV_CMS"] < 100]
stations = []
with open("new_selectstation.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        m = re.match(r"(\d+):\s*\(\s*([-\d.]+),\s*([-\d.]+)\s*\)", line)
        if m:
            sid = m.group(1)
            lon = float(m.group(2))
            lat = float(m.group(3))
            stations.append({"station_id": sid, "geometry": Point(lon, lat)})
        else:
            print("error：", line)

if stations:
    stations_gdf = gpd.GeoDataFrame(stations, geometry="geometry", crs="EPSG:4326")
    stations_gdf = gpd.clip(stations_gdf, bbox)
else:
    stations_gdf = gpd.GeoDataFrame(columns=["station_id", "geometry"], crs="EPSG:4326")

fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})  


# ax.add_feature(cfeature.COASTLINE, linewidth=0.8)  
# ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5) 
ax.add_feature(cfeature.LAND, edgecolor='white', facecolor='#E8ECEF') 
# ax.add_feature(cfeature.OCEAN, edgecolor='/')  

clipped_usa.plot(ax=ax, color="none", edgecolor="white", zorder=1)

main_rivers.plot(ax=ax, color="#0076A2", linewidth=2.5, zorder=4)
secondary_rivers.plot(ax=ax, color="#00BFFF", linewidth=0.8, zorder=3)
# small_rivers.plot(ax=ax, color='#89C3EB', linewidth=0.3, label="Small Rivers", zorder=2)

if not stations_gdf.empty:
    stations_gdf.plot(ax=ax, color='#FF4D00', markersize=60, marker='h', zorder=5)
min_lon, max_lon = -125.0, -66.0
min_lat, max_lat = 22.0, 54.0

parallels = np.arange(int(min_lat), int(max_lat) + 1, 5) 
meridians = np.arange(int(min_lon), int(max_lon) + 1, 10) 
ax.set_xticks(meridians)
ax.set_yticks(parallels)

ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
ax.set_aspect('equal')
ax.margins(0)
# ax.set_title("Major River Networks in the USA with Stations", fontsize=25, pad=15)
# ax.set_xlabel("Longitude", fontsize=20)
# ax.set_ylabel("Latitude", fontsize=20)
# ax.legend(loc="lower right", fontsize=15)
plt.savefig('rivernet_with_coastline.png', dpi=300, bbox_inches="tight")

plt.show()
