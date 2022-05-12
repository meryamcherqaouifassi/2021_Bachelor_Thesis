# -*- coding: utf-8 -*-
"""
Spyder Editor

This script is made to visualise data from era5 with cartopy
"""

# Imports 
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import matplotlib as mpl

# Load data 
path_era5 = '/work/FAC/FGSE/IDYST/tbeucler/default/meryam/data/era5/media/rasp'+\
    '/Elements/weather-benchmark' 
ds = xr.open_dataset(path_era5+'/1.40625deg/2m_temperature/'+\
                               '2m_temperature_1979_1.40625deg.nc')
    
time_test = np.datetime64('1979-12-01T00:00:00.000000000')  #datetime64 format enables to pick a year, daytime, hour...

# Geographic coordinates 
central_lon, central_lat = 33.5731104, -7.5898434   #coordinates of Casablanca
lat_min = 28
lat_max = 35
lon_min = -11
lon_max = -4
extent = [lon_min, lon_max, lat_min, lat_max]

norm = mpl.colors.Normalize(vmin=283.15, vmax=303.15)   #sets a temperature range colorbar

# Visualise data 
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_lon)})
ax.set_extent(extent)
ax.coastlines(resolution='50m')
map = ds.sel(time=time_test)['t2m'].plot.imshow(cmap='coolwarm', norm=norm)

sub_set = ds.sel(time=time_test)['t2m']

lon_indices = np.logical_and(ds.lon>=175,   #gives the longitudes array's indices of the area around Casablanca
                        ds.lon<=182)

lat_indices = np.logical_and(ds.lat>=28,    #gives the latitudes array's indices of the area around Casablanca
                        ds.lat<=38)
























