# this script is made to try different bilinear interpolations of Era5 temperataures around Casablanca

# Imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from metpy.interpolate import interpolate_to_grid
#import arcpy

# ds_era5 is a dataset object of xarrat.core.dataset module
# path of era5 dataset
path_era5 = '/work/FAC/FGSE/IDYST/tbeucler/default/meryam/data/era5/media/rasp/Elements/weather-benchmark/1.40625deg/2m_temperature/'
# load era5 dataset
ds_era5 = xr.open_mfdataset(path_era5+'2m_temperature_*_1.40625deg.nc')
# convert era5 temperatures from Kelvin to Celsius
ds_era5['t2m'] = ds_era5['t2m'] - 273.15

# isolate era5 temperatures around casablanca
lon_indices = np.logical_and(ds_era5.lon>=170,ds_era5.lon<=184)
    #gives the longitudes array's indices of the area around Casablanca
lat_indices = np.logical_and(ds_era5.lat>=28.5,ds_era5.lat<=39)
    #gives the latitudes array's indices of the area around Casablanca
ds_era5 = ds_era5.isel({'lon':lon_indices,'lat':lat_indices})

# ds_casa is a dataframe
# path of casa dataset
path_casa = '/work/FAC/FGSE/IDYST/tbeucler/default/meryam/2021_Bachelor_Thesis/files'
# load casa dataset
ds_casa = pd.read_csv(path_casa+'/data_casa.csv')
# keep only rows with HGHT value equal 58
ds_casa = ds_casa.loc[ds_casa['HGHT'] == 58]
# extract TEMP corresponding to a certain value of HGHT
ds_casa = ds_casa.loc[:,['DATE', 'TEMP']]

# reshape the data to have the same temporal resolution
ds_casa['DATE'] = pd.to_datetime(ds_casa['DATE'])
ds_era5['time'] = pd.to_datetime(ds_era5['time'])
# keep only times = 00:00
#ds_casa = ds_casa.loc[ds_casa['DATE'].dt.hour == 12]
ds_casa = ds_casa.sort_values(by='DATE')
ds_era5.where(ds_era5['time.hour']==12, drop=True)
ds_casa = ds_casa[np.logical_and(ds_casa.DATE.dt.year>=1979,ds_casa.DATE.dt.year<=2018)]
#ds_casa = ds_casa.dropna()
# make data_casa temperatures a float
ds_casa['TEMP'] = ds_casa['TEMP'].astype(np.float32)
# remove ds_casa outliers < 0 and > 45
ds_casa = ds_casa[ds_casa['TEMP'] > 0]
ds_casa = ds_casa[ds_casa['TEMP'] < 45]
ds_era5 = ds_era5.sel(time=ds_casa.DATE.to_numpy())

# set casablanca coordinates
x0 = 33.5883100     # latitude
y0 = 172.38862 #-3.80569       # longitude
coords = [x0, y0]

# linear interpolation 
lin_interp = ds_era5.interp({'lat':x0, 'lon':y0}, method='linear')
# nearest interpolation
near_interp = ds_era5.interp({'lat':x0, 'lon':y0}, method='nearest')
# quadratic interpolation
quad_interp = ds_era5.interp({'lat':x0, 'lon':y0}, method='quadratic')
# cubic interpolation
cub_interp = ds_era5.interp({'lat':x0, 'lon':y0}, method='cubic')

# plot of the observed and interpolated temperature
fig, ax = plt.subplots(dpi = 200)
ax.plot(ds_casa['DATE'],ds_casa['TEMP'], label = 'Observed')
lin_interp.plot.scatter('time', 't2m', label = 'Interpolated', s = 0.25, c = 'black')
ax.set_xlabel('Date')
ax.set_ylabel('Temperature (°C)')
ax.legend()
plt.tight_layout()
plt.show()

# select from ds_era5 the year 2000
ds_era5_2000 = ds_era5.sel(time=ds_era5.time.dt.year == 2000)
# linear interpolation of ds_era5_2000
lin_interp_2000 = ds_era5_2000.interp({'lat':x0, 'lon':y0}, method='linear')
# plot of ds_casa + lin_interp
fig, ax = plt.subplots(dpi = 200)
ax.plot(ds_casa.loc[ds_casa.DATE.dt.year == 2000, 'DATE'],ds_casa.loc[ds_casa.DATE.dt.year == 2000, 'TEMP'], label = 'Observed')
lin_interp_2000.plot.scatter('time', 't2m', label = 'Interpolated', s = 0.25, c = 'black')
ax.set_xlabel('Date')
ax.set_ylabel('Temperature (°C)')
ax.legend()
plt.tight_layout()
plt.show()

# nearest-neighbor interpolation of ds_era5_2000
near_interp_2000 = ds_era5_2000.interp({'lat':x0, 'lon':y0}, method='nearest')
# plot of ds_casa + lin_interp
fig, ax = plt.subplots(dpi = 200)
ax.plot(ds_casa.loc[ds_casa.DATE.dt.year == 2000, 'DATE'],ds_casa.loc[ds_casa.DATE.dt.year == 2000, 'TEMP'], label = 'Observed')
near_interp_2000.plot.scatter('time', 't2m', label = 'Interpolated', s = 0.25, c = 'black')
ax.set_xlabel('Date')
ax.set_ylabel('Temperature (°C)')
ax.legend()
plt.tight_layout()
plt.show()

# quadratic interpolation of ds_era5_2000
quad_interp_2000 = ds_era5_2000.interp({'lat':x0, 'lon':y0}, method='quadratic')
# plot of ds_casa + lin_interp
fig, ax = plt.subplots(dpi = 200)
ax.plot(ds_casa.loc[ds_casa.DATE.dt.year == 2000, 'DATE'],ds_casa.loc[ds_casa.DATE.dt.year == 2000, 'TEMP'], label = 'Observed')
quad_interp_2000.plot.scatter('time', 't2m', label = 'Interpolated', s = 0.25, c = 'black')
ax.set_xlabel('Date')
ax.set_ylabel('Temperature (°C)')
ax.legend()
plt.tight_layout()
plt.show()

# cubic interpolation of ds_era5_2000
cub_interp_2000 = ds_era5_2000.interp({'lat':x0, 'lon':y0}, method='cubic')
# plot of ds_casa + lin_interp
fig, ax = plt.subplots(dpi = 200)
ax.plot(ds_casa.loc[ds_casa.DATE.dt.year == 2000, 'DATE'],ds_casa.loc[ds_casa.DATE.dt.year == 2000, 'TEMP'], label = 'Observed')
cub_interp_2000.plot.scatter('time', 't2m', label = 'Interpolated', s = 0.25, c = 'black')
ax.set_xlabel('Date')
ax.set_ylabel('Temperature (°C)')
ax.legend()
plt.tight_layout()
plt.show()

# observed vs interpolated temperatures plot
fig, ax = plt.subplots(dpi=200)
ax.plot(ds_casa.loc[ds_casa.DATE.dt.year == 2000, 'TEMP'], lin_interp_2000['t2m'])

