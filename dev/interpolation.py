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
ds_casa = ds_casa.loc[ds_casa['HGHT'] == "58"]
# extract TEMP corresponding to a certain value of HGHT
ds_casa = ds_casa.loc[:,['DATE', 'TEMP']]

# reshape the data to have the same temporal resolution
ds_casa['DATE'] = pd.to_datetime(ds_casa['DATE'])
ds_casa = ds_casa.sort_values(by='DATE')
ds_era5.where(ds_era5['time.hour']==12, drop=True)
ds_casa = ds_casa[np.logical_and(ds_casa.DATE.dt.year>=1979,ds_casa.DATE.dt.year<=2018)]
ds_era5 = ds_era5.sel(time=ds_casa.DATE.to_numpy())
# make data_casa temperatures a float
ds_casa['TEMP'] = ds_casa['TEMP'].astype(np.float32)
# remove ds_casa outliers < 0 and > 45
ds_casa = ds_casa[ds_casa['TEMP'] > 0]
ds_casa = ds_casa[ds_casa['TEMP'] < 45]

# set casablanca coordinates
x0 = 33.5883100     # latitude
y0 = 172.38862 #-3.80569       # longitude
coords = [x0, y0]

# draw a gridded map of the area
lat_min = 28.5
lat_max = 39
lon_min = -14
lon_max = 0
extent = [lon_min, lon_max, lat_min, lat_max]
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(y0)})
ax.set_extent(extent)
ax.coastlines(resolution='50m')
ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
ax.plot(y0, x0, marker='.', markersize=40, color='red')
ax.set_xlabel('longitude')
ax.set_ylabel('latitude')
ax.set_extent([lon_min, lon_max, lat_min, lat_max])

# linear interpolation 
lin_interp = ds_era5.interp({'lat':x0, 'lon':y0}, method='linear')
# nearest interpolation
near_interp = ds_era5.interp({'lat':x0, 'lon':y0}, method='nearest')
# quadratic interpolation
near_interp = ds_era5.interp({'lat':x0, 'lon':y0}, method='quadratic')
# cubic interpolation
near_interp = ds_era5.interp({'lat':x0, 'lon':y0}, method='cubic')

# plot of the observed and interpolated temperature
fig, ax = plt.subplots(dpi = 200)
ax.plot(ds_casa['DATE'],ds_casa['TEMP'], label = 'Observed')
lin_interp.plot.scatter('time', 't2m')
#ax.plot(lin_interp['time'],lin_interp['t2m'], label = 'Interpolated')
#ax.plot(lin_interp['t2m'], label = 'Interpolated')
ax.set_xlabel('Date')
ax.set_ylabel('Temperature (°C)')
ax.legend()
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(dpi = 200)
#ax.plot(ds_casa['DATE'], ds_casa['TEMP'].isel(time = '2000-01-01'))
#ax.plot(lin_interp['time'], lin_interp['t2m'].sel(time = '2000-01-01'))
#ax.plot(lin_interp)
plt.tight_layout()
ax.set_xlabel('Date')
ax.set_ylabel('Temperature (°C)')
plt.tight_layout()
plt.show()

# function that calculates the distance between era5 points and casablanca coordinates
#def distance(x,y):
#    return np.sqrt((x-x0)**2 + (y-y0)**2)

# go trough all the ds_era5 points and calculate the distance between them and casablanca and put them in a new array
#distances = np.zeros(len(ds_era5.lat))
#for i in range(len(ds_era5.lat)):
#    distances[i] = distance(ds_era5.lon[i], ds_era5.lat[i])

# add the distance array to the ds_era5 dataset
#ds_era5['distance'] = distances

# keep the 4 points with the smallest distance to casablanca
#ds_era5 = ds_era5.values[ds_era5['distance'] == np.min(ds_era5['distance'])]

# find the 4 closest points to casablanca
# sort the distances array
#distances = np.sort(distances)
# find the 4 closest points to casablanca
#closest_points = np.zeros(4)
#for i in range(4):
#    closest_points[i] =  distances[i]
# create a new dataframe with the 4 closest points
#ds_era5_closest = ds_era5.isel(time=closest_points)






# reduce era5 dataset to x0 + 5 and x0 - 5, y0 + 5 and y0 - 5
#ds_era5 = ds_era5.sel(lon=slice(x0-5,x0+5),lat=slice(y0-5,y0+5))

# using interpolate_to_grid function from metpy.interpolate module, 
# interpolate era5 temperatures to casablanca coordinates
#ds_era5_interpolated = interpolate_to_grid(x0, y0, ds_era5)
#gridx, ridy, t2m = interpolate_to_grid(x0, y0, ds_era5.t2m.values, interp_type='linear')
# extract interpolated temperatures
#t2m_interpolated = ds_era5_interpolated['t2m']

# make a map of era5 points around casablanca
#fig, ax = plt.subplots(figsize=(10,10), subplot_kw=dict(projection=ccrs.PlateCarree()))
#ax.set_extent([y0-0.5, y0+0.5, x0-0.5, x0+0.5])
#ax.coastlines()



# find the 4 closest points of ds_era5 to x0, y0
# find the distance between x0, y0 and each point of ds_era5
#distances = np.sqrt((ds_era5.lon - x0)**2 + (ds_era5.lat - y0)**2)
# find the index of the 4 closest points of ds_era5 to x0, y0
#indexes = np.argsort(distances)[:4]
# extract the 4 closest points of ds_era5 to x0, y0
#ds_era5_4 = ds_era5.isel(time=indexes)

