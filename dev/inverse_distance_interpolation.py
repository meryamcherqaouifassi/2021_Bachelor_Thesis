#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 11:26:51 2022

@author: mcherqao
"""
# Imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
#import arcpy

# path of era5 dataset
path_era5 = '/work/FAC/FGSE/IDYST/tbeucler/default/meryam/data/era5/media/rasp/Elements/weather-benchmark/1.40625deg/2m_temperature/'
# load era5 dataset
ds_era5 = xr.open_mfdataset(path_era5+'2m_temperature_*_1.40625deg.nc')

# path of casa dataset
path_casa = '/work/FAC/FGSE/IDYST/tbeucler/default/meryam/2021_Bachelor_Thesis/files'
# load casa dataset
ds_casa = pd.read_csv(path_casa+'/data_casa.csv')
# keep only rows with HGHT value equal 58
ds_casa = ds_casa.loc[ds_casa['HGHT'] == "58"]
# extract TEMP corresponding to a certain value of HGHT
ds_casa = ds_casa.loc[:,['DATE', 'TEMP']]

# isolate era5 temperatures around casablanca
lon_indices = np.logical_and(ds_era5.lon>=172,ds_era5.lon<=184)
    #gives the longitudes array's indices of the area around Casablanca
lat_indices = np.logical_and(ds_era5.lat>=28.5,ds_era5.lat<=39)
    #gives the latitudes array's indices of the area around Casablanca
ds_era5 = ds_era5.isel({'lon':lon_indices,'lat':lat_indices})


# convert era5 temperatures from Kelvin to Celsius
ds_era5['t2m'] = ds_era5['t2m'] - 273.15

# set casablanca coordinates
x0 = 33.5883100 # latitude
y0 = -3.80569 # longitude

# reshape the data to have the same temporal resolution
ds_casa['DATE'] = pd.to_datetime(ds_casa['DATE'])
ds_casa = ds_casa.sort_values(by='DATE')
ds_era5.where(ds_era5['time.hour']==12, drop=True)
ds_casa = ds_casa[np.logical_and(ds_casa.DATE.dt.year>=1979,ds_casa.DATE.dt.year<=2018)]
ds_era5 = ds_era5.sel(time=ds_casa.DATE.to_numpy())

# draw a gridded map of the area
lat_min = 28.5
lat_max = 39
lon_min = -14
lon_max = 0
extent = [lon_min, lon_max, lat_min, lat_max]
# Visualise area 
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(y0)})
ax.set_extent(extent)
ax.coastlines(resolution='50m')
ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
ax.plot(y0, x0, marker='.', markersize=40, color='red')
ax.set_xlabel('longitude')
ax.set_ylabel('latitude')
ax.set_extent([lon_min, lon_max, lat_min, lat_max])

# Visualise area closer 
lat_min = 32
lat_max = 35
lon_min = -10
lon_max = -5
extent = [lon_min, lon_max, lat_min, lat_max]
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(y0)})
ax.set_extent(extent)
ax.coastlines(resolution='50m')
ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
ax.plot(y0, x0, marker='.', markersize=30, color='red')
ax.set_xlabel('longitude')
ax.set_ylabel('latitude')
ax.set_extent([lon_min, lon_max, lat_min, lat_max])

# function that calculates the distance between era5 points and casablanca coordinates
def distance(x,y):
    return np.sqrt((x-x0)**2 + (y-y0)**2)

# go trough all the ds_era5 points and calculate the distance between them and casablanca and put them in a new array
distances = np.zeros(len(ds_era5.lat))
for i in range(len(ds_era5.lat)):
    distances[i] = distance(ds_era5.lon[i], ds_era5.lat[i])

# calculate the sum of the distances
sum_distances = np.sum(distances)

# calculate de weight of each ds_era5 point
weights = np.zeros(len(ds_era5.lat))
for i in range(len(ds_era5.lat)):
    weights[i] = distances[i]/sum_distances

# calculate casablanca's temperatures as the inverse distance interpolation of the ds_era5 points for each day
temps_interpolated = np.zeros(len(ds_casa.DATE))
for i in range(len(ds_casa.DATE)):
    temps_interpolated[i] = np.sum(weights*ds_era5.t2m[i])

# function that calculate the distance between era5 gridpoints and casa coordinates
#def distance(x, y, x0, y0):
#    return np.sqrt((x - x0)**2 + (y - y0)**2)

# go through all the era5 gridpoints and calculate the distance to casa 
# and put it in a new array
#dist = np.zeros((ds_era5.lat.size, ds_era5.lon.size))
#for i in range(ds_era5.lat.size):
#    for j in range(ds_era5.lon.size):
#        dist[i,j] = distance(ds_era5.lat[i], ds_era5.lon[j], x0, y0)

# calculate de sum of the distances 
#dist_sum = np.sum(dist)

# calculate the weight of each era5 gridpoint
#weight = np.zeros((ds_era5.lat.size, ds_era5.lon.size))
#for i in range(ds_era5.lat.size):
#    for j in range(ds_era5.lon.size):
#        weight[i,j] = dist[i,j]/dist_sum

# calculate casablanca's temperature as the weighted mean of the era5 temperatures
#casa_temp = np.sum(ds_era5.t2m*weight)

# calculate the weight of each era5 gridpoint
#weight = dist/dist_sum

# calculate casablanca's temperature as the weighted mean of the era5 temperatures
#casa_temp = weight*ds_era5.t2m.values

# put casablanca's temperature in a dataframe with the date as a column
#casa_temp_df = pd.DataFrame(casa_temp, columns=['TEMP'])
#casa_temp_df['DATE'] = ds_casa.DATE


        




