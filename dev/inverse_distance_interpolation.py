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

# path of era5 dataset
path_era5 = '/work/FAC/FGSE/IDYST/tbeucler/default/meryam/data/era5/media/rasp/Elements/weather-benchmark/1.40625deg/2m_temperature/'
# load era5 dataset
ds_era5 = xr.open_mfdataset(path_era5+'2m_temperature_*_1.40625deg.nc')

# path of casa dataset
path_casa = '/work/FAC/FGSE/IDYST/tbeucler/default/meryam/2021_Bachelor_Thesis/files'
# load casa dataset
ds_casa = pd.read_csv(path_casa+'data_casa.csv')
# extract TEMP corresponding to a value of HGHT
ds_casa = ds_casa.loc[:,['TEMP']].loc[ds_casa['HGHT'] == 58]

# isolate era5 temperatures around casablanca (étendre la zone ?)
lon_indices = np.logical_and(ds_era5.lon>=175,ds_era5.lon<=182)
    #gives the longitudes array's indices of the area around Casablanca
lat_indices = np.logical_and(ds_era5.lat>=28,ds_era5.lat<=38)
    #gives the latitudes array's indices of the area around Casablanca
ds_era5 = ds_era5.isel({'lon':lon_indices,'lat':lat_indices})
    #isolate Casablanca's temp data

# set casablanca coordinates
x0 = 33.5883100 # latitude
y0 = -7.6113800 # longitude

# calculate the distance between era5 gridpoints and casa coordinates
def distance(x, y, x0, y0):
    return np.sqrt((x - x0)**2 + (y - y0)**2)

# go through all the era5 gridpoints and calculate the distance to casa 
# and put it in a new array
dist = np.zeros((ds_era5.lat.size, ds_era5.lon.size))
for i in range(ds_era5.lat.size):
    for j in range(ds_era5.lon.size):
        dist[i,j] = distance(ds_era5.lat[i], ds_era5.lon[j], x0, y0)

# calculate de sum of the distances 
dist_sum = np.sum(dist, axis=1)

# calculate the weight of each era5 gridpoint
weight = np.zeros((ds_era5.lat.size, ds_era5.lon.size))
for i in range(ds_era5.lat.size):
    for j in range(ds_era5.lon.size):
        weight[i,j] = dist[i,j]/dist_sum[i]





        




