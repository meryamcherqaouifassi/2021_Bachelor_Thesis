#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:32:08 2022

@author: mcherqao
"""

# Imports 
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm_notebook as tqdm

# Load era5 data
path_era5 = '/work/FAC/FGSE/IDYST/tbeucler/default/meryam/data/era5/media/rasp/Elements/weather-benchmark/1.40625deg/2m_temperature'
t2m = xr.open_mfdataset(path_era5+'2m_temperature/*.nc')

# Load casablanca data
path_datacasa = '/work/FAC/FGSE/IDYST/tbeucler/default/meryam/data/data_casa'
t2m_casa = xr.open_dataset(path_datacasa+'Casablanca/*.nc').temp

# Isolate temperatures around Casablanca
lon_indices = np.logical_and(t2m.lon>=175,t2m.lon<=182)
    #gives the longitudes array's indices of the area around Casablanca
lat_indices = np.logical_and(t2m.lat>=28,t2m.lat<=38)
    #gives the latitudes array's indices of the area around Casablanca
t2m = t2m.isel({'lon':lon_indices,'lat':lat_indices})
    #isolate Casablanca's temp data
    
# Reshape
x.shape = (,4)
y.shape = (,)

# Training and test datasets
x_train = t2m.sel(time=slice('1979', '2010'))
x_test = t2m.sel(time=slice('2011', '2018'))
y_train = t2m_casa.sel(time=slice('1973', '2010'))
y_test = t2m_casa.sel(time=slice('2011', '2021'))

# Normalization 
norm_x_train = (x_train-mean(x_train))/std(x_train)
norm_x_test = (x_test-mean(x_test))/std(x_test)
norm_y_train = (y_train-mean(y_train))/std(y_train)
norm_y_test = (y_test-mean(y_test))/std(y_test)

# Linear regression 
lr = LinearRegression()
lr.fit(x_train, y_train)
lr.pred.ct(x_test)



