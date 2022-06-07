#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 23:38:59 2022

@author: mcherqao
"""

# Imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns


# path of era5 dataset
path_era5 = '/work/FAC/FGSE/IDYST/tbeucler/default/meryam/data/era5/media/rasp/Elements/weather-benchmark/1.40625deg/2m_temperature/'
# load era5 dataset
ds_era5 = xr.open_mfdataset(path_era5+'2m_temperature_*_1.40625deg.nc')
# isolate era5 temperatures around casablanca
lon_indices = np.logical_and(ds_era5.lon>=170,ds_era5.lon<=182)
    #gives the longitudes array's indices of the area around Casablanca
lat_indices = np.logical_and(ds_era5.lat>=28,ds_era5.lat<=38)
    #gives the latitudes array's indices of the area around Casablanca
ds_era5 = ds_era5.isel({'lon':lon_indices,'lat':lat_indices})
    #isolate Casablanca's temp data
# convert era5 temperatures from Kelvin to Celsius
ds_era5['t2m'] = ds_era5['t2m'] - 273.15
ds_era5 = ds_era5.stack({'point':['lat','lon']})

# path of casa dataset
path_casa = '/work/FAC/FGSE/IDYST/tbeucler/default/meryam/2021_Bachelor_Thesis/files'
# load casa dataset
ds_casa = pd.read_csv(path_casa+'/data_casa.csv')
# keep only rows with HGHT value equal 58
ds_casa = ds_casa.loc[ds_casa['HGHT'] == 58]
# extract TEMP corresponding to a certain value of HGHT
ds_casa = ds_casa.loc[:,['DATE', 'TEMP']]
# make data_casa temperatures a float
ds_casa['TEMP'] = ds_casa['TEMP'].astype(np.float32)
# convert 'DATE' from string to datetime and put it in chronological order
ds_casa['DATE'] = pd.to_datetime(ds_casa['DATE'])
ds_casa = ds_casa.sort_values(by='DATE')
# remove ds_casa outliers < 0 and > 45
ds_casa = ds_casa[ds_casa['TEMP'] > 0]
ds_casa = ds_casa[ds_casa['TEMP'] < 45]

# reshape data to fit the LinearRegression model (nb samples, nb features)
ds_era5.where(ds_era5['time.hour']==12, drop=True)
ds_casa = ds_casa[np.logical_and(ds_casa.DATE.dt.year>=1979,ds_casa.DATE.dt.year<=2018)]
ds_era5 = ds_era5.sel(time=ds_casa.DATE.to_numpy())


# distribution of casa temperatures
fig, ax = plt.subplots(dpi=150)
sns.distplot(ds_casa['TEMP'])
ax.set_xlim(-10, 45)
plt.tight_layout()
ax.set_xlabel('Temperature [°C]')
ax.set_ylabel('Frequency')
plt.show()
plt.savefig('/work/FAC/FGSE/IDYST/tbeucler/default/meryam/2021_Bachelor_Thesis/figures')


# scatter plot of casa data
fig, ax = plt.subplots(dpi = 150)
ax.plot(ds_casa['DATE'],ds_casa['TEMP'])
plt.tight_layout()
ax.set_xlabel('Date')
ax.set_ylabel('Temperature [°C]')
plt.show()

# distribution of era5 data
fig, ax = plt.subplots(dpi = 600)
sns.distplot(ds_era5['t2m'])
plt.tight_layout()
ax.set_xlabel('Temperature [°C]')
ax.set_ylabel('Frequency')
plt.show()