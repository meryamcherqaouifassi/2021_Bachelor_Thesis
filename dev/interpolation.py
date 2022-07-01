#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:21:14 2022

@author: mcherqao

This script is made to try different interpolations of Era5 temperataures 
around Casablanca
"""

# imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.gofplots import qqplot
import seaborn as sns


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

ds_casa = ds_casa.sort_values(by='DATE')
ds_era5.where(ds_era5['time.hour']==12, drop=True)
ds_casa = ds_casa[np.logical_and(ds_casa.DATE.dt.year>=1979,ds_casa.DATE.dt.year<=2018)]

# make data_casa temperatures a float
ds_casa['TEMP'] = ds_casa['TEMP'].astype(np.float32)

# remove ds_casa outliers < 0 and > 45
ds_casa = ds_casa[ds_casa['TEMP'] > 0]
ds_casa = ds_casa[ds_casa['TEMP'] < 45]
ds_era5 = ds_era5.sel(time=ds_casa.DATE.to_numpy())

# set casablanca coordinates
x0 = 33.5883100                # latitude
y0 = 172.38862 #-3.80569       # longitude
coords = [x0, y0]

# split train and test sets
x, y = ds_era5.t2m, ds_casa
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# linear interpolation 
#lin_interp = ds_era5.t2m.interp({'lat':x0, 'lon':y0}, method='linear')
lin_interp = x_train.interp({'lat':x0, 'lon':y0}, method='linear')
lin_interp = pd.DataFrame(lin_interp.values)
lin_interp.columns = ['TEMP']
r2_lin = r2_score(y_train['TEMP'], lin_interp.values)
rmse_lin = mean_squared_error(y_train['TEMP'], lin_interp.values, squared=False)
mae_lin = mean_absolute_error(y_train['TEMP'], lin_interp.values)

lin_interp_test = x_test.interp({'lat':x0, 'lon':y0}, method='linear')
r2_lin_test = r2_score(y_test['TEMP'], lin_interp_test)
rmse_lin_test = mean_squared_error(y_test['TEMP'], lin_interp_test.values, squared=False)
mae_lin_test = mean_absolute_error(y_test['TEMP'], lin_interp_test.values)


# nearest interpolation
near_interp = x_train.interp({'lat':x0, 'lon':y0}, method='nearest')
near_interp = pd.DataFrame(near_interp.values)
near_interp.columns = ['TEMP']
r2_near = r2_score(y_train['TEMP'], near_interp.values)
rmse_near = mean_squared_error(y_train['TEMP'], near_interp.values, squared=False)
mae_near = mean_absolute_error(y_train['TEMP'], near_interp.values)

near_interp_test = x_test.interp({'lat':x0, 'lon':y0}, method='nearest')
r2_near_test = r2_score(y_test['TEMP'], near_interp_test)
rmse_near_test = mean_squared_error(y_test['TEMP'], near_interp_test.values, squared=False)
mae_near_test = mean_absolute_error(y_test['TEMP'], near_interp_test.values)


# quadratic interpolation
quad_interp = x_train.interp({'lat':x0, 'lon':y0}, method='quadratic')
quad_interp = pd.DataFrame(quad_interp.values)
quad_interp.columns = ['TEMP']
r2_quad = r2_score(y_train['TEMP'], quad_interp.values)
rmse_quad = mean_squared_error(y_train['TEMP'], quad_interp.values, squared=False)
mae_quad = mean_absolute_error(y_train['TEMP'], quad_interp.values)

quad_interp_test = x_test.interp({'lat':x0, 'lon':y0}, method='quadratic')
r2_quad_test = r2_score(y_test['TEMP'], quad_interp_test)
rmse_quad_test = mean_squared_error(y_test['TEMP'], quad_interp_test.values, squared=False)
mae_quad_test = mean_absolute_error(y_test['TEMP'], quad_interp_test.values)

# cubic interpolation
cub_interp = x_train.interp({'lat':x0, 'lon':y0}, method='cubic')
cub_interp = pd.DataFrame(cub_interp.values)
cub_interp.columns = ['TEMP']
r2_cub = r2_score(y_train['TEMP'], cub_interp.values)
rmse_cub = mean_squared_error(y_train['TEMP'], cub_interp.values, squared=False)
mae_cub = mean_absolute_error(y_train['TEMP'], cub_interp.values)

cub_interp_test = x_test.interp({'lat':x0, 'lon':y0}, method='cubic')
r2_cub_test = r2_score(y_test['TEMP'], cub_interp_test)
rmse_cub_test = mean_squared_error(y_test['TEMP'], cub_interp_test.values, squared=False)
mae_cub_test = mean_absolute_error(y_test['TEMP'], cub_interp_test.values)


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
# plot of ds_casa + lin_interp_2000
fig, ax = plt.subplots(dpi = 200)
ax.plot(ds_casa.loc[ds_casa.DATE.dt.year == 2000, 'DATE'],ds_casa.loc[ds_casa.DATE.dt.year == 2000, 'TEMP'], label = 'Observed')
ax.plot(ds_casa.loc[ds_casa.DATE.dt.year == 2000, 'DATE'], lin_interp_2000.t2m, label = 'Interpolated')
ax.set_xlabel('Date')
ax.set_ylabel('Temperature (°C)')
ax.legend()
plt.tight_layout()
plt.show()

# nearest-neighbor interpolation of ds_era5_2000
near_interp_2000 = ds_era5_2000.interp({'lat':x0, 'lon':y0}, method='nearest')
# plot of ds_casa + near_interp_2000
fig, ax = plt.subplots(dpi = 200)
ax.plot(ds_casa.loc[ds_casa.DATE.dt.year == 2000, 'DATE'],ds_casa.loc[ds_casa.DATE.dt.year == 2000, 'TEMP'], label = 'Observed')
ax.plot(ds_casa.loc[ds_casa.DATE.dt.year == 2000, 'DATE'], near_interp_2000.t2m, label = 'Interpolated')
ax.set_xlabel('Date')
ax.set_ylabel('Temperature (°C)')
ax.legend()
plt.tight_layout()
plt.show()

# quadratic interpolation of ds_era5_2000
quad_interp_2000 = ds_era5_2000.interp({'lat':x0, 'lon':y0}, method='quadratic')
# plot of ds_casa + quad_interp_2000
fig, ax = plt.subplots(dpi = 200)
ax.plot(ds_casa.loc[ds_casa.DATE.dt.year == 2000, 'DATE'],ds_casa.loc[ds_casa.DATE.dt.year == 2000, 'TEMP'], label = 'Observed')
ax.plot(ds_casa.loc[ds_casa.DATE.dt.year == 2000, 'DATE'], quad_interp_2000.t2m, label = 'Interpolated')
ax.set_xlabel('Date')
ax.set_ylabel('Temperature (°C)')
ax.legend()
plt.tight_layout()
plt.show()

# cubic interpolation of ds_era5_2000
cub_interp_2000 = ds_era5_2000.interp({'lat':x0, 'lon':y0}, method='cubic')
# plot of ds_casa + cub_interp_2000
fig, ax = plt.subplots(dpi = 200)
ax.plot(ds_casa.loc[ds_casa.DATE.dt.year == 2000, 'DATE'],ds_casa.loc[ds_casa.DATE.dt.year == 2000, 'TEMP'], label = 'Observed')
ax.plot(ds_casa.loc[ds_casa.DATE.dt.year == 2000, 'DATE'], cub_interp_2000.t2m, label = 'Interpolated')
ax.set_xlabel('Date')
ax.set_ylabel('Temperature (°C)')
ax.legend()
plt.tight_layout()
plt.show()

# observed vs interpolated temperatures plot
fig, ax = plt.subplots(dpi=400)
ax.scatter(ds_casa['TEMP'], lin_interp['t2m'], s=1)
ref_vals = np.linspace(5,30,100)
ax.plot(ref_vals,ref_vals, c='black')
plt.tight_layout()
ax.set_xlabel('Observation [°C]')
ax.set_ylabel('Prediction [°C]')
plt.show()

fig, ax = plt.subplots(dpi=400)
ax.scatter(ds_casa['TEMP'], near_interp['t2m'], s=1)
ref_vals = np.linspace(5,30,100)
ax.plot(ref_vals,ref_vals, c='black')
plt.tight_layout()
ax.set_xlabel('Observation [°C]')
ax.set_ylabel('Prediction [°C]')
plt.show()

fig, ax = plt.subplots(dpi=400)
ax.scatter(ds_casa['TEMP'], quad_interp['t2m'], s=1)
ref_vals = np.linspace(5,30,100)
ax.plot(ref_vals,ref_vals, c='black')
plt.tight_layout()
ax.set_xlabel('Observation [°C]')
ax.set_ylabel('Prediction [°C]')
plt.show()

fig, ax = plt.subplots(dpi=400)
ax.scatter(ds_casa['TEMP'], cub_interp['t2m'], s=1)
ref_vals = np.linspace(5,30,100)
ax.plot(ref_vals,ref_vals, c='black')
plt.tight_layout()
ax.set_xlabel('Observation [°C]')
ax.set_ylabel('Prediction [°C]')
plt.show()

# two-dimensional density plots
fig, ax = plt.subplots(dpi=600)
sns.kdeplot(data = y_train['TEMP'], multiple = 'stack', label = 'Observed')
sns.kdeplot(data = lin_interp['TEMP'], multiple = 'stack', label = 'Interpolated')
plt.tight_layout()
ax.legend()
ax.set_xlabel('Temperatures')
ax.set_ylabel('Density')
plt.show()

fig, ax = plt.subplots(dpi=600)
sns.kdeplot(data = y_train['TEMP'], multiple = 'stack', label = 'Observed')
sns.kdeplot(data = near_interp['TEMP'], multiple = 'stack', label = 'Interpolated')
plt.tight_layout()
ax.legend()
ax.set_xlabel('Temperatures')
ax.set_ylabel('Density')
plt.show()

fig, ax = plt.subplots(dpi=600)
sns.kdeplot(data = y_train['TEMP'], multiple = 'stack', label = 'Observed')
sns.kdeplot(data = quad_interp['TEMP'], multiple = 'stack', label = 'Interpolated')
plt.tight_layout()
ax.legend()
ax.set_xlabel('Temperatures')
ax.set_ylabel('Density')
plt.show()

fig, ax = plt.subplots(dpi=600)
sns.kdeplot(data = y_train['TEMP'], multiple = 'stack', label = 'Observed')
sns.kdeplot(data = cub_interp['TEMP'], multiple = 'stack', label = 'Interpolated')
plt.tight_layout()
ax.legend()
ax.set_xlabel('Temperatures')
ax.set_ylabel('Density')
plt.show()

# make a scatter of ds_casa_quantile and lin_interp_quantile
y_train = y_train.reset_index()
lin_interp = lin_interp.reset_index()
res_lin = [y_train['TEMP'][i]-lin_interp['TEMP'][i] for i in range (len(lin_interp))]
res_lin = np.array(res_lin)
fig, ax = plt.subplots(dpi=600)
qqplot(res_lin, line='r', ax=ax)
plt.tight_layout()
ax.set_xlabel('Observation quantile [°C]')
ax.set_ylabel('Prediction quantile [°C]')
plt.show()

# make a scatter of ds_casa_quantile and near_interp_quantile
near_interp = near_interp.reset_index()
res_near = [y_train['TEMP'][i]-near_interp['TEMP'][i] for i in range (len(near_interp))]
res_near = np.array(res_near)
fig, ax = plt.subplots(dpi=600)
qqplot(res_near, line='r', ax=ax)
plt.tight_layout()
ax.set_xlabel('Observation quantile [°C]')
ax.set_ylabel('Prediction quantile [°C]')
plt.show()

# make a scatter of ds_casa_quantile and quad_interp_quantile
quad_interp = quad_interp.reset_index()
res_quad = [y_train['TEMP'][i]-quad_interp['TEMP'][i] for i in range (len(quad_interp))]
res_quad = np.array(res_lin)
fig, ax = plt.subplots(dpi=600)
qqplot(res_quad, line='r', ax=ax)
plt.tight_layout()
ax.set_xlabel('Observation quantile [°C]')
ax.set_ylabel('Prediction quantile [°C]')
plt.show()

# make a scatter of ds_casa_quantile and cub_interp_quantile
cub_interp = cub_interp.reset_index()
res_cub = [y_train['TEMP'][i]-cub_interp['TEMP'][i] for i in range (len(lin_interp))]
res_cub = np.array(res_cub)
fig, ax = plt.subplots(dpi=600)
qqplot(res_cub, line='r', ax=ax)
plt.tight_layout()
ax.set_xlabel('Observation quantile [°C]')
ax.set_ylabel('Prediction quantile [°C]')
plt.show()
