# Imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# path of era5 dataset
path_era5 = '/work/FAC/FGSE/IDYST/tbeucler/default/meryam/data/era5/media/rasp/Elements/weather-benchmark/1.40625deg/2m_temperature/'
# load era5 dataset
ds_era5 = xr.open_mfdataset(path_era5+'2m_temperature_*_1.40625deg.nc')

# path of casa dataset
path_casa = '/work/FAC/FGSE/IDYST/tbeucler/default/meryam/2021_Bachelor_Thesis/files'
# load casa dataset
ds_casa = pd.read_csv(path_casa+'/data_casa.csv')
# extract TEMP corresponding to a certain value of HGHT
ds_casa = ds_casa.loc[:,['DATE', 'TEMP']].loc[ds_casa['HGHT'] == 58]

# isolate era5 temperatures around casablanca
lon_indices = np.logical_and(ds_era5.lon>=175,ds_era5.lon<=182)
    #gives the longitudes array's indices of the area around Casablanca
lat_indices = np.logical_and(ds_era5.lat>=28,ds_era5.lat<=38)
    #gives the latitudes array's indices of the area around Casablanca
ds_era5 = ds_era5.isel({'lon':lon_indices,'lat':lat_indices})
    #isolate Casablanca's temp data

#calculate the number of time steps
x_steps = len(ds_era5.DATE)
y_steps = len(ds_casa.DATE)

# reshape 
ds_era5.shape = (x_steps,4)
ds_casa.shape = (y_steps,)

# separating training and test data
x_train = ds_era5.sel(time=slice('1979', '2010'))
x_test = ds_era5.sel(time=slice('2011', '2018'))
y_train = ds_casa.sel(time=slice('1973', '2010'))
y_test = ds_casa.sel(time=slice('2011', '2021'))

# normalization by mean and std
x_train = (x_train - x_train.mean()) / x_train.std()
x_test = (x_test - x_test.mean()) / x_test.std()
y_train = (y_train - y_train.mean()) / y_train.std()
y_test = (y_test - y_test.mean()) / y_test.std()

# linear regression
reg = LinearRegression().fit(x_train, y_train)
reg.predict(x_test)

# plot of the linear regression
plt.scatter(x_train, y_train, color='blue')

# calculate the climatological baseline
y_mean = y_train.mean()


