# Imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns


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
lon_indices = np.logical_and(ds_era5.lon>=175,ds_era5.lon<=182)
    #gives the longitudes array's indices of the area around Casablanca
lat_indices = np.logical_and(ds_era5.lat>=28,ds_era5.lat<=38)
    #gives the latitudes array's indices of the area around Casablanca
ds_era5 = ds_era5.isel({'lon':lon_indices,'lat':lat_indices})
    #isolate Casablanca's temp data

# convert era5 temperatures from Kelvin to Celsius
ds_era5['t2m'] = ds_era5['t2m'] - 273.15

# convert 'DATE' from string to datetime and put it in chronological order
ds_casa['DATE'] = pd.to_datetime(ds_casa['DATE'])
ds_casa = ds_casa.sort_values(by='DATE')

# distribution of casa temperatures
fig, ax = plt.subplots()
sns.distplot(ds_casa['TEMP'])
ax.set_xlim(-10, 45)
plt.show()

# remove ds_casa outliers < 0 and > 45
ds_casa = ds_casa[ds_casa['TEMP'] > 0]
ds_casa = ds_casa[ds_casa['TEMP'] < 45]

# scatter plot of casa data
ds_casa['TEMP'] = ds_casa['TEMP'].astype(np.float32)
ds_casa.plot('DATE','TEMP')

# distribution of era5 data
fig, ax = plt.subplots()
sns.distplot(ds_era5['t2m'])

# plot of temperatures of era5 and casa
fig, ax = plt.subplots()
ax.plot(ds_era5['t2m'], ds_casa['TEMP'], '.')
ax.set_xlabel('temperature (°C)')
ax.set_ylabel('temperature (°C)')
plt.show()

#calculate the number of time steps
x_steps = len(ds_era5.time)
y_steps = len(ds_casa.loc[:,["DATE"]])

# crush era5 longitude and latitude into a single feature
ds_era5 = ds_era5.assign_coords(lon=ds_era5.lon.to_series().values,
                                lat=ds_era5.lat.to_series().values)

# reshape data to fit the LinearRegression model (nb samples, nb features)
X = ds_era5.t2m.values.reshape(x_steps,2)
Y = ds_casa.TEMP.values.reshape(y_steps,1)

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


