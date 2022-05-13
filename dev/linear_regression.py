# Imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, r2_score


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

# make data_casa temperatures a float
ds_casa['TEMP'] = ds_casa['TEMP'].astype(np.float32)

# remove ds_casa outliers < 0 and > 45
ds_casa = ds_casa[ds_casa['TEMP'] > 0]
ds_casa = ds_casa[ds_casa['TEMP'] < 45]

# scatter plot of casa data
ds_casa.plot('DATE','TEMP')

# distribution of era5 data
fig, ax = plt.subplots()
sns.distplot(ds_era5['t2m'])

# calculate the number of time steps
x_steps = len(ds_era5.time)
y_steps = len(ds_casa.loc[:,["DATE"]])

# reshape data to fit the LinearRegression model (nb samples, nb features)
ds_era5.where(ds_era5['time.hour']==12, drop=True)
ds_casa = ds_casa[np.logical_and(ds_casa.DATE.dt.year>=1979,ds_casa.DATE.dt.year<=2018)]
ds_era5 = ds_era5.sel(time=ds_casa.DATE.to_numpy())
ds_era5 = ds_era5.stack({'point':['lat','lon']})

# plot of temperatures of era5 and casa
fig, ax = plt.subplots()
ax.plot(ds_era5['t2m'], ds_casa['TEMP'], '.')
ax.set_xlabel('temperature (°C)')
ax.set_ylabel('temperature (°C)')
plt.show()

# separating training and test data
x, y = ds_era5.t2m.values, ds_casa['TEMP']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, 
                                                    random_state=0)

# normalization by mean and std
#x_train = (x_train - x_train.mean()) / x_train.std()
#x_test = (x_test - x_test.mean()) / x_test.std()
#y_train = (y_train - y_train.mean()) / y_train.std()
#y_test = (y_test - y_test.mean()) / y_test.std()

# linear regression
reg = LinearRegression().fit(x_train, y_train)
reg.predict(x_test)
print(reg.coef_)
print(reg.intercept_)

# cross-validation 
cv_results = cross_validate(reg, x, y, cv=5, scoring = 'neg_mean_squared_error')

# Running Evaluation Metrics
predictions = reg.predict(x_test)
r2 = r2_score(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)
print('The r2 is: ', r2)
print('The rmse is: ', rmse)

# plot of temperatures of era5 and casa with the regression line
fig, ax = plt.subplots()
ax.plot(x_test, y_test, '.')
ax.plot(x_test, predictions, 'r-')
ax.set_xlabel('temperature (°C)')
ax.set_ylabel('temperature (°C)')
plt.show()

# calculate the climatological baseline
y_mean = y_train.mean()




