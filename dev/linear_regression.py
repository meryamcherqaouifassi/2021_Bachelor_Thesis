# Imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, r2_score
import cartopy.crs as ccrs


# path of era5 dataset
path_era5 = '/work/FAC/FGSE/IDYST/tbeucler/default/meryam/data/era5/media/rasp/Elements/weather-benchmark/1.40625deg/2m_temperature/'
# load era5 dataset
ds_era5 = xr.open_mfdataset(path_era5+'2m_temperature_*_1.40625deg.nc')
# isolate era5 temperatures around casablanca
lon_indices = np.logical_and(ds_era5.lon>=170,ds_era5.lon<=175)
    #gives the longitudes array's indices of the area around Casablanca
lat_indices = np.logical_and(ds_era5.lat>=30,ds_era5.lat<=35)
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
# make data_casa temperatures a float
ds_casa['TEMP'] = ds_casa['TEMP'].astype(np.float32)


# calculate the number of time steps
x_steps = len(ds_era5.time)
y_steps = len(ds_casa.loc[:,["DATE"]])

# separating training and test data
x, y = ds_era5.t2m, ds_casa
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# normalization by mean and std
x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)
x_test = (x_test - x_test.mean(axis=0)) / x_test.std(axis=0)
#y_train = (y_train - y_train.mean(axis=0)) / y_train.std(axis=0)
#y_test = (y_test - y_test.mean(axis=0)) / y_test.std(axis=0)

# linear regression
reg = LinearRegression().fit(x_train, y_train['TEMP'])
reg.predict(x_test)
print(reg.coef_)
print(reg.intercept_)
a = reg.coef_
b = reg.intercept_ 

# cross-validation 
cv_results = cross_validate(reg, x, y['TEMP'], cv=5, scoring = 'neg_mean_squared_error')

# Running Evaluation Metrics
predictions = reg.predict(x_test)
r2 = r2_score(y_test['TEMP'], predictions)
rmse = mean_squared_error(y_test['TEMP'], predictions, squared=False)
print('The r2 is: ', r2)
print('The rmse is: ', rmse)

# keep from ds_casa only the dates of the test data
#ds_casa = ds_casa.loc[ds_casa.DATE.isin(y_test.index)]
# plot of the ds_casa temperatures in time, and the predicted temperatures
fig, ax = plt.subplots(dpi = 300)
ax.plot(y_test['DATE'], y_test['TEMP'], '.')
ax.scatter(predictions, color='red', label = 'linear regression')
ax.set_xlabel('Date')
ax.set_ylabel('Temperature (°C)')
ax.legend()
plt.tight_layout()
plt.show()

# plot of observed vs predicted temperatures, with x=y line
fig, ax = plt.subplots(dpi=300)
ax.scatter(y_test['TEMP'], predictions, s=1,)
ref_vals = np.linspace(10,30,100)
ax.plot(ref_vals,ref_vals, c='black', label = 'x=y')
ax.set_xlabel('Observed temperature (°C)')
ax.set_ylabel('Predicted temperature (°C)')
ax.legend()
plt.tight_layout()
plt.show()

# visualisation 
#central_lon, central_lat = 33.5731104, -7.5898434   #coordinates of Casablanca
#lat_min = 28
#lat_max = 35
#lon_min = -11
#lon_max = -4
#extent = [lon_min, lon_max, lat_min, lat_max]
#time_test = 2000
#norm = mpl.colors.Normalize(vmin=283.15, vmax=303.15) 
#fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_lon)})
#ax.set_extent(extent)
#ax.coastlines(resolution='50m')
#ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
#ax.set_title('Predicted temperature for ' + str(time_test) + '-03-18')
#ax.set_xlabel('longitude')
#ax.set_ylabel('latitude')
#ax.set_extent([lon_min, lon_max, lat_min, lat_max])
#map = ds.sel(time='2000-03-18')['t2m'].plot.imshow(cmap='coolwarm', norm=norm)
#ax.scatter(ds_era5.lon, ds_era5.lat, c=ds_era5.sel(time='2000-03-18'),
#              cmap='RdYlBu_r', norm=norm, s=0.5, transform=ccrs.PlateCarree())
#plt.show()
#map = y_test.sel(time=time_test)['t2m'].plot.imshow(cmap='coolwarm', norm=norm)



# calculate the monthly climatological baseline of casa temperatures
# calculate the mean of all the january, february..., december temperatures for all the years
# and put them in a dataframe
ds_casa_monthly = ds_casa.resample('M').mean()
# calculate the r2 of the monthly climatological baseline
r2_monthly = r2_score(ds_casa_monthly['TEMP'], ds_casa_monthly['TEMP'].mean())

# calculate the weekly climatological baseline of casa temperatures
# calculate the mean of all the first, second, ..., 52nd weeks temperatures for all the years
# and put them in a dataframe
ds_casa_weekly = ds_casa.resample('W').mean()
# calculate the r2 of the weekly climatological baseline
r2_weekly = r2_score(ds_casa_weekly['TEMP'], ds_casa_weekly['TEMP'].mean())

# calculate the daily climatological baseline of casa temperatures
# calculate the mean of all the 01/01, 02/01, ..., 31/01, 01/02, 02/02, ..., 31/02, 01/03, ..., 31/12 temperatures for all the years
# and put them in a dataframe
ds_casa_daily = ds_casa.resample('D').mean()
# calculate the r2 of the daily climatological baseline
r2_daily = r2_score(ds_casa_daily['TEMP'], ds_casa_daily['TEMP'].mean())

# calculate the persistence baseline of casa temperatures
# calculate the mean of all the temperatures for all the years
# and put them in a dataframe
ds_casa_persistence = ds_casa.resample('A').mean()
# calculate the r2 of the persistence baseline
r2_persistence = r2_score(ds_casa_persistence['TEMP'], ds_casa_persistence['TEMP'].mean())

# scatter of the r2 as a function of the number of inputs 
fig, ax = plt.subplots(dpi = 300)
ax.scatter(x_steps, r2)
# add a y=1 line
ax.plot(x_steps, [1]*len(x_steps), 'k--')
# add a y = r2_monthly line
ax.plot(x_steps, [r2_monthly]*len(x_steps), 'r--')
# add a y = r2_weekly line
ax.plot(x_steps, [r2_weekly]*len(x_steps), 'g--')
# add a y = r2_daily line
ax.plot(x_steps, [r2_daily]*len(x_steps), 'b--')
# add a y = r2_persistence line
ax.plot(x_steps, [r2_persistence]*len(x_steps), 'y--')
ax.set_xlabel('Number of inputs')
ax.set_ylabel('R2')
plt.tight_layout()
plt.show()

