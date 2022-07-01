# imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.graphics.gofplots import qqplot
from sklearn.metrics import mean_absolute_error

# path of era5 dataset
path_era5 = '/work/FAC/FGSE/IDYST/tbeucler/default/meryam/data/era5/media/rasp/Elements/weather-benchmark/1.40625deg/2m_temperature/'
# load era5 dataset
ds_era5 = xr.open_mfdataset(path_era5+'2m_temperature_*_1.40625deg.nc')
# isolate era5 temperatures around casablanca
lon_indices = np.logical_and(ds_era5.lon>=171,ds_era5.lon<=173)
    #gives the longitudes array's indices of the area around Casablanca
lat_indices = np.logical_and(ds_era5.lat>=33,ds_era5.lat<=35)
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
r2_reg = r2_score(y_test['TEMP'], predictions)
rmse_reg = mean_squared_error(y_test['TEMP'], predictions, squared=False)
abs_error_reg = mean_absolute_error(y_test['TEMP'], predictions)
print('The r2 (coefficient of determination) is: ', r2_reg)
print('The rmse (mean squared error) is: ', rmse_reg)
print('The mean absolute error is: ', abs_error_reg)

pred_train = reg.predict(x_train)
r2_reg_train = r2_score(y_train['TEMP'], pred_train)
rmse_reg_train = mean_squared_error(y_train['TEMP'], pred_train, squared=False)
mae_reg_train = mean_absolute_error(y_train['TEMP'], pred_train)

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

# qqplot of observed and predicted temperatures
y_test = y_test.reset_index()
res_reg = [y_test['TEMP'][i]-predictions[i] for i in range (len(predictions))]
res_reg = np.array(res_reg)
fig, ax = plt.subplots(dpi=600)
qqplot(res_reg, line='r', ax=ax)
plt.tight_layout()
ax.set_xlabel('Observation quantile [°C]')
ax.set_ylabel('Prediction quantile [°C]')
plt.show()

# two-dimensional density plots
fig, ax = plt.subplots(dpi=600)
sns.kdeplot(data = y_test['TEMP'], multiple = 'stack', label = 'Observed')
sns.kdeplot(data = predictions, multiple = 'stack', label = 'Interpolated')
plt.tight_layout()
ax.legend()
ax.set_xlabel('Temperatures')
ax.set_ylabel('Density')
plt.show()

# for each time step, calculate the correlation between era5 and casa temperatures
corr_list = []

for j in range(len(ds_era5.point)):
    corr_list.append(np.corrcoef(ds_era5.sel(point = ds_era5.point[j].values).t2m.values, ds_casa['TEMP']))

correlations = [item[0][1] for item in corr_list]
points = [item for item in ds_era5.point]

# put the correlation values in a dataframe
corr_df = pd.DataFrame(columns=['corr', 'points'])
corr_df["corr"] = correlations
corr_df["points"] = points
# put the correlations in descending order
corr_df = corr_df.sort_values(by='corr', ascending=False)

# create new empty dataframes
sub_x = pd.DataFrame(columns=["points", "temp"])
sub_y = np.empty((y_steps,2))
sub_reg = np.empty((x_steps,2))
sub_pred = np.empty((x_steps,2))
sub_r2 = np.empty((x_steps,2))  
# for all the points in corr_df


for i in range(2):
    point = corr_df.iloc[i].points
    sub_x["temp"].append(x.sel(point = point))
    
    for j in range(len(x)):
        sub_x["points"].append(point) 
    

for i in range(len(corr_df)):
    sub_x[i] = x[i]
    sub_y[i] = y[i]
    # if the length of sub_x is < 2
    if len(sub_x) < 2:
        sub_reg[i] = LinearRegression().fit(sub_x, sub_y['TEMP'])
        sub_pred[i] = sub_reg[i].predict(sub_x)
        sub_r2[i] = r2_score(sub_y['TEMP'], sub_pred[i])
    # if the length of sub_x is >= 2
    else:
        sub_x_train, sub_x_test, sub_y_train, sub_y_test = train_test_split(sub_x, sub_y, test_size=0.2, random_state=0)
        sub_x_train = (sub_x_train - sub_x_train.mean(axis=0)) / sub_x_train.std(axis=0)
        sub_x_test = (sub_x_test - sub_x_test.mean(axis=0)) / sub_x_test.std(axis=0)
        sub_reg[i] = LinearRegression().fit(sub_x_train, sub_y_train['TEMP'])
        sub_pred[i] = sub_reg[i].predict(sub_x_test)
        sub_r2[i] = r2_score(sub_y_test['TEMP'], sub_pred[i]) 

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
# calculate the mean of all the january, february..., december temperatures for 
# all the years and put them in a dataframe
# for all the dates in ds_casa
ds_casa_str = y_train.astype('str')
year = [datenow[0:4] for datenow in ds_casa_str['DATE']]
year = np.array(year)
month = [d[5:7] for d in ds_casa_str['DATE']]
month = np.array(month)
day = [datenow[8:10] for datenow in ds_casa_str['DATE']]
day = np.array(day)
        
train_means_months = np.zeros(12)
index = 0
for month_now in np.unique(month):
    train_means_months[index] = y_train[(month == month_now)].mean()
    index = index + 1
    
# calculate the predicted temperatures as the average of the month it belongs to
# for all the dates in y_test
predictions_months = np.zeros(len(y_test))
for i in range(len(y_test)):
    predictions_months[i] = train_means_months[int(month[i]) - 1]

# calculate the r2, rmse and absolute mean error of the monthly baseline and the predictions
r2_months = r2_score(y_test['TEMP'], predictions_months)
print('r2_months: ' + str(r2_months))
rmse_months = np.sqrt(mean_squared_error(y_test['TEMP'], predictions_months))
print('rmse_months: ' + str(rmse_months))
abs_mean_error_months = np.mean(np.abs(y_test['TEMP'] - predictions_months))
print('abs_mean_error_months: ' + str(abs_mean_error_months))

# reset y_test indexes 
y_test = y_test.reset_index()
# calculate the residuals of the monthly baseline and the predictions
residuals_months = [y_test['TEMP'][i]-predictions_months[i] for i in range(len(predictions_months))]
# put them in a dataframe
residuals_months = pd.DataFrame(residuals_months)
# summary statistics of the residuals
print(residuals_months.describe())
# plot residuals
fig, ax = plt.subplots(dpi=600)
residuals_months.plot()
plt.show()
# histogram plot
fig, ax = plt.subplots(dpi=600)
sns.distplot(residuals_months)
ax.set_ylabel('Frequency')
plt.show()
# Plot Q-Q plot
residuals_months = [y_test['TEMP'][i]-predictions_months[i] for i in range(len(predictions_months))]
residuals_months = np.array(residuals_months)
fig, ax = plt.subplots(dpi=600)
qqplot(residuals_months, line='r', ax=ax)
plt.show()



# calculate the daily climatological baseline of casa temperatures
# calculate the mean of all the 01/01, 02/01, ..., 31/01, 01/02, 02/02, ..., 31/02, 01/03, ..., 31/12 temperatures for all the years
# and put them in a dataframe
means_days = np.zeros(372)
index = 0
for month_now in np.unique(month):
    for day_now in np.unique(day):
        means_days[index] = y_train[(month == month_now) & (day == day_now)].mean()
        index = index + 1
        
# calculate the predicted temperatures as the average of the day it belongs to
# for all the dates in y_test
predictions_days = np.zeros(len(y_test))
for i in range(len(y_test)):
    predictions_days[i] = means_days[int(month[i]) - 1]
    
    
# calculate the r2, rmse and absolute mean error of the monthly baseline and the predictions
r2_days = r2_score(y_test['TEMP'], predictions_days)
print('r2_days: ' + str(r2_days))
rmse_days = np.sqrt(mean_squared_error(y_test['TEMP'], predictions_days))
print('rmse_days: ' + str(rmse_days))
abs_mean_error_days = np.mean(np.abs(y_test['TEMP'] - predictions_days))
print('abs_mean_error_days: ' + str(abs_mean_error_days))

# calculate the residuals of the monthly baseline and the predictions
residuals_days = [y_test['TEMP'][i]-predictions_days[i] for i in range(len(predictions_days))]
# put them in a dataframe
residuals_days = pd.DataFrame(residuals_days)
# summary statistics of the residuals
print(residuals_days.describe())
# plot residuals
fig, ax = plt.subplots(dpi=600)
residuals_days.plot()
plt.show()
# histogram plot
fig, ax = plt.subplots(dpi=600)
sns.distplot(residuals_days)
ax.set_ylabel('Frequency')
plt.show()
# Plot Q-Q plot
residuals_days = [y_test['TEMP'][i]-predictions_days[i] for i in range(len(predictions_days))]
residuals_days = np.array(residuals_days)
fig, ax = plt.subplots(dpi=600)
qqplot(residuals_days, line='r', ax=ax)
plt.show()



# calculate the persistence baseline of casa temperatures
# today's temperature is tomorrow's temperature
# Create a lag feature
var = pd.DataFrame(ds_casa['TEMP'])
dataframe = pd.concat([var.shift(1), var], axis=1)
dataframe.columns = ['t', 't+1']
print(dataframe.head(5))

# Split series into train and test sets
X = dataframe.values
train_size = int(len(X) * 0.7)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]

# Create a baseline model (Naive model)
def model_persistence(x):
  return x

# walk-forward validation
predicted = list()
for X in test_X:
  yhat = model_persistence(X)
  predicted.append(yhat)
rmse_pers = np.sqrt(mean_squared_error(test_y, predicted))
r2_pers = r2_score(test_y, predicted)
mae_pers = mean_absolute_error(test_y, predicted)
print('Test RMSE: %.3f' % rmse_pers)
print('Test R2: %.3f' % r2_pers)
print('Test MAE: %.3f' % mae_pers)

# naive forecast
predictions = [X for X in test_X]
# calculate residuals
residuals = [test_y[i]-predictions[i] for i in range(len(predictions))]
residuals = pd.DataFrame(residuals)
print(residuals.head())

# summary statistics
print(residuals.describe())

# plot residuals
fig, ax = plt.subplots(dpi=400)
residuals.plot()
plt.show()
# histogram plot
fig, ax = plt.subplots(dpi=600)
sns.distplot(residuals)
ax.set_ylabel('Frequency')
plt.show()
# Plot Q-Q plot
residuals = [test_y[i]-predictions[i] for i in range(len(predictions))]
residuals = np.array(residuals)
fig, ax = plt.subplots(dpi=600)
qqplot(residuals, line='r', ax=ax)
plt.show()




# scatter of the r2 as a function of the number of inputs
xaxis = np.zeros(12898)
for i in range(1,12899,1):
    xaxis[i-1]=i

 
fig, ax = plt.subplots(dpi = 300)
ax.scatter(xaxis, r2_reg*xaxis)
# add a y=1 line
ax.plot(xaxis, [1]*xaxis, 'k--')
# add y = r2_pers line
ax.plot([r2_pers]*xaxis, 0*xaxis, 'g--')
# add a y = r2_monthly line
#ax.plot(x_steps, [r2_months]*len(x_steps), 'r--')
# add a y = r2_daily line
#ax.plot(x_steps, [r2_days]*len(x_steps), 'b--')
ax.set_xlabel('Number of inputs')
ax.set_ylabel('R2')
plt.tight_layout()
plt.show()

