#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 17:36:48 2022

@author: mcherqao
"""
# Imports 
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# Geographic coordinates 
central_lon, central_lat = -7.5898434, 33.5731104  #coordinates of Casablanca
lat_min = 20
lat_max = 40
lon_min = -20
lon_max = 0
extent = [lon_min, lon_max, lat_min, lat_max]

# Visualise data 
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines(resolution='50m')
ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
ax.set_extent([lon_min, lon_max, lat_min, lat_max])
ax.plot(central_lon, central_lat, marker='.', markersize=50, color='red')
ax.set_title('Casablanca')
plt.show()
