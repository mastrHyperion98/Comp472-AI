# Import files
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import FormatStrFormatter
from functions import compute_threshold,generate_grid, assign_block_id, compute_crime_rate, generate_map

step = 0.002
# imports as a pandas dataframe
crime_df = gpd.read_file('data/crime_dt.shp')
points = crime_df.centroid
# get our x and y coordinates isolated
x_coords = np.array(points.x)
y_coords = np.array(points.y)
# compute our maximum and minimum x and y coordinates
min_x = x_coords.min()
max_x = x_coords.max()
min_y = y_coords.min()
max_y = y_coords.max()
# draw our map
"""""
map = crime_df.plot(color='yellow', marker="s")
map.set_xticks(np.arange(min_x, max_x+0.002, 0.002))
map.set_yticks(np.arange(min_y, max_y+0.002, 0.002))
map.set_xticklabels(np.arange(min_x, max_x+0.002, 0.002),rotation=270)
map.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
map.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
map.set_facecolor('xkcd:purple')
plt.show()
"""""
block_frame = generate_grid(step, min_x, max_x, min_y, max_y)
block_id = assign_block_id(block_frame, crime_df)
crime_df['block_id'] = block_id
block_frame = compute_crime_rate(block_frame, crime_df)
# Sort by crime rate
block_frame.sort_values(by='crime_rate', ascending=False, inplace=True)
print(block_frame)
_50th, _75th, _90th = compute_threshold(block_frame)
# Read in the selected input
#threshold = int(input("Select the threshold to use: 50, 75 or 90 (default is 50): "))

#if threshold != 75:
 #   if threshold != 90:
  #      threshold = 50

#print("Selected threshold: {}".format(threshold))

generate_map(block_frame, 50, min_x, max_x, min_y, max_y)

ncols = int((max_x-min_x)/step)
nrows = int((max_y-min_y)/step)
# an array with linearly increasing values
array = np.zeros(nrows*ncols)
array = array.reshape((nrows, ncols))
array[0][10] = 1
array[1][10] = 1
fig, ax = plt.subplots()
# axis format
ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# define extend
extent=(min_x,max_x,min_y,max_y)
# set imshow to our array of data
ax.imshow(array, interpolation='nearest', extent=extent, origin='lower')
# define ticks
x_major_ticks = np.arange(min_x, max_x+0.002, 0.01)
x_minor_ticks = np.arange(min_x, max_x+0.002, 0.002)
y_major_ticks = (np.arange(min_y, max_y+0.002, 0.005))
y_minor_ticks = (np.arange(min_y, max_y+0.002, 0.002))
# set ticks
ax.set_xticks(x_major_ticks)
ax.set_xticks(x_minor_ticks, minor=True)
ax.set_yticks(y_major_ticks)
ax.set_yticks(y_minor_ticks, minor=True)
plt.show()