# Import files
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import FormatStrFormatter
from scipy.spatial import KDTree
from node import Node
from functions import compute_threshold, generate_grid, assign_block_id, compute_crime_rate, generate_map, \
    position_id_dict, normalize_position, prompt_position, astar_search

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
block_frame = generate_grid(step, min_x, max_x, min_y, max_y)
block_id = assign_block_id(block_frame, crime_df)
crime_df['block_id'] = block_id
block_frame = compute_crime_rate(block_frame, crime_df)
# Sort by crime rate
block_frame.sort_values(by='crime_rate', ascending=False, inplace=True)
_50th, _75th, _90th = compute_threshold(block_frame)
# Read in the selected input
threshold = int(input("Select the threshold to use: "))

print('\n****Runtime_Configuration*****\n')
print("Selected threshold: {}".format(threshold))
block_frame = generate_map(block_frame, threshold)
block_frame.sort_values(by='block_id', ascending=True, inplace=True)
ncols = int(round((max_x - min_x) / step, 0))
nrows = int(round((max_y - min_y) / step, 0))

# an array with linearly increasing values
obstacles = np.array(block_frame['danger'])
array = obstacles.reshape((nrows, ncols))
# We need to get our start Node and Goal Node
# (-73.59, 45.49), , ), (-73.59, 45.53)
start_pos = prompt_position(block_frame)
print('The start position is:{}'.format(start_pos))
goal_pos = prompt_position(block_frame)
print('The goal position is: {}'.format(goal_pos))
# create a map of points and block index --> to look up in the obstacle grid
map = position_id_dict(block_frame,step)
obstacles = obstacles.tolist()
print('STARTING SEARCH')
path = astar_search(map, obstacles, start_pos, goal_pos, step)
print('SEARCH OVER!')
print(path)



# DRAW GRAPH
fig, ax = plt.subplots()
# axis format
ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# define extend
extent = (min_x, max_x, min_y, max_y)
# set imshow to our array of data
# loop through path
if path is not None:
    for i in range(len(path) - 1):
        (x1, y1) = path[i]
        (x2, y2) = path[i+1]
        ax.plot([x1, x2], [y1, y2], 'r')

ax.imshow(array, interpolation='nearest', extent=extent, origin='lower')
# define ticks
x_major_ticks = np.arange(min_x, max_x + step, 0.01)
x_minor_ticks = np.arange(min_x, max_x + step, step)
y_major_ticks = (np.arange(min_y, max_y + step, 0.005))
y_minor_ticks = (np.arange(min_y, max_y + step, step))
# set ticks
ax.set_xticks(x_major_ticks)
ax.set_xticks(x_minor_ticks, minor=True)
ax.set_yticks(y_major_ticks)
ax.set_yticks(y_minor_ticks, minor=True)
plt.grid(which='minor')
plt.show()