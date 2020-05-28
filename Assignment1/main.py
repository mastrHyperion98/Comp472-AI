# -------------------------------------------------------
# Assignment 1
# Written by Steven Smith 40057065
# For COMP 472 Section KX - Summer 2020
# -------------------------------------------------------
# Import files
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from math import ceil
from matplotlib.ticker import FormatStrFormatter
from functions import description, generate_grid, assign_block_id, compute_crime_rate, generate_map, \
    position_id_dict, prompt_position, astar_search

# This the main execution file. The function calls are in the function file

# Get initial setup data
print('\n**** Initial Setup ****\n')
threshold = int(input("Select the threshold to use: "))
print("Selected threshold: {}".format(threshold))
step = float(input("Select the step size to use: "))
print("Selected stepping is: {}".format(step))

# imports as a pandas dataframe
print('\n**** Extracting Data ***\n')
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
print('\n**** Generating Map and Acquiring Required Data ****\n')

# create a dataframe containing our block information
block_frame = generate_grid(step, min_x, max_x, min_y, max_y)
# Generate a Serries of block id
block_id = assign_block_id(block_frame, crime_df)
crime_df['block_id'] = block_id
block_frame = compute_crime_rate(block_frame, crime_df)
# Sort by crime rate
block_frame.sort_values(by='crime_rate', ascending=False, inplace=True)
description(block_frame)
# Read in the selected input
# generate our map and obstacles
block_frame = generate_map(block_frame, threshold)
block_frame.sort_values(by='block_id', ascending=True, inplace=True)
ncols = ceil((max_x - min_x) / step)
nrows = ceil((max_y - min_y) / step)
# an array with linearly increasing values
obstacles = np.array(block_frame['danger'])
array = obstacles.reshape((nrows, ncols))
# We need to get our start Node and Goal Node
# (-73.59, 45.49), , ), (-73.59, 45.53)
# create a map of points and block index --> to look up in the obstacle grid
position_map = position_id_dict(block_frame,step)
obstacles = obstacles.tolist()

# Print out the block_frame
print(block_frame)

print('\n**** Select Start and Goal Positions ****\n')
start_pos = prompt_position(block_frame, True)
print('The start position is:{}'.format(start_pos))
goal_pos = prompt_position(block_frame, False)
print('The goal position is: {}'.format(goal_pos))

# use the astar_search to find the ideal path
print('\n**** Searching for Path ****\n')
path = astar_search(position_map, obstacles, start_pos, goal_pos, step)
print('Path is: {}'.format(path))
print('\n**** Generating Graphics! ****\n')
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
ax.set_title('District Map with Steps of {} and a threshold of {}.'.format(step,threshold))
plt.show()

print('Program has terminated successfully!')