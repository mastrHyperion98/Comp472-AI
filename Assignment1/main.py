# Import files
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors
from functions import compute_threshold,generate_grid, assign_block_id, compute_crime_rate

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
compute_threshold(block_frame)
# Now we want to define to create a dataframe that contains the crime rate for every element in our blocks

