# -------------------------------------------------------
# Assignment 1
# Written by Steven Smith 40057065
# For COMP 472 Section KX - Summer 2020
# -------------------------------------------------------

# This file contains functions and classes necessary to the computation of the
# data and search

import numpy as np
import pandas as pd
from math import ceil
import time
# Node class for our Tree
# This class represents a node for our list


class Node:
    def __init__(self, position:(), parent:()):
        #position is a tuple
        self.position = position
        #parent is another node
        self.parent = parent
        self.g = 0 # Distance to start node
        self.h = 0 # heuristic cost
        self.f = 0 # Total cost

    # Compare nodes
    def __eq__(self, other):
        return self.position == other.position

    # Sort nodes by cost
    def __lt__(self, other):
         return self.f < other.f

    # Print node
    def __repr__(self):
        return '({0},{1})'.format(self.position, self.f)


# This function creates a data frame that is composed of all the blocks defined by our stepping function and its
# respective id
def generate_grid(step, min_x, max_x, min_y, max_y):
    block_id = []
    id_counter = 0
    lower_x = []
    lower_y = []
    upper_x = []
    upper_y = []

    ncols = ceil((max_x - min_x) / step)
    nrows = ceil((max_y - min_y) / step)

    y = min_y - step
    for j in range(nrows):
        y = y + step
        x = min_x
        for i in range(ncols):
            lower_x.append(round(x, 3))
            lower_y.append(round(y, 3))
            upper_x.append(round(x + step, 3))
            upper_y.append(round(y + step, 3))
            x = x + step
            block_id.append(id_counter)
            id_counter = id_counter + 1

    dataframe = pd.DataFrame(lower_x, columns=['lower_x'])
    dataframe['lower_y'] = lower_y
    dataframe['upper_x'] = upper_x
    dataframe['upper_y'] = upper_y
    dataframe['block_id'] = block_id
    # then we can compute the crime_rate and add that to our block
    return dataframe


def assign_block_id(block_frame, crime_df):
    # traverse the points one by on

    block_id = []

    for point in crime_df['geometry']:
        id = block_frame[(block_frame.lower_x <= point.x) & (block_frame.upper_x > point.x) &
                         (block_frame.lower_y <= point.y) & (block_frame.upper_y > point.y)]['block_id'].iloc[0]
        block_id.append(id)

    # return our block_id list
    return block_id


def compute_crime_rate(block_frame, crime_df):
    crime_count_per_block = pd.DataFrame(crime_df['block_id'].value_counts().reset_index())
    crime_count_per_block.columns = ['block_id', 'crime_rate']
    df = pd.merge(block_frame, crime_count_per_block, how='left', on='block_id')
    df['crime_rate'].fillna(0, inplace=True)
    return df


def description(sorted):
    _50th = sorted['crime_rate'].quantile(.5)
    _75th = sorted['crime_rate'].quantile(.75)
    _90th = sorted['crime_rate'].quantile(.90)
    sum = sorted['crime_rate'].sum()
    print("Crime Rate mean: {}".format(sorted['crime_rate'].mean()))
    print("Crime Rate median: {}".format(sorted['crime_rate'].median()))
    print("Crime Rate standard deviation: {}".format(sorted['crime_rate'].std()))
    print('50th quartile: {}'.format(_50th))
    print('75th quartile: {}'.format(_75th))
    print('90th quartile: {}'.format(_90th))
    print('\n**** Dataframe content ****\n')

# generate danger index
def generate_map(block_frame, threshold):
    count = block_frame.shape[0]
    cutoff = block_frame['crime_rate'].quantile(threshold/100.0)
    is_crime_high = pd.Series(np.zeros(count))
    block_frame = block_frame.reset_index()
    sub_frame = block_frame[(block_frame.crime_rate >= cutoff)]
    # i will have to modify that to include duplicates
    for index in range(len(sub_frame)):
        is_crime_high[index] = 1

    block_frame['danger'] = is_crime_high
    return block_frame


def position_id_dict(block_frame, step):
    dict = {}
    for index, block in block_frame.iterrows():
        low_x = block['lower_x']
        low_y = block['lower_y']
        upper_x = block['upper_x']
        upper_y = block['upper_y']
        id = block['block_id']

        positions = [(low_x,low_y),(low_x,upper_y),(upper_x,low_y),(upper_x,upper_y)]
        for position in positions:
            # check if position in dic
            if position in dict:
                dict[position].append(id)
            else:
                dict[position] = [id]
        # should return a dictionary list of all valid positions and its block id --> we talking several thousands points
    return dict


# normalizes a position in a given block to the lower left position of the block
def normalize_position(block_frame, x, y):
    frame = block_frame[(block_frame.lower_x <= x) & (block_frame.upper_x > x) &
                        (block_frame.lower_y <= y) & (block_frame.upper_y > y)]
    if len(frame) > 0:
        x = frame.lower_x.iloc[0]
        y = frame.lower_y.iloc[0]
    return (x, y)


def prompt_position(block_frame, start):
    invalid_input = True
    while invalid_input:
        if start: message = 'start'
        else: message = 'goal'
        start_pos = tuple(
            float(x.strip()) for x in input('Enter a {} position(ex: -73.589, 45.490) : '.format(message)).split(','))
        if (-73.59 <= start_pos[0] <= -73.55
                and 45.53 >= start_pos[1] >= 45.49):
            # normalize position to lower left if not an edge
            start_pos = normalize_position(block_frame, start_pos[0], start_pos[1])
            invalid_input = False
        else:
            print('Invalid input! Try again!')
    return start_pos


def astar_search(position_map, obstacles, start, end, step):
    # Create lists for open nodes and closed nodes
    open = []
    closed = []

    # We need to create our start and goal node
    start_node = Node(start, None)
    goal_node = Node(end, None)

    # We want to add our start/root node to the list of open nodes
    open.append(start_node)
    start_time = time.process_time()
    # We need to loop through our list until all open nodes are closed
    while len(open) > 0:

        if time.process_time() - start_time >= 10:
            print('Time is up. The optimal path is not found.')
            return None

        # Sort the open list to get the node with the lowest cost first
        # Our nodes are sorted by their cost and we want to find the lowest cost path to reach the goal
        open.sort()

        # We want to get the first element in the list. The node with the lowest cost
        current_node = open.pop(0)

        # Add the current node to the closed list
        # Since we are visiting it
        closed.append(current_node)

        # Check if we have reached the goal, return the path
        if current_node == goal_node:
            path = []
            while current_node != start_node:
                path.append(current_node.position)
                current_node = current_node.parent
            # path.append(start)
            # Return reversed path
            if not path:
                message = '“Due to blocks, no path is found. Please change the map and try again”'
                print(message)
                return None
            print('Time Required for search: {}'.format(time.process_time() - start_time))
            path.append(start_node.position)
            return path[::-1]

        # unzip the position tuple
        (x, y) = current_node.position

        # Get neighbors
        # These are the connecting nodes. We also have access to diagonals.
        neighbors = [(round(x - step, 3), y), (round(x - step, 3), round(y - step, 3)), (x, round(y - step, 3))
            , (round(x + step, 3), round(y - step, 3)), (round(x + step, 3), y),
                     (round(x + step, 3), round(y + step, 3)), (x, round(y + step, 3)),
                     (round(x - step, 3), round(y + step, 3))]

        # Neighbours denote all the available positions to which we can move. They will be checked for validity below

        # Loop neighbors and check validation conditions and assign weight
        for next in neighbors:

            # If the position is out of bounds skip it -- invalid position
            # Condition 1
            if next not in position_map:
                continue

            # Condition 2: Position not on boundary... namely  -73.590 < x < -73.550
            # 45.490 < x < 45.530
            # if True skip boundary position
            if -73.590 == next[0] or -73.550 == next[0] or 45.490 == next[1] or 45.530 == next[1]:
                if next != goal_node.position:
                    continue

            # list of block indices for our position
            id_list = position_map[next]
            # Create a neighbor node
            neighbor = Node(next, current_node)
            node_cost = 0
            # Check if the neighbor is in the closed list
            if neighbor in closed:
                continue
            # Condition 3: Is next move a diagonal
            # Is a diagonal if the list intersection length is 1
            ids_current = set(position_map[current_node.position])
            id_common = list(ids_current.intersection(id_list))
            if len(id_common) == 1:
                id_neigh = id_common[0]
                # id is a block cannot move diagonally across
                if obstacles[int(id_neigh)] == 1:
                    continue
                else:
                    node_cost = 1.5
            # if not diagonal than do following
            else:
                # Condition 4: Evaluate if the path taken touches a block area
                # Essentially the edge is the two blocks that the points shares with the other edge
                num_obstacles = 0
                for id in id_common:
                    if obstacles[int(id)] == 1:
                        num_obstacles = num_obstacles + 1
                # if both are obstacles we cannot traverse
                if num_obstacles == 2:
                    continue
                elif num_obstacles == 1:
                    node_cost = 1.3
                else:
                    node_cost = 1
            # Generate heuristics (Manhattan distance) -- move to seperate function
            neighbor.g = neighbor.parent.g + node_cost
            neighbor.h = abs(neighbor.position[0] - goal_node.position[0]) + abs(
                neighbor.position[1] - goal_node.position[1])
            neighbor.f = neighbor.g + neighbor.h
            # Check if neighbor is in open list and if it has a lower f value
            if in_open(open, neighbor):
                open.append(neighbor)

    # Return None, no path is found
    return None


def in_open(open, neighbor):
    for node in open:
        if neighbor == node and neighbor.f >= node.f:
            return False
    return True


'''''
Test Start and Goal
-73.590,45.490
-73.550,45.530

-73.578, 45.508
-73.576, 45.51

# No solution @ 50 threshold step 0.002
-73.590,45.490
-73.568,45.514

'''''
