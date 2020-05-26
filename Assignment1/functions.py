import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import FormatStrFormatter


# This function creates a data frame that is composed of all the blocks defined by our stepping function and its
# respective id
def generate_grid(step, min_x, max_x, min_y, max_y):
    block_id = []
    id_counter = 0
    lower_x = []
    lower_y = []
    upper_x = []
    upper_y = []

    ncols = int(round((max_x - min_x) / step, 0))
    nrows = int(round((max_y - min_y) / step, 0))

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


def compute_threshold(sorted):
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
    return _50th, _75th, _90th


# generate danger index
def generate_map(block_frame, threshold):
    count = block_frame.shape[0]
    cutoff = int(count * (threshold / 100.0))
    is_crime_high = pd.Series(np.zeros(count))
    block_frame = block_frame.reset_index()

    for index in range(cutoff):
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
        # list of x elements
        x = np.arange(low_x, upper_x, step)
        # list of y elements
        y = np.arange(low_y, upper_y, step)

        for _x in x:
            for _y in y:
                position = (round(_x, 3), round(_y, 3))
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
    return (x,y)


