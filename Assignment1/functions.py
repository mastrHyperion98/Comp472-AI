import geopandas as gpd
import pandas as pd


# This function creates a data frame that is composed of all the blocks defined by our stepping function and its
# respective id
def generate_grid(step, min_x, max_x, min_y, max_y):
    num_block_y = round((max_y + step - min_y) / step, 0)
    num_block_x = round((max_x - min_x) / step, 0)
    block_id = []
    id_counter = 0

    lower_x = []
    lower_y = []
    upper_x = []
    upper_y = []

    x = min_x
    y = min_y
    while x <= max_x + step:
        while y <= max_y + step:
            lower_x.append(round(x, 3))
            lower_y.append(round(y, 3))
            upper_x.append(round(x + step, 3))
            upper_y.append(round(y + step, 3))
            y = y + step
            block_id.append(id_counter)
            id_counter = id_counter + 1
        x = x + step
        y = min_y

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
    block_frame = pd.merge(block_frame, crime_count_per_block, on='block_id')


def compute_threshold(block_frame):
    