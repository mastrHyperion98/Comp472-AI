# -------------------------------------------------------
# Assignment 2
# Written by Steven Smith 40057065
# For COMP 472 Section -  – Summer 2020
# --------------------------------------------------------
# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Splits our data around two given years
def split_data(data, year1, year2):
    data_2018 = data[(data['Created At'].str.find(year1) >= 0)]
    data_2019 = data[(data['Created At'].str.find(year2) >= 0)]
    return data_2018, data_2019


def generate_vocabulary(data, column):
    lower_cased = data[column].str.lower()
    vocabulary = []
    for statement in lower_cased:
        tokens = statement.split(' ')
        for token in tokens:
            vocabulary.append(token)

    return np.unique(vocabulary)


# main function
def main():
    data = pd.read_csv('data/hns_2018_2019.csv')
    # Divide our data into two partitions based on year
    data_2018, data_2019 = split_data(data, '2018', '2019')
    vocabulary = pd.DataFrame(generate_vocabulary(data_2018, 'Title'), columns=['word'])
    print(vocabulary)


if __name__ == "__main__":
    main()
