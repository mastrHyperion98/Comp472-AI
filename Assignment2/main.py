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
    # strip the string and cast to lower
    lower_cased = data[column].str.strip().str.lower()
    vocabulary = []
    for statement in lower_cased:
        tokens = statement.split(' ')
        for token in tokens:
            vocabulary.append(token)

    return np.unique(vocabulary)


def generate_frequency_per_type(data, freq_column, column, type):
    data_type = data[(data[column] == type)]
    sentences = data[freq_column].str.strip().str.lower()
    dict = {}
    for sentence in sentences:
        tokens = sentence.split(' ')
        for token in tokens:
            if token in dict:
                dict[token] = dict[token] + 1
            else:
                dict[token] = 1
    return pd.DataFrame(dict, columns=['word, frequency'])


# main function
def main():
    data = pd.read_csv('data/hns_2018_2019.csv')
    # Divide our data into two partitions based on year
    data_2018, data_2019 = split_data(data, '2018', '2019')
    vocabulary = pd.DataFrame(generate_vocabulary(data_2018, 'Title'), columns=['word'])
    # use to perform left merge on word
    generate_frequency_per_type(data_2018,'Title', 'Post Type', 'story')


if __name__ == "__main__":
    main()
