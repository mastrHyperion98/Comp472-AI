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
            # disregard empty strings
            if not token:
                continue
            vocabulary.append(token)

    return np.unique(vocabulary)


def generate_frequency_per_type(data, freq_column, column, type):
    data_type = data[(data[column] == type)]
    sentences = data_type[freq_column].str.strip().str.lower()
    dict = {}
    for sentence in sentences:
        tokens = sentence.split(' ')
        for token in tokens:
            if token in dict:
                dict[token] = dict[token] + 1
            else:
                dict[token] = 1 # Increase to 2 for smoothness
    column_name = type+'_frequency'
    keys = list(dict.keys())
    values = list(dict.values())
    dict = {'word':keys, column_name:values}
    return pd.DataFrame(dict)


def generate_frequency_frame(data, vocabulary):
    freq_story = generate_frequency_per_type(data, 'Title', 'Post Type', 'story')
    freq_ask_hn = generate_frequency_per_type(data, 'Title', 'Post Type', 'ask_hn')
    freq_show_hn = generate_frequency_per_type(data, 'Title', 'Post Type', 'show_hn')
    freq_poll = generate_frequency_per_type(data, 'Title', 'Post Type', 'poll')
    vocabulary = pd.merge(vocabulary, freq_story, how='left', on='word')
    vocabulary = pd.merge(vocabulary, freq_ask_hn, how='left', on='word')
    vocabulary = pd.merge(vocabulary, freq_show_hn, how='left', on='word')
    vocabulary = pd.merge(vocabulary, freq_poll, how='left', on='word')
    # fill NA frequency with 0 // increase to 1 for smoothness
    vocabulary.fillna(0, inplace=True)

    return vocabulary


# gets the probability of a given value in a column
def probability_of(data, value, feature):
    num_observations = len(data[feature])
    num_entries = len(data[(data[feature]) == value])

    return float(num_entries/num_observations)


# get the probability of each word given a post type
def conditional_probability(vocabulary, list_types, delta):
    # Probability = P(W|C) = (frequency of w in C) /  total number of words in C + delta * size of vocab

    for type in list_types:
        probabilities = {}
        total_numbers_of_words_in_c = vocabulary[type+'_frequency'].sum()
        size_vocabulary = len(vocabulary)
        words = vocabulary['word'].tolist()
        for word in words:
            tmp = vocabulary[(vocabulary.word == word)]
            tmp = tmp[type+'_frequency'].tolist()
            frequency = tmp.pop()
            cp = (frequency + delta) / (total_numbers_of_words_in_c + (delta*size_vocabulary))
            probabilities[word] = cp
        # we need to merge probabilities into vocabulary
        keys = list(probabilities.keys())
        values = list(probabilities.values())
        dict = pd.DataFrame({'word':keys, 'p(word | {})'.format(type):values})
        vocabulary = pd.merge(vocabulary, dict, how='left', on='word')
    return vocabulary


# main function
def main():
    data = pd.read_csv('data/hns_2018_2019.csv')
    # Divide our data into two partitions based on year
    data_2018, data_2019 = split_data(data, '2018', '2019')
    list_types = np.unique(data_2018['Post Type'])
    vocabulary = pd.DataFrame(generate_vocabulary(data_2018, 'Title'), columns=['word'])
    # use to perform left merge on word
    vocabulary = generate_frequency_frame(data_2018, vocabulary)
    vocabulary = conditional_probability(vocabulary,list_types,0.5)
    print(vocabulary)


# Executes main function
if __name__ == "__main__":
    main()
