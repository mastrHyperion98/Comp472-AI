# -------------------------------------------------------
# Assignment 2
# Written by Steven Smith 40057065
# For COMP 472 Section -  – Summer 2020
# --------------------------------------------------------
# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import math
import time


# Splits our data around two given years
def split_data(data, year1, year2):
    data_2018 = data[(data['Created At'].str.find(year1) >= 0)]
    data_2019 = data[(data['Created At'].str.find(year2) >= 0)]
    return data_2018, data_2019


def generate_vocabulary(data, column):
    # strip the string and cast to lower
    lower_cased = data[column].str.strip().str.lower()
    vocabulary = []
    removed = []

    for statement in lower_cased:
        if statement.find('show hn') >= 0:
            vocabulary.append('show_hn')
            statement = statement.replace('show hn', '')
        if statement.find('ask hn') >= 0:
            vocabulary.append('ask_hn')
            statement = statement.replace('ask hn', '')

        tokens = word_tokenize(statement)
        for token in tokens:
            # only keep alpha numeric or alpha
            if token.isalpha() or token.isalnum() or token == '?' or token == ':' or token == '!':
                vocabulary.append(token)
            else:
                removed.append(token)
    return np.unique(vocabulary)


def generate_frequency_per_type(data, freq_column, column, type):
    data_type = data[(data[column] == type)]
    sentences = data_type[freq_column].str.strip().str.lower()
    dict = {}
    for sentence in sentences:
        if sentence.find('show hn') >= 0:
            token = 'show_hn'
            if token in dict:
                dict[token] = dict[token] + 1
            else:
                dict[token] = 1  # Increase to 2 for smoothness
            sentence = sentence.replace('show hn', '')

        if sentence.find('ask hn') >= 0:
            token = 'ask_hn'
            if token in dict:
                dict[token] = dict[token] + 1
            else:
                dict[token] = 1  # Increase to 2 for smoothness
            sentence = sentence.replace('ask hn', '')

        tokens = word_tokenize(sentence)
        for token in tokens:

            if token.isalpha() or token.isalnum() or token == '?' or token == ':' or token == '!':
                if token in dict:
                    dict[token] = dict[token] + 1
                else:
                    dict[token] = 1  # Increase to 2 for smoothness

    column_name = type + '_frequency'
    keys = list(dict.keys())
    values = list(dict.values())
    dictionary = {'word': keys, column_name: values}
    return pd.DataFrame(dictionary)


def generate_frequency_frame(data, vocabulary):
    print('COMPUTING FREQUENCY...')
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

    return float(num_entries / num_observations)


# get the probability of each word given a post type
def conditional_probability(vocabulary, list_types, delta):
    # Probability = P(W|C) = (frequency of w in C) /  total number of words in C + delta * size of vocab
    print('COMPUTING CONDITIONAL PROBABILITIES...')
    for type in list_types:
        probabilities = {}
        total_numbers_of_words_in_c = vocabulary[type + '_frequency'].sum()
        size_vocabulary = len(vocabulary)
        words = vocabulary['word'].tolist()
        for word in words:
            tmp = vocabulary[(vocabulary.word == word)]
            tmp = tmp[type + '_frequency'].tolist()
            frequency = tmp.pop()
            cp = (frequency + delta) / (total_numbers_of_words_in_c + (delta * size_vocabulary))
            probabilities[word] = cp
        # we need to merge probabilities into vocabulary
        keys = list(probabilities.keys())
        values = list(probabilities.values())
        dict = pd.DataFrame({'word': keys, 'p(word | {})'.format(type): values})
        vocabulary = pd.merge(vocabulary, dict, how='left', on='word')
    return vocabulary


def vocabulary_to_file(vocabulary):
    print('WRITING VOCABULARY TO FILE...')
    filename = 'model-2018.txt'
    file = open(filename, 'w', encoding='utf-8')
    to_write = ''
    for i in range(len(vocabulary)):
        index = i + 1
        word = vocabulary.at[i, 'word']
        frequency_story = vocabulary.at[i, 'story_frequency']
        frequency_ask = vocabulary.at[i, 'ask_hn_frequency']
        frequency_show = vocabulary.at[i, 'show_hn_frequency']
        frequency_poll = vocabulary.at[i, 'poll_frequency']
        cp_story = vocabulary.at[i, 'p(word | story)']
        cp_ask = vocabulary.at[i, 'p(word | ask_hn)']
        cp_show = vocabulary.at[i, 'p(word | show_hn)']
        cp_poll = vocabulary.at[i, 'p(word | poll)']
        to_write = to_write + ('{}  {}  {}  {}  {}  {}  {}  {}  {}  {}\n'.format(index, word, frequency_story, cp_story,
                                                                                 frequency_ask, cp_ask, frequency_show,
                                                                                 cp_show, frequency_poll, cp_poll))
    file.write(to_write)
    file.close()
    print('WRITING COMPLETE!')


class Naives_Bayes:
    def __init__(self):
        self.X = None
        self.y = None
        self.prior = {}

    def fit(self, X, y):

        length = len(X)
        for type in y:
            self.prior[type] = (len(X[X['Post Type'] == type]) + 1) / (length + 1)

        vocabulary = pd.DataFrame(generate_vocabulary(X, 'Title'), columns=['word'])
        # use to perform left merge on word
        vocabulary = generate_frequency_frame(X, vocabulary)
        vocabulary = conditional_probability(vocabulary, y, 0.5)
        vocabulary = vocabulary.sort_values(by=['word'])
        vocabulary_to_file(vocabulary)
        self.X = vocabulary
        self.y = y

    def predict(self, test):
        # argmaxcjlog(P(cj)) + Σlog(P(wi|c))
        predictions = []
        all_scores = {}
        for document in test:
            if 'selenium: storing data efficiently' == document:
                print("HELLO")
            document_lower = document.lower()
            show_hn = False
            ask_hn = False

            if document_lower.find('show hn') >= 0:
                document_lower = document.replace('show hn', '')
                show_hn = True
            if document_lower.find('ask hn') >= 0:
                document_lower = document_lower.replace('ask hn', '')
                ask_hn = True

            scores = {}
            for target in self.y:
                p_target = math.log10(self.prior[target])
                # check document for show hn or ask_hn
                tokens = word_tokenize(document_lower)
                if show_hn:
                    tokens.append('show_hn')
                if ask_hn:
                    tokens.append('ask_hn')

                score = p_target
                for token in tokens:
                    data = self.X
                    sub_frame = data[(data.word == token)]
                    if len(sub_frame) == 0:
                        continue
                    else:
                        sub_frame.reset_index(inplace=True)
                        score = score + math.log10(sub_frame.at[0, 'p(word | {})'.format(target)])
                scores[target] = score

            all_scores[document] = scores
            predict = max(scores, key=scores.get)
            predictions.append(predict)
        return all_scores, predictions


# main function
def main():
    start_time = time.process_time()
    data = pd.read_csv('data/hns_2018_2019.csv')
    # Divide our data into two partitions based on year
    data_2018, data_2019 = split_data(data, '2018', '2019')
    list_types = np.unique(data_2018['Post Type']).tolist()
    list_types.append('poll')
    NB = Naives_Bayes()
    NB.fit(data_2018, list_types)
    test_documents = data_2019['Title'].head(20).tolist()
    scores, predictions = NB.predict(test_documents)
    print(predictions)
    print(data_2019.head(20)['Post Type'])
    #print(scores)
   #print(time.process_time() - start_time)


# Executes main function
if __name__ == "__main__":
    main()
