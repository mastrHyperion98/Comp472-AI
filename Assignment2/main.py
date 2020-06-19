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
from sklearn.metrics import accuracy_score
import math
import time


# Splits our data around two given years
def split_data(data, year1, year2):
    data_2018 = data[(data['Created At'].str.find(year1) >= 0)]
    data_2019 = data[(data['Created At'].str.find(year2) >= 0)]
    return data_2018, data_2019


def generate_vocabulary(data, stop_words=[], min_len=1, max_len=10000):
    # strip the string and cast to lower
    lower_cased = data['Title'].str.strip().str.lower()
    vocabulary = []
    removed = []

    for statement in lower_cased:
        if "show hn" in statement or "show_hn" in statement:
            vocabulary.append('show_hn')
            statement = statement.replace('show hn', '')
            statement = statement.replace('show_hn', '')
        if "ask hn" in statement or "ask_hn" in statement:
            vocabulary.append('ask_hn')
            statement = statement.replace('ask hn', '')
            statement = statement.replace('ask_hn', '')

        tokens = word_tokenize(statement)
        for token in tokens:
            # only keep alpha numeric or alpha
            if min_len <= len(token) <= max_len and (
                    token.isalpha() or token.isalnum() or token == '?' or token == ':' or token == '!') and token not in stop_words:
                vocabulary.append(token)
            else:
                removed.append(token)

    vocabulary = np.unique(vocabulary)
    return vocabulary, removed


def write_generated_vocab_and_removed(vocabulary, removed):
    ## Print Vocabulary and removed words
    filename = 'vocabulary.txt'
    with open(filename, 'w', encoding='utf-8') as file:
        for word in vocabulary:
            file.write("%s\n" % word)
        file.close()

    filename = "remove_word.txt"
    removed = np.unique(removed)
    with open(filename, 'w', encoding='utf-8') as file:
        for word in removed:
            file.write("%s\n" % word)
        file.close()


def generate_frequency_per_type(data, freq_column, column, type):
    data_type = data[(data[column] == type)]
    sentences = data_type[freq_column].str.strip().str.lower()
    dict = {}
    for sentence in sentences:
        if "show hn" in sentence or "show_hn" in sentence:
            token = 'show_hn'
            if token in dict:
                dict[token] = dict[token] + 1
            else:
                dict[token] = 1  # Increase to 2 for smoothness
            sentence = sentence.replace('show hn', '')
            sentence = sentence.replace('show_hn', '')

        if "ask hn" in sentence or "ask_hn" in sentence:
            token = 'ask_hn'
            if token in dict:
                dict[token] = dict[token] + 1
            else:
                dict[token] = 1  # Increase to 2 for smoothness
            sentence = sentence.replace('ask hn', '')
            sentence = sentence.replace('ask_hn', '')

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


def frequency_sum(a,b,c,d):
    return a+b+c+d


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
    vocabulary.fillna(0, inplace=True)
    # Store word frequency
    vocabulary['frequency'] = vocabulary.apply(
        lambda row: frequency_sum(row['story_frequency'],row['show_hn_frequency'],row['ask_hn_frequency'],row['poll_frequency']), axis=1)
    # fill NA frequency with 0 // increase to 1 for smoothness
    return vocabulary


# gets the probability of a given value in a column
def probability_of(frequency, total, size, delta):
    return (frequency + delta) / (total + (delta * size))


# get the probability of each word given a post type
def conditional_probability(vocabulary, list_types, delta):
    # Probability = P(W|C) = (frequency of w in C) /  total number of words in C + delta * size of vocab
    print('COMPUTING CONDITIONAL PROBABILITIES...')
    for type in list_types:
        probabilities = {}
        total_numbers_of_words_in_c = vocabulary[type + '_frequency'].sum()
        size_vocabulary = len(vocabulary)
        vocabulary['p(word | {})'.format(type)] = vocabulary.apply(
            lambda row: probability_of(frequency=row[type + '_frequency'], total=total_numbers_of_words_in_c,
                                       size=size_vocabulary, delta=delta), axis=1)
    return vocabulary


def vocabulary_to_file(filename, vocabulary):
    print('WRITING VOCABULARY TO FILE...')
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
        self.vocab = None
        self.prior = None

    def fit(self, X, y, prior):
        X.set_index("word", drop=True, inplace=True)
        self.X = X.to_dict(orient="index")
        self.y = y
        self.prior = prior

    def predict(self, test):
        # argmaxcjlog(P(cj)) + Σlog(P(wi|c))
        predictions = []
        all_scores = []
        counter = 0
        for document in test:
            counter = counter + 1
            document_lower = document.lower()
            show_hn = False
            ask_hn = False

            if 'show hn' in document_lower or 'show_hn' in document_lower:
                document_lower = document.replace('show hn', '')
                document_lower = document.replace('show_hn', '')
                show_hn = True

            if 'ask hn' in document_lower or 'ask_hn' in document_lower:
                document_lower = document_lower.replace('ask hn', '')
                document_lower = document_lower.replace('ask_hn', '')
                ask_hn = True

            tokens = word_tokenize(document_lower)
            if show_hn:
                tokens.append('show_hn')
            if ask_hn:
                tokens.append('ask_hn')
            scores = {}
            for target in self.y:
                p_target = math.log10(self.prior[target])
                # check document for show hn or ask_hn
                score = p_target
                for token in tokens:
                    data = self.X
                    if token in data:
                        value = math.log10(data[token]['p(word | {})'.format(target)])
                        score = score + value
                scores[target] = score
            all_scores.append(scores)
            predict = max(scores, key=scores.get)
            predictions.append(predict)
        return all_scores, predictions


def write_results_to_file(filename, predictions, score, documents, true_values):
    with open(filename, 'w', encoding='utf8') as file:
        counter = 1
        for i in range(len(documents)):
            string = '{}  {}  {}  {}  {}  {}  {}  {}  {}\n'.format(counter, documents[i],
                                                                   predictions[i],
                                                                   score[i]['story'],
                                                                   score[i]['ask_hn'],
                                                                   score[i]['show_hn'],
                                                                   score[i]['poll'],
                                                                   true_values[counter - 1],
                                                                   true_values[counter - 1] == predictions[i])
            counter = counter + 1
            file.write(string)
        file.close()


def task_1_and_2(data_2018, list_types, data_2019, prior):
    print('****TASK 1 and 2****\n')
    words, removed_words = generate_vocabulary(data_2018)
    write_generated_vocab_and_removed(words, removed_words)
    vocabulary = pd.DataFrame(words, columns=['word'])
    # use to perform left merge on word
    vocabulary = generate_frequency_frame(data_2018, vocabulary)
    vocabulary = conditional_probability(vocabulary, list_types, 0.5)
    vocabulary = vocabulary.sort_values(by=['word'])
    vocabulary_to_file('model-2018.txt', vocabulary)
    NB = Naives_Bayes()
    NB.fit(vocabulary, list_types, prior)
    test_documents = data_2019['Title'].tolist()
    print('PREDICTING TEST RESULT')
    scores, predictions = NB.predict(test_documents)
    print('WRITING RESULTS TO FILE ...')
    write_results_to_file('baseline-result.txt', predictions, scores, data_2019.Title.tolist(),
                          data_2019['Post Type'].tolist())
    print('WRITING RESULTS TO FILE COMPLETED')
    print('Accuracy Score on test data: ', accuracy_score(y_true=data_2019['Post Type'].tolist(),
                                                          y_pred=predictions))


def exp1(data_2018, list_types, data_2019, prior):
    print('\n****EXPERIMENT 1*****\n')
    stop_words = []

    with open('data/stopwords.txt', 'r') as file:
        Lines = file.readlines()
        for line in Lines:
            stop_words.append(line.strip())

    words, removed_words = generate_vocabulary(data_2018, stop_words=stop_words)
    vocabulary = pd.DataFrame(words, columns=['word'])
    # use to perform left merge on word
    vocabulary = generate_frequency_frame(data_2018, vocabulary)
    vocabulary = conditional_probability(vocabulary, list_types, 0.5)
    vocabulary = vocabulary.sort_values(by=['word'])
    vocabulary_to_file('stopword-model.txt', vocabulary)
    NB = Naives_Bayes()
    NB.fit(vocabulary, list_types, prior)
    test_documents = data_2019['Title'].tolist()
    print('PREDICTING TEST RESULT')
    scores, predictions = NB.predict(test_documents)
    print('WRITING RESULTS TO FILE ...')
    write_results_to_file('stopword-result.txt', predictions, scores, data_2019.Title.tolist(),
                          data_2019['Post Type'].tolist())
    print('WRITING RESULTS TO FILE COMPLETED')
    print('Experiment 1 Accuracy Score on test data: ', accuracy_score(y_true=data_2019['Post Type'].tolist(),
                                                                       y_pred=predictions))


def exp2(data_2018, list_types, data_2019, prior):
    print('\n****EXPERIMENT 2*****\n')
    stop_words = []

    with open('data/stopwords.txt', 'r') as file:
        Lines = file.readlines()
        for line in Lines:
            stop_words.append(line.strip())

    words, removed_words = generate_vocabulary(data_2018, min_len=3, max_len=8)
    vocabulary = pd.DataFrame(words, columns=['word'])
    # use to perform left merge on word
    vocabulary = generate_frequency_frame(data_2018, vocabulary)
    vocabulary = conditional_probability(vocabulary, list_types, 0.5)
    vocabulary = vocabulary.sort_values(by=['word'])
    vocabulary_to_file('wordlength-model.txt', vocabulary)
    NB = Naives_Bayes()
    NB.fit(vocabulary, list_types, prior)
    test_documents = data_2019['Title'].tolist()
    print('PREDICTING TEST RESULT')
    scores, predictions = NB.predict(test_documents)
    print('WRITING RESULTS TO FILE ...')
    write_results_to_file('wordlength-result.txt', predictions, scores, data_2019.Title.tolist(),
                          data_2019['Post Type'].tolist())
    print('WRITING RESULTS TO FILE COMPLETED')
    print('Experiment 2 Accuracy Score on test data: ', accuracy_score(y_true=data_2019['Post Type'].tolist(),
                                                                       y_pred=predictions))

def exp3(data_2018, list_types, data_2019, prior):
    print('\n****EXPERIMENT 3****\n')
    words, removed_words = generate_vocabulary(data_2018)
    write_generated_vocab_and_removed(words, removed_words)
    vocabulary = pd.DataFrame(words, columns=['word'])
    # use to perform left merge on word
    vocabulary = generate_frequency_frame(data_2018, vocabulary)
    vocabulary = conditional_probability(vocabulary, list_types, 0.5)
    vocabulary = vocabulary.sort_values(by=['word'])
    vocabulary_to_file('model-2018.txt', vocabulary)
    NB = Naives_Bayes()

    test_documents = data_2019['Title'].tolist()

    print("PERFORMING PART 1: LOWEST FREQUENCY")
    # Perform part 1
    y = []
    x = []

    for i in range(5):
        threshold = i*5
        if threshold == 0:
            threshold = 1
        dataset = vocabulary[(vocabulary.frequency > threshold)]
        columns = ['p(word | story)', 'p(word | ask_hn)', 'p(word | show_hn)', 'p(word | poll)']
        dataset = dataset.drop(columns, axis=1).copy()
        dataset = conditional_probability(dataset, list_types, 0.5)
        x.append(len(dataset))
        NB.fit(dataset, list_types, prior)
        scores, predictions = NB.predict(test_documents)
        y.append(accuracy_score(y_true=data_2019['Post Type'].tolist(),
                                                          y_pred=predictions))

    print("DRAWING FIGURE 1")
    fig, ax = plt.subplots()
    ax.plot(x,y)
    ax.set_title('Figure 1: Classifier Performance by Removing Least Common Words')
    ax.set_xlabel('Vocabulary Size')
    ax.set_ylabel('Model Accuracy')
    plt.show()

    # Then gradually remove the top 5% most frequent words,
    # the 10% most frequent words, 15%, 20% and 25% most frequent words.

    y = []
    x = []
    for i in range(5):
        threshold = ((i * 5) + 5)/100
        dataset = vocabulary[(vocabulary.frequency < vocabulary.frequency.quantile(1-threshold))]
        # recompute probabilities
        columns = ['p(word | story)','p(word | ask_hn)','p(word | show_hn)','p(word | poll)']
        dataset = dataset.drop(columns, axis=1).copy()
        dataset = conditional_probability(dataset, list_types, 0.5)
        x.append(len(dataset))
        NB.fit(dataset, list_types, prior)
        scores, predictions = NB.predict(test_documents)
        y.append(accuracy_score(y_true=data_2019['Post Type'].tolist(),
                                y_pred=predictions))

    print("DRAWING FIGURE 2")
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title('Figure 2: Classifier Performance by Removing Most Common Words')
    ax.set_xlabel('Vocabulary Size')
    ax.set_ylabel('Model Accuracy')
    plt.show()


# main function
def main():
    start_time = time.process_time()
    data = pd.read_csv('data/hns_2018_2019.csv')
    # Divide our data into two partitions based on year
    data_2018, data_2019 = split_data(data, '2018', '2019')
    list_types = np.unique(data_2018['Post Type']).tolist()
    list_types.append('poll')
    prior = {}
    length = len(data_2018)
    for type in list_types:
        prior[type] = (len(data_2018[data_2018['Post Type'] == type]) + 1) / (length + 1)

    task_1_and_2(data_2018, list_types, data_2019, prior)
    exp1(data_2018, list_types, data_2019, prior)
    exp2(data_2018, list_types, data_2019, prior)
    exp3(data_2018, list_types, data_2019, prior)
    print(time.process_time() - start_time)


# Executes main function
if __name__ == "__main__":
    main()
