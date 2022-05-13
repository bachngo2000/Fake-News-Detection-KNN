import numpy as np
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from math import sqrt

# defining data processing function
def preprocess_string(text):

    port_stem = PorterStemmer()
    processed_string = re.sub('[^a-zA-Z]', ' ', text)
    processed_string = processed_string.lower()
    processed_string = processed_string.split()
    processed_string = [port_stem.stem(word) for word in processed_string if not word in stopwords.words('english')]
    processed_string = ' '.join(processed_string)
    return processed_string

# calculating the euclidean distance between 2 point
def calc_euclidean_distance(coordinate1, coordinate2):
    return np.sqrt(np.sum((coordinate1 - coordinate2).power(2)))

# Calculating how accurate the model's prediction is
def calc_accuracy(true_label, pred_label):
    no_true_labels = len(true_label)
    no_matched_labels = np.sum(true_label == pred_label)
    acc = no_matched_labels / no_true_labels
    return acc

class K_Nearest_Neighbors:

    # initialize the instance and define k (number of neighbors)
    def __init__(self, k_val):

        self.k = k_val

    # fit the data into the model
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # predict the labels for a set of input X
    def predict(self, X):
        y_pred = [self.label_predict(x) for x in X]
        return np.array(y_pred)

    # predict a label for individual article
    def label_predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [calc_euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_index = np.argsort(distances)[: self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_index]
        # return the most common class label
        mostCommon_label = Counter(k_neighbor_labels).most_common(1)
        return mostCommon_label[0][0]


if __name__ == '__main__':

    # stopwords in English
    print(stopwords.words('english'))

    # Data Pre-processing:
    # loading the dataset to a pandas DataFrame
    news_dataset = pd.read_csv('news_data.csv')

    # print the first 5 rows of the dataFrame
    print(news_dataset.head())

    # counting the number of missing values in the dataset
    print(news_dataset.isnull().sum())

    # replacing the null values with empty string
    news_dataset = news_dataset.fillna('')

    # separating the data and label
    X = news_dataset.drop(columns='eval', axis=1)
    Y = news_dataset['eval']

    # Preprocess, stem, and "clean" the text
    news_dataset['title'] = news_dataset['title'].apply(preprocess_string)
    print(news_dataset['title'])

    # separating the data and eval
    X = news_dataset['title'].values
    y = news_dataset['eval'].values

    # Convert the textual data to numerical data
    vectorizer = TfidfVectorizer()
    vectorizer.fit(X)

    X = vectorizer.transform(X)

    # split the dataset into training and testing samples + labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=Y, random_state=2)

    print("Size of X_train: \n", )

    # k-value that returns the highest accuracy
    k = 13

    # instantiate a KNN model 
    clf = K_Nearest_Neighbors(k_val=k)
    print("------------------Training In Progress------------------------")
    print("Training Examples: ", X_train.shape)
    clf.fit(X_train, y_train)
    print('------------------------Training Finished!')

    predictions = clf.predict(X_test)
    print("KNN classification accuracy:", calc_accuracy(y_test, predictions))
