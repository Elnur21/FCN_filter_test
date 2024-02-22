import numpy as np
import sklearn

from models.FCN import Classifier_FCN
from utils.helper  import *


def fit_classifier(dataset_name):
    df = read_dataset(dataset_name)
    x_train = df[0]
    y_train = df[1]
    x_test = df[2]
    y_test = df[3]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    classifier = Classifier_FCN("results/", input_shape, nb_classes, verbose=True)

    classifier.fit(x_train, y_train, x_test, y_test, y_true)



