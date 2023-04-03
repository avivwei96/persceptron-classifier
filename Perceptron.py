import numpy as np


def tuple_to_ind_and_vec(tuple_to_change):
    # this function get a tuple from itertuples() and returns the ind and a vector
    ind = tuple_to_change[0]
    vec = list(tuple_to_change)
    vec.append(1)  # to add another weight for the constant
    vec.pop(0)

    return ind, vec, len(vec)


class Perceptron(object):
    # this class is a perceptron model who is trying to find the best hyperplane
    # that separate two group of vectors in the vector space
    def __init__(self):
        self.weights_ = None
        self._trn_error = None

    def get_trn_error(self):
        return self._trn_error

    def fix_weights(self, features_tuple, labels):
        # this function get a tuple from itertuples and fix the weights of the perceptron
        # the function return 0 if the weights doesnt need to be fix and 1 otherwise

        index, vector, vector_size = tuple_to_ind_and_vec(features_tuple)
        const_num = self.weights_[-1]
        res = np.dot(vector, self.weights_)

        # checks if the dot product between the weights and the current vector match the lable
        # if not fix the weights
        if res >= 0 and labels[index] <= 0:
            self.weights_ = [self.weights_[i] - vector[i] for i in range(vector_size)]
            const_num -= 1
        elif res < 0 and labels[index] > 0:
            self.weights_ = [self.weights_[i] + vector[i] for i in range(vector_size)]
            const_num += 1
        else:
            return 0

        self.weights_[-1] = const_num
        return 1

    def _check_train_score(self, features, labels):
        # this function check the train score of the last prediction
        counter = wrong_predictions = 0
        labels = labels.reset_index(drop=True)
        # iterate over the the vectors to check the correct predictions
        for features_tup in features.itertuples():
            if self._single_predict(features_tup) == labels[counter]:
                wrong_predictions += 1
            counter += 1
        self._trn_error = 1 - (wrong_predictions / counter)

    def fit(self, features, labels):
        # this function get data and train the perceptron with that data
        features_len = features.shape[1]
        self.weights_ = [0 for x in range(features_len)]
        self.weights_.append(0)

        counter = 0  # counter for all the feature vectors
        is_there_errors = True

        while counter < 1000 and is_there_errors:
            is_there_errors = False
            for features_tup in features.itertuples():
                if self.fix_weights(features_tup, labels):
                    is_there_errors = True
            counter += 1

        self._check_train_score(features, labels)

    def _single_predict(self, features_tuple):
        # this function predict for a single vector
        index, vector, vector_size = tuple_to_ind_and_vec(features_tuple)
        res = np.dot(vector, self.weights_)
        if res >= 0:
            return 1
        else:
            return -1

    def predict(self, features):
        # this function give prediction to a group of vector of features
        labels_predict = list()

        for features_tup in features.itertuples():
            labels_predict.append(self._single_predict(features_tup))

        return labels_predict

    def score(self, features, labels):
        # this function returns the score (correct prediction / total prediction ) of the model
        # that was train

        predicted_labels = self.predict(features)
        counter = good_pred = 0
        labels = labels.reset_index(drop=True)

        # iterate over the predictions to count the correct predictions
        for index in range(len(predicted_labels)):
            if predicted_labels[index] == labels[index]:
                good_pred += 1
            counter += 1

        return good_pred / counter
