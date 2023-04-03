import pandas as pd
import Perceptron


def prepare_data(data_in_csv):
    # this function prepare the the for the wanted main
    df = pd.read_csv(data_in_csv)
    trn_features = df.sample(frac=0.8)  # to shuffle the data to train and test
    tst_features = df.drop(trn_features.index)
    # to fit the data to the labels of the perceptron
    trn_labels = trn_features[trn_features.columns[-1]].replace(to_replace=0, value=-1)
    tst_labels = tst_features[tst_features.columns[-1]].replace(to_replace=0, value=-1)
    # remove the labels column
    trn_features = trn_features.drop(columns=trn_features.columns[-1])
    tst_features = tst_features.drop(columns=tst_features.columns[-1])

    return trn_features, trn_labels, tst_features, tst_labels


if __name__ == "__main__":
    with open('Processed Wisconsin Diagnostic Breast Cancer.csv', 'r', encoding='utf8') as file:
        train_features, train_labels, test_features, test_labels = prepare_data(file)

    model = Perceptron.Perceptron()
    model.fit(train_features, train_labels)

    print("The weights of the model after training is:")
    for i, w in enumerate(model.weights_):
        print("w{} = {}".format(i, w))

    print("The training error is {} ".format(model.get_trn_error()))
    print("The Generalization error on the test is {} ".format(1 - model.score(test_features, test_labels)))  # the 1 - score is to get the error
