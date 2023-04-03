import Perceptron
import pandas as pd


if __name__ == "__main__":
    model = Perceptron.Perceptron()
    data = [[-2, -1], [0, 0], [2, 1], [1, 2], [-2, 2], [-3, 0]]
    labels = [-1, 1, 1, 1, -1, -1]

    data = pd.DataFrame(data)
    data["labels"] = labels

    model.fit(data.drop(columns=['labels']), data["labels"])  # to fit the data into the data that the model gets
    print("The weight vector is {}".format(model.weights_))
    print("The score is {}".format(model.score(data.drop(columns=['labels']), data["labels"])))
