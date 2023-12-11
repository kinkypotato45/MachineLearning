import numpy as np
import pandas as pd


class logistic:
    def __init__(self, var, data, weights=None):
        if weights is None:
            self.weights = [0] * (data.shape[1]-1)
        else:
            self.weights = weights
        # print(self.weights)
        self.var = var
        self.data = data

        self.inputdict = {}
        self.outputdict = {}

    def train(self, epochs, r, d):
        firstval = 1
        for t in range(epochs):
            rate = r/(1 + r/d * t)
            table = self.data.sample(frac=1).to_numpy()
            for row in table:
                x = row[:-1]
                if not self.inputdict.__contains__(row[-1]):
                    self.inputdict[row[-1]] = firstval
                    self.outputdict[firstval] = row[-1]
                    firstval -= 2
                y = self.inputdict[row[-1]]

                calc = -y/(1+np.exp(y * np.dot(self.weights, x)))
                # print(x, y)
                # print(calc)
                weightder = []
                for i in range(len(row)-1):
                    weightder.append(
                        calc * row[i] + 2 * self.weights[i]/self.var)
                # print(weightder)
                # print(weightder)
                # print(self.weights - np.multiply(rate, weightder))
                self.weights = self.weights - np.multiply(rate, weightder)
                # print(self.weights)

                # self.weights = self.weights-np.dot(rate, weightder)
    def predict(self, x):
        sign = np.sign(np.dot(x, self.weights))
        return self.outputdict[sign]

        # return np.dot(x, self.weights)


# frame = pd.read_csv("data/train.csv", header=None)
# frame = frame.insert(0, "bias", 1)
# df = pd.DataFrame([1]*len(frame))
# print(frame.insert(0, "bias", 1))
# frame.insert(0, "bias", 1)
# print(frame.shape[1])
# model = logistic(.1, frame)
# model.train(5, .0025, .5)
# data = frame.to_numpy()
# err = 0
# total = 0
# for row in data:
#     total += 1
#     x = row[:-1]
#     y = row[-1]
#     if model.predict(x) != y:
#         err += 1
#     print(model.predict(x), y)
# print(err/total)
