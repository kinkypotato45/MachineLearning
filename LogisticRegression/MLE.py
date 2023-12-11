import numpy as np
import pandas as pd


class MLE:
    def __init__(self,  data, weights=None):
        if weights is None:
            self.weights = [0] * (data.shape[1]-1)
        else:
            self.weights = weights
        # print(self.weights)
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

                calc = -(y - np.dot(x, self.weights))
                # print(x, y)
                # print(calc)
                weightder = []
                for i in range(len(row)-1):
                    weightder.append(calc * row[i])
                self.weights = self.weights - np.multiply(rate, weightder)

                # self.weights = self.weights-np.dot(rate, weightder)
    def predict(self, x):
        sign = np.sign(np.dot(x, self.weights))
        return self.outputdict[sign]
