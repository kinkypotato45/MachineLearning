"""
A library implementing the perceptron algorithms. Ни пуха, ни пера!
"""
import pandas as pd
import numpy as np
import copy
from pandas.core.indexes.api import Axis

frame = pd.read_csv("bank-note-1/train.csv", header=None)
# print(frame)
# print(len(frame))


class Perceptron:
    def __init__(self, frame):
        """
        Takes in a dataframe, and create a linear classifier

        Parameters
        ----------
        Frame: a pandas dataframe, where each column represents real value,
        and the last column is a binary classfication.

        r: the scale of how much to change weight vector w
        """
        ones = pd.DataFrame({"bias": [1] * len(frame)})
        self.samples = ones.join(frame).to_numpy()

    def learn(self, r, epochs):
        self.w = [0] * (len(self.samples[0]) - 1)
        # print(self.w)
        yvalues = 1
        self.valuedict = {}
        self.returndict = {}
        for _ in range(epochs):
            for i in self.samples:
                x = i[:-1]
                y = i[-1]
                if not self.valuedict.__contains__(y):
                    self.valuedict[y] = yvalues
                    self.returndict[yvalues] = y
                    yvalues -= 2
                if np.dot(self.w, x) * self.valuedict[y] <= 0:
                    self.w = self.w + r * x * self.valuedict[y]

    def predict(self, input):
        input = np.append([1], input)
        # print(input)
        if np.dot(self.w, input) < 0:
            return self.returndict[-1]
        else:
            return self.returndict[1]

    def returnWeights(self):
        return self.w


class VotedPerceptron:
    def __init__(self, frame):
        """
        Takes in a dataframe, and create a linear classifier

        Parameters
        ----------
        Frame: a pandas dataframe, where each column represents real value,
        and the last column is a binary classfication.

        r: the scale of how much to change weight vector w
        """
        ones = pd.DataFrame({"bias": [1] * len(frame)})
        self.samples = ones.join(frame).to_numpy()
        self.valuedict = {}
        self.returndict = {}
        self.WC = []
        # self.classes = {}
        # print(self.samples)

    def learn(self, r, epochs):
        self.w = [0] * (len(self.samples[0]) - 1)
        # print(self.w)
        yvalues = 1
        c = 1
        for _ in range(epochs):
            for i in self.samples:
                x = i[:-1]
                y = i[-1]
                if not self.valuedict.__contains__(y):
                    self.valuedict[y] = yvalues
                    self.returndict[yvalues] = y
                    yvalues -= 2
                if np.dot(self.w, x) * self.valuedict[y] <= 0:
                    # print(self.w, c)
                    self.WC.append((self.w.copy(), copy.copy(c)))
                    self.w = self.w + r * x * self.valuedict[y]
                    c = 1
                else:
                    c += 1
        self.WC.append((self.w.copy(), copy.copy(c)))

    def predict(self, input):
        input = np.append([1], input)
        value = 0
        for weight, c in self.WC:
            value += c * np.sign(np.dot(weight, input))

        if value < 0:
            return self.returndict[-1]
        else:
            return self.returndict[1]

    def returnWeights(self):
        return self.WC


class AveragePerceptron:
    def __init__(self, frame):
        """
        Takes in a dataframe, and create a linear classifier

        Parameters
        ----------
        Frame: a pandas dataframe, where each column represents real value,
        and the last column is a binary classfication.

        r: the scale of how much to change weight vector w
        """
        ones = pd.DataFrame({"bias": [1] * len(frame)})
        self.samples = ones.join(frame).to_numpy()
        # self.classes = {}
        # print(self.samples)

    def learn(self, r, epochs):
        self.w = [0] * (len(self.samples[0]) - 1)
        self.a = [0] * (len(self.samples[0]) - 1)

        # print(self.w)
        yvalues = 1
        self.valuedict = {}
        self.returndict = {}
        for _ in range(epochs):
            for i in self.samples:
                x = i[:-1]
                y = i[-1]
                if not self.valuedict.__contains__(y):
                    self.valuedict[y] = yvalues
                    self.returndict[yvalues] = y
                    yvalues -= 2
                if np.dot(self.w, x) * self.valuedict[y] <= 0:
                    self.w = self.w + r * x * self.valuedict[y]
                self.a = self.a + self.w

    def predict(self, input):
        input = np.append([1], input)
        value = np.dot(self.a, input)

        if value < 0:
            return self.returndict[-1]
        else:
            return self.returndict[1]

    def returnWeights(self):
        return self.a


#
# this = VotedPerceptron(frame)
# values = this.learn(0.5, 1)
# frame = frame.to_numpy()
# # count = 0
# for i in range(100):
#     row = frame[i]
#     # print(row)
#     x = row[:-1]
#     y = row[-1]
#     # print(frame[i])
#     print(this.predict(x), y)
