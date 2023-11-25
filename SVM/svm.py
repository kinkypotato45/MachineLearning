"""
A library implementing the perceptron algorithms. Ни пуха, ни пера!
"""
import pandas as pd
import numpy as np
import random
import copy
from pandas.core.indexes.api import Axis


random.seed(10)
# print(frame)
# print(len(frame))


def hingeLoss(x, y, w):
    values = [0]
    values.append(1 - y * np.dot(x, w))
    return max(values)


class primalSVM:
    def __init__(self, frame):
        """
        Takes in a dataframe, and create a linear classifier

        Parameters
        ----------
        Frame: a pandas dataframe, where each column represents real value,
        and the last column is a binary classfication.

        r: the scale of how much to change weight vector w
        """
        frame = pd.DataFrame(frame)
        ones = pd.DataFrame({"bias": [1] * len(frame)})
        self.samples = ones.join(frame)

    def learn(self, C, gamma, a, epochs):
        yvalues = 1
        self.valuedict = {}
        self.returndict = {}

        self.w = [0] * (len(self.samples.columns) - 1)
        yvalues = 1
        self.valuedict = {}
        self.returndict = {}
        n = len(self.samples)
        gamma_0 = gamma
        for t in range(epochs):
            gamma = gamma_0 / (1 + gamma_0 / a * t)
            examples = self.samples.sample(frac=1).to_numpy()
            for row in examples:
                x = row[:-1]
                y = row[-1]
                # print(y)
                if not self.valuedict.__contains__(y):
                    self.valuedict[y] = yvalues
                    self.returndict[yvalues] = y
                    yvalues -= 2

                loss = hingeLoss(x, self.valuedict[y], self.w)
                # print(loss)
                if loss >= 0:
                    tempw = self.w.copy()
                    self.w = (
                        self.w
                        - np.multiply(gamma, tempw)
                        + np.multiply(gamma * C * n * self.valuedict[y], x)
                    )
                else:
                    bias = self.w[0]
                    self.w = np.multiply((1 - gamma), self.w)
                    self.w[0] = bias

    def predict(self, input):
        input = np.append([1], input)
        if np.dot(self.w, input) < 0:
            return self.returndict[-1]
        else:
            return self.returndict[1]

    def returnWeights(self):
        return self.w

        # class VotedPerceptron:
        #     def __init__(self, frame):
        #         """
        #         Takes in a dataframe, and create a linear classifier
        #
        #         Parameters
        #         ----------
        #         Frame: a pandas dataframe, where each column represents real value,
        #         and the last column is a binary classfication.
        #
        #         r: the scale of how much to change weight vector w
        #         """
        #         ones = pd.DataFrame({"bias": [1] * len(frame)})
        #         self.samples = ones.join(frame).to_numpy()
        #         self.valuedict = {}
        #         self.returndict = {}
        #         self.WC = []
        #         # self.classes = {}
        #         # print(self.samples)
        #
        #     def learn(self, r, epochs):
        #         self.w = [0] * (len(self.samples[0]) - 1)
        #         # print(self.w)
        #         yvalues = 1
        #         c = 1
        #         for _ in range(epochs):
        #             for i in self.samples:
        #                 x = i[:-1]
        #                 y = i[-1]
        #                 if not self.valuedict.__contains__(y):
        #                     self.valuedict[y] = yvalues
        #                     self.returndict[yvalues] = y
        #                     yvalues -= 2
        #                 if np.dot(self.w, x) * self.valuedict[y] <= 0:
        #                     # print(self.w, c)
        #                     self.WC.append((self.w.copy(), copy.copy(c)))
        #                     self.w = self.w + r * x * self.valuedict[y]
        #                     c = 1
        #                 else:
        #                     c += 1
        #         self.WC.append((self.w.copy(), copy.copy(c)))
        #
        #     def predict(self, input):
        #         input = np.append([1], input)
        #         value = 0
        #         for weight, c in self.WC:
        #             value += c * np.sign(np.dot(weight, input))
        #
        #         if value < 0:
        #             return self.returndict[-1]
        #         else:
        #             return self.returndict[1]
        #
        #     def returnWeights(self):
        #         return self.WC
        #
        #
        # class AveragePerceptron:
        #     def __init__(self, frame):
        #         """
        #         Takes in a dataframe, and create a linear classifier
        #
        #         Parameters
        #         ----------
        #         Frame: a pandas dataframe, where each column represents real value,
        #         and the last column is a binary classfication.
        #
        #         r: the scale of how much to change weight vector w
        #         """
        #         ones = pd.DataFrame({"bias": [1] * len(frame)})
        #         self.samples = ones.join(frame).to_numpy()
        #         # self.classes = {}
        #         # print(self.samples)
        #
        #     def learn(self, r, epochs):
        #         self.w = [0] * (len(self.samples[0]) - 1)
        #         self.a = [0] * (len(self.samples[0]) - 1)
        #
        #         # print(self.w)
        #         yvalues = 1
        #         self.valuedict = {}
        #         self.returndict = {}
        #         for _ in range(epochs):
        #             for i in self.samples:
        #                 x = i[:-1]
        #                 y = i[-1]
        #                 if not self.valuedict.__contains__(y):
        #                     self.valuedict[y] = yvalues
        #                     self.returndict[yvalues] = y
        #                     yvalues -= 2
        #                 if np.dot(self.w, x) * self.valuedict[y] <= 0:
        #                     self.w = self.w + r * x * self.valuedict[y]
        #                 self.a = self.a + self.w
        #
        #     def predict(self, input):
        #         input = np.append([1], input)
        #         value = np.dot(self.a, input)
        #
        #         if value < 0:
        #             return self.returndict[-1]
        #         else:
        return self.returndict[1]


#
#     def returnWeights(self):
#         return self.a
#


#
# frame = pd.read_csv("bank-note-1/test.csv", header=None).to_numpy()
# # this = VotedPerceptron(frame)
# # values = this.learn(0.5, 1)
# svm = primalSVM(frame)
# svm.learn(C=100 / 873, gamma=0.1, a=1, epochs=100)
# print(svm.returnWeights())
# count = 0
#
# for i in range(len(frame)):
#     row = frame[i]
#     # print(row)
#     x = row[:-1]
#     y = row[-1]
#     if svm.predict(x) == y:
#         count += 1
#     # print(frame[i])
#     # print(svm.predict(x), y)
# print(count)
#
# print(count / len(frame))
