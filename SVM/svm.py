"""
A library implementing the perceptron algorithms. Ни пуха, ни пера!
"""
import pandas as pd
import numpy as np
import random
import copy
import scipy as sp
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


def dual(input):
    x = input[0]
    y = input[1]
    a = input[2]
    xTx = np.matmul(x, np.transpose(x))
    yyT = np.outer(y, y)
    aaT = np.outer(a, a)
    # print(aaT)
    thissum = 0
    for i in range(len(xTx)):
        for j in range(len(xTx[i])):
            thissum += xTx[i][j] * yyT[i][j] * aaT[i][j]
    # print(thissum)
    return 1 / 2 * thissum - sum(a)


class dualSVM:
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
        # print(frame)
        ones = pd.DataFrame({"bias": [1] * len(frame)})
        self.samples = ones.join(frame)

    # def dual(self, x, y, alpha):
    #     value = np.transpose(x)
    #     return -sum(alpha)

    def learn(self):
        x = self.samples[self.samples.columns[:-1]].to_numpy()
        # print(x)
        y = self.samples[self.samples.columns[-1]].to_numpy()
        yvalues = 1
        self.valuedict = {}
        self.returndict = {}

        for i in range(len(y)):
            if not self.valuedict.__contains__(y[i]):
                self.valuedict[y[i]] = yvalues
                self.returndict[yvalues] = y[i]
                yvalues -= 2
            y[i] = self.valuedict[y[i]]
        # print(y)

        # const = ({'type', 'eq', 'fun': })
        # tempframe = self.samples.to_numpy()
        self.alpha = [1] * len(y)
        print(dual((x, y, self.alpha)))

        # res = sp.optimize.minimize(dual, (x, y, self.alpha))
        # yvalues = 1
        # self.valuedict = {}
        # self.returndict = {}
        #
        # self.w = [0] * (len(self.samples.columns) - 1)
        # yvalues = 1
        # self.valuedict = {}
        # self.returndict = {}
        # n = len(self.samples)
        # gamma_0 = gamma
        # for t in range(epochs):
        #     gamma = gamma_0 / (1 + gamma_0 / a * t)
        #     examples = self.samples.sample(frac=1).to_numpy()
        #     for row in examples:
        #         x = row[:-1]
        #         y = row[-1]
        #         # print(y)
        #         if not self.valuedict.__contains__(y):
        #             self.valuedict[y] = yvalues
        #             self.returndict[yvalues] = y
        #             yvalues -= 2
        #
        #         loss = hingeLoss(x, self.valuedict[y], self.w)
        #         # print(loss)
        #         if loss >= 0:
        #             tempw = self.w.copy()
        #             self.w = (
        #                 self.w
        #                 - np.multiply(gamma, tempw)
        #                 + np.multiply(gamma * C * n * self.valuedict[y], x)
        #             )
        #         else:
        #             bias = self.w[0]
        #             self.w = np.multiply((1 - gamma), self.w)
        #             self.w[0] = bias

        #


frame = pd.read_csv("bank-note-1/test.csv", header=None).to_numpy()

# this = VotedPerceptron(frame)
# values = this.learn(0.5, 1)
# svm = dualSVM(frame)
# svm.learn()
# print(svm.returnWeights())
# count = 0

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
