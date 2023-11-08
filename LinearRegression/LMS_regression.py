import pandas as pd
import numpy as np
import copy
import math


class LMS_regression:
    def __init__(
        self, matrix, sample_size=None, weights=None, learning_rate=None, bound=None
    ):
        if sample_size is None:
            self.sample_size = len(matrix)
        else:
            self.sample_size = sample_size
        if weights is None:
            self.weights = [0] * (len(matrix[0]))
        else:
            self.weights = weights
        if learning_rate is None:
            self.learning_rate = 0.01
        else:
            self.learning_rate = learning_rate
        if bound is None:
            self.bound = 10 ** (-6)
        else:
            self.bound = bound
        self.frame = pd.DataFrame(matrix)
        self.frame.insert(0, "bias", 1)
        self.changedist = math.inf

    def descend(self, steps=None):
        if steps is None:
            self.steps = math.inf
        else:
            self.steps = steps
        # subsample = copy.copy(self.frame).sample(self.sample_size)
        # attr = subsample[subsample.columns[:-1]].to_numpy()
        # values = subsample[subsample.columns[-1]].to_numpy()
        i = 0
        print(self.steps)
        while self.changedist > self.bound and i < self.steps:
            i += 1
            print(i)
            subsample = copy.copy(self.frame).sample(self.sample_size)
            attr = subsample[subsample.columns[:-1]].to_numpy()
            values = subsample[subsample.columns[-1]].to_numpy()
            errSum = []
            for i in range(len(attr)):
                # print(attr[i])
                # print(values[i])
                err = -values[i] + np.dot(attr[i], self.weights)
                errSum.append(err)
            # print(errSum)
            change = self.learning_rate * np.matmul(errSum, attr)
            # print(change)
            # print(change)
            # print(np.linalg.norm(change))
            new_weight = self.weights - change
            self.weights = new_weight
            self.changedist = np.linalg.norm(change)
        print(i)

    def MSE(self):
        summation = 0
        attr = self.frame[self.frame.columns[:-1]].to_numpy()
        values = self.frame[self.frame.columns[-1]].to_numpy()
        for i in range(len(attr)):
            summation += (values[i] - np.dot(self.weights, attr[i])) ** 2
        return summation
