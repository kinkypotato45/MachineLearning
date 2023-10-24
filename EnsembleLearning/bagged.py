"""Class constructor for a bootstrapped voting tree."""
import math

import numpy as np
import pandas as pd

from DecisionTree import treebuilder

# train = np.loadtxt("data/bank/train.csv")
# test = np.loadtxt("data/bank/test.csv")
train = pd.read_csv("data/bank/train.csv")
test = pd.read_csv("data/bank/test.csv")

# print(train)


class Bag:
    """A class constructor for a bag of tree."""

    def __init__(self, data, sample_size, itr):
        """
        Tool for creating a number of trees, which vote on an answer.

        data: a dataframe or array, sample_size: number of random rows sampled
        from the dataset, itr: number of trees.
        """
        self.data = pd.DataFrame(data)
        self.sample_size = sample_size
        self.itr = itr
        self.trees = []
        for _ in range(itr):
            subset = self.data.sample(n=sample_size, replace=True)
            sub_tree = treebuilder.BuildTree(subset.to_numpy(), math.inf, "E")
            self.trees.append(sub_tree)
            # print(subTree)

    def predict(self, X):
        """
        Tool for predicting what an attribute will be; takes in an
        n dimensional vector (one less than that of the data).

        X: a vector to be predicted
        """
        # arr = []
        # for tree in self.trees:
        #     arr.append(tree.predict(X))
        result_arr = [tree.predict(X) for tree in self.trees]
        return max(set(result_arr), key=result_arr.count)


train = pd.DataFrame(train)
trainingbag = Bag(data=train.to_numpy(), sample_size=100, itr=10)
X = test[test.columns[:-1]].to_numpy()
# print(len(X))

y = test[test.columns[-1]]
results = [trainingbag.predict(x) for x in X]
count = 0
for i in range(len(results)):
    if results[i] == y[i]:
        count += 1

print(count / len(results))
# print(results)
