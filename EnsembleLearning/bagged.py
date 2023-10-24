# from sklearn.tree import DecisionTreeClassifier
from DecisionTree import treebuilder
import numpy as np
import pandas as pd
import math


# train = np.loadtxt("data/bank/train.csv")
# test = np.loadtxt("data/bank/test.csv")
train = pd.read_csv("data/bank/train.csv")
test = pd.read_csv("data/bank/test.csv")

# print(train)


class bag:
    def __init__(self, data, sample_size, itr):
        self.data = pd.DataFrame(data)
        self.sample_size = sample_size
        self.itr = itr
        self.trees = []
        for _ in range(itr):
            subset = self.data.sample(n=sample_size, replace=True)
            subTree = treebuilder.build_tree(subset.to_numpy(), math.inf, 'E')
            self.trees.append(subTree)
            # print(subTree)

    def predict(self, X):
        # arr = []
        # for tree in self.trees:
        #     arr.append(tree.predict(X))
        resultArr = [tree.predict(X) for tree in self.trees]
        return max(set(resultArr), key=resultArr.count)


train = pd.DataFrame(train)
testingbag = bag(data=train.to_numpy(), sample_size=100, itr=1)
X = train[train.columns[:-1]].to_numpy()
# print(len(X))

y = train[train.columns[-1]]
results = [testingbag.predict(x) for x in X]
count = 0
for i in range(len(results)):
    if results[i] == y[i]:
        count += 1

print(count/len(results))
# print(results)
