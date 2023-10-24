"""Code for testing the bagged trees."""
import numpy as np
import pandas as pd
from DecisionTree import treebuilder

# train = np.loadtxt("data/bank/train.csv")
# test = np.loadtxt("data/bank/test.csv")
train = pd.read_csv("data/bank/train.csv")
test = pd.read_csv("data/bank/test.csv")

# print(train)


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

print(count / len(results))
# print(results)
