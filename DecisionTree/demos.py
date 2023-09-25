from ucimlrepo import fetch_ucirepo
import treebuilder
import pandas as pd
import numpy as np

# fetch dataset
training_cars = []
test_cars = []
with open("train.csv", 'r') as f:
    for line in f:
        terms = line.strip().split(',')
        training_cars.append(terms)
with open("test.csv", 'r') as f:
    for line in f:
        terms = line.strip().split(',')
        test_cars.append(terms)
training_cars = np.array(training_cars)
test_cars = np.array(test_cars)

truth = training_cars[:, -1]
test_cars = np.delete(training_cars, -1, axis=1)
for runs in range(1, 7):
    est_E_1 = treebuilder.build_tree(training_cars, runs, "E")
    est_ME_1 = treebuilder.build_tree(training_cars, runs, "ME")
    est_GI_1 = treebuilder.build_tree(training_cars, runs, "GI")
    err1 = []
    err2 = []
    err3 = []

    for i in test_cars:
        err1.append(est_E_1.predict(i))
        err2.append(est_ME_1.predict(i))
        err3.append(est_GI_1.predict(i))
    percent1 = 0
    percent2 = 0
    percent3 = 0
    for i in range(len(truth)):
        if truth[i] == err1[i]:
            percent1 += 1
        if truth[i] == err2[i]:
            percent2 += 1
        if truth[i] == err3[i]:
            percent3 += 1
    length = len(truth)
    print(percent1/length, percent2/length, percent3/length)
