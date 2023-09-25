import numpy as np
import pandas as pd
import treebuilder
training_bank = []

testing_bank = []
with open("bank-4/train.csv", 'r') as f:
    for line in f:
        terms = line.strip().split(',')
        training_bank.append(terms)
with open("bank-4/test.csv", 'r') as f:
    for line in f:
        terms = line.strip().split(',')
        testing_bank.append(terms)
training_bank = np.array(training_bank)
testing_bank = np.array(testing_bank)
trainingframe = pd.DataFrame(training_bank)

testing_truth = testing_bank[:, -1]
training_truth = training_bank[:, -1]

test_bank = np.delete(testing_bank, -1, axis=1)
train_bank = np.delete(training_bank, -1, axis=1)
for runs in range(1, 17):
    est_E_1 = treebuilder.build_tree(training_bank, runs, "E")
    est_ME_1 = treebuilder.build_tree(training_bank, runs, "ME")
    est_GI_1 = treebuilder.build_tree(training_bank, runs, "GI")
    test_err1 = []
    test_err2 = []
    test_err3 = []
    train_err1 = []
    train_err2 = []
    train_err3 = []

    for i in train_bank:
        train_err1.append(est_E_1.predict(i))
        train_err2.append(est_ME_1.predict(i))
        train_err3.append(est_GI_1.predict(i))
    for i in test_bank:
        test_err1.append(est_E_1.predict(i))
        test_err2.append(est_ME_1.predict(i))
        test_err3.append(est_GI_1.predict(i))

    test_percent1 = 0
    test_percent2 = 0
    test_percent3 = 0

    train_percent1 = 0
    train_percent2 = 0
    train_percent3 = 0

    for i in range(len(testing_truth)):
        if testing_truth[i] == test_err1[i]:
            test_percent1 += 1
        if testing_truth[i] == test_err2[i]:
            test_percent2 += 1
        if testing_truth[i] == test_err3[i]:
            test_percent3 += 1
        if training_truth[i] == train_err1[i]:
            train_percent1 += 1
        if training_truth[i] == train_err2[i]:
            train_percent2 += 1
        if training_truth[i] == train_err3[i]:
            train_percent3 += 1

    length = len(testing_truth)
    length2 = len(training_truth)
    print(test_percent1/length, train_percent1/length2, test_percent2/length,
          train_percent2/length2, test_percent3/length, train_percent3/length2)
