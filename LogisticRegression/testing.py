# import network.py
import numpy as np
import pandas as pd

from LogisticRegression import logistic
from MLE import MLE


def main():
    training = pd.read_csv('data/train.csv', header=None)
    nptrain = training.to_numpy()
    testing = pd.read_csv('data/test.csv', header=None)
    nptest = testing.to_numpy()
    vars = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
    # networks = {}
    print(
        'question 2.3a\n Training and testing error for log regression for weigths set to zero, \n'
    )
    for i in vars:
        print('error for variance', i)
        logs = logistic(i, training)
        logs.train(8, 0.0025, .5)
        count = 0
        err = 0
        for row in nptrain:
            count += 1
            if logs.predict(row[:-1]) != row[-1]:
                err += 1
            # print(network.predict(row[:-1]), row[-1])
        print('training error:', err / count)
        count = 0
        err = 0
        for row in nptest:
            count += 1
            if logs.predict(row[:-1]) != row[-1]:
                err += 1
            # print(network.predict(row[:-1]), row[-1])
        print('testing error:', err / count, '\n')

    print('question 2.3b\n Training and testing error for MLE (LMS regression) \n')
    mle = MLE(training)
    mle.train(8, .0025, .5)
    count = 0
    err = 0
    for row in nptrain:
        count += 1
        if mle.predict(row[:-1]) != row[-1]:
            err += 1
        # print(network.predict(row[:-1]), row[-1])
    print('training error:', err / count)
    count = 0
    err = 0
    for row in nptest:
        count += 1
        if mle.predict(row[:-1]) != row[-1]:
            err += 1
        # print(network.predict(row[:-1]), row[-1])
    print('testing error:', err / count, '\n')


if __name__ == '__main__':
    main()
