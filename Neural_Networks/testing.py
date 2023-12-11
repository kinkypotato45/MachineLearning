# import network.py
import pandas as pd
import numpy as np
from network import NN


def main():
    training = pd.read_csv('data/train.csv', header=None)
    nptrain = training.to_numpy()
    testing = pd.read_csv('data/test.csv', header=None)
    nptest = testing.to_numpy()
    widths = [5, 10, 25, 50, 100]
    # networks = {}
    print('question 2.2b\n Training and testing error for weight set to zero, \n')
    for i in widths:
        print('error for width', i, 'depth 2 NN:')
        thisnetwork = NN(i, 2, training, lambda: 0)
        thisnetwork.train(8, 0.025, 1)
        count = 0
        err = 0
        for row in nptrain:
            count += 1
            if thisnetwork.predict(row[:-1]) != row[-1]:
                err += 1
            # print(network.predict(row[:-1]), row[-1])
        print('training error:', err / count)
        count = 0
        err = 0
        for row in nptest:
            count += 1
            if thisnetwork.predict(row[:-1]) != row[-1]:
                err += 1
            # print(network.predict(row[:-1]), row[-1])
        print('testing error:', err / count, "\n")

    print('question 2.2c\nTraining and testing error for randomized weights from a Normal dist\n ')
    for i in widths:
        print('error for width', i, 'depth 2 NN:')
        thisnetwork = NN(i, 2, training, lambda: np.random.normal())
        thisnetwork.train(8, 0.025, 1)
        count = 0
        err = 0
        for row in nptrain:
            count += 1
            if thisnetwork.predict(row[:-1]) != row[-1]:
                err += 1
            # print(network.predict(row[:-1]), row[-1])
        print('training error:', err / count)
        count = 0
        err = 0
        for row in nptest:
            count += 1
            if thisnetwork.predict(row[:-1]) != row[-1]:
                err += 1
            # print(network.predict(row[:-1]), row[-1])
        print('testing error:', err / count, '\n')


if __name__ == '__main__':
    main()
