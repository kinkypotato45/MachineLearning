from svm import primalSVM
import pandas as pd
import numpy as np


def runTest(C, gamma, a, epochs):
    print("C = ", C, "gamma = ", gamma, "a = ", a, "epochs = ", epochs)
    test = pd.read_csv("bank-note-1/test.csv", header=None).to_numpy()
    train = pd.read_csv("bank-note-1/train.csv", header=None).to_numpy()

    svm = primalSVM(train)
    svm.learn(C=C, gamma=gamma, a=a, epochs=epochs)
    # testing = primalSVM(train)

    testerr = 0
    trainerr = 0
    for row in test:
        if row[-1] != svm.predict(row[:-1]):
            testerr += 1
    for row in train:
        if row[-1] != svm.predict(row[:-1]):
            trainerr += 1
    print("testing error: ", testerr / len(test))
    print("training error: ", trainerr / len(train))


def main():
    C = [100 / 873, 500 / 873, 700 / 873]
    for c in C:
        runTest(C=c, gamma=0.01, a=c, epochs=100)
    # test = pd.read_csv("ban-note-1/test.csv", header=None).to_numpy()
    # train = pd.read_csv("bank-note-1/train.csv", header=None).to_numpy()
    # this = VotedPerceptron(frame)
    # values = this.learn(0.5, 1)
    # train1 = primalSVM(train)
    # train2 = primalSVM(train)
    # train3 = primalSVM(train)
    #
    # test1 = primalSVM(test)
    # test2 = primalSVM(test)
    # test3 = primalSVM(test)

    # test1.learn(C=100 / 873, gamma=0.05, a=200 / 873, epochs=100)
    # count = 0
    # for i in range(len(test)):
    #     row = test[i]
    #     # print(row)
    #     x = row[:-1]
    #     y = row[-1]
    #     if test1.predict(x) == y:
    #         count += 1
    #     # print(frame[i])
    #     # print(svm.predict(x), y)
    # print(count)


if __name__ == "__main__":
    main()
