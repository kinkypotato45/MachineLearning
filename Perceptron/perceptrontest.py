"""
Test file for perceptron
"""
import pandas as pd
from perceptron import Perceptron

if __name__ == "__main__":
    frame = pd.read_csv("bank-note-1/train.csv", header=None)
    this = Perceptron(frame)
    this.learn(0.5, 1)
    frame = frame.to_numpy()
    for i in range(100):
        row = frame[i]
        x = row[:-1]
        y = row[-1]
        # print(frame[i])
        print(this.predict(x), y)
