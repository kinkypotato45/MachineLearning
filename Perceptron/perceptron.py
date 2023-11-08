"""
A library implementing the perceptron algorithms. Ни пуха, ни пера!
"""
import pandas as pd
import numpy as np
from pandas.core.indexes.api import Axis

frame = pd.read_csv("bank-note-1/train.csv", header=None)
print(frame)
print(len(frame.columns))


class Perceptron:
    def __init__(self, frame, r):
        """
        Takes in a dataframe, and create a linear classifier

        Parameters
        ----------
        Frame: a pandas dataframe, where each column represents real value,
        and the last column is a binary classfication.

        r: the scale of how much to change weight vector w
        """
        self.weights = [0] * len(frame.columns)
        self.frame = frame.clone()
