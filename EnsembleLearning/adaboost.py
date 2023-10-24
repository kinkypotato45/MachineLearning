"""File for Adaboosting class."""
import copy
import math
import splitters
import pandas as pd
import numpy as np


def weighted_gain(input_array, gain_function, weights):
    """
    Use to calculate the best attribute to split for other methods.

    :param input_array: the table of inputs and labels
    :param gain_function: the function used to calculate gain
    :param splits: columns which have already been split
    :return: returns the column on which the data is to be split, according to the gain function
    """
    input_array = input_array.transpose()
    labelrow = input_array[len(input_array)-1]
    information = []
    if len(set(labelrow)) == 1:
        return -1
    for row in range(len(input_array)-1):
        columnvars = {}
        entropysum = 0
        for column in range(len(input_array[row])):
            if not columnvars.__contains__(input_array[row][column]):
                # or input_array[row][column] == "unknown":
                columnvars[input_array[row][column]] = []
            columnvars[input_array[row][column]].append(labelrow[column])
        for i in columnvars.values():
            subentropy = gain_function(i)
            entropysum += len(i)/len(labelrow)*subentropy
        # print(gain_function(input_array[len(inputArray)-1]) - entropysum)
        information.append(entropysum)
    indexofmin = information.index(min(information))
    return indexofmin


class Node:
    """Node class for constructing tree."""

    def __init__(self, input_array, max_depth, calculator, weights):
        """
        Take three arguments to create tree.

        input_array:array to build tree
        max_depth: max depth tree can build
        calculator: heuristic used to split data
        """
        self.result = None
        self.split_value = None
        self.children = {}
        self.input_array = input_array
        self.weights = weights
        if max_depth == 0:
            # resultArr = input_array[:, -1]
            # self.result = max(
            #     set(resultArr), key=input_array[:, -1].count())
            counter = {}
            for i in input_array[:, -1]:
                if not counter.__contains__(i):
                    counter[i] = 0
                counter[i] += 1
            maxvalue = 0
            for i in counter:
                if counter[i] > maxvalue:
                    maxvalue = counter[i]
                    self.result = i
            return
        self.split_value = weighted_gain(
            input_array, calculator, weights)
        if self.split_value == -1:
            self.result = self.input_array[:, -1][0]
            return
        frame = pd.DataFrame(input_array)
        # if self.median_value is None:
        groups = frame.groupby(self.split_value)
        split_value_values = set(input_array[:, self.split_value])
        for i in split_value_values:
            # arr = groups.get_group(i).to_numpy()
            self.children[i] = Node(groups.get_group(
                i).to_numpy(), max_depth - 1, calculator)

    def predict(self, vector):
        """
        Predict variables using the tree.

        vector: a vector of the same dimension with attribute unknown
        """
        if self.result is not None:
            return self.result
        if not self.children.__contains__(vector[self.split_value]):
            common = None
            frequency = {}
            max_value = 0
            for i in self.input_array[:, self.split_value]:
                if not frequency.__contains__(i):
                    frequency[i] = 0
                frequency[i] += 1
            for i in frequency.keys():
                if frequency[i] > max_value:
                    max_value = frequency[i]
                    common = i
            return self.children[common].predict(vector)
        return self.children[vector[self.split_value]].predict(vector)


class BuildTree:
    """Use this to build a decision tree."""

    def __init__(self, input_array, max_depth, calculator, weights):
        """
        Build your tree using this.

        args:
        input_array: any input array with last values to be calcuated.
        max_depth: the max height of tree; may terminated before reached
        calculator: heuristic used to calucate split column;
        E for entropy, GI for gini and
        """
        self.frame = pd.DataFrame(data=input_array, columns=None)
        # print(self.frame)
        self.medians = []
        self.weights = weights
        for column in self.frame.columns:
            if pd.api.types.is_numeric_dtype(self.frame[column]):
                median_value = self.frame[column].median()
                self.frame[column] = (
                    self.frame[column] > median_value).astype(int)
                self.medians.append(median_value)        # match calculator:
            else:
                self.medians.append(None)
        # print(self.frame)
        # print(self.frame.to_numpy())
        match calculator:
            case "E":
                self.calc = splitters.info_gain_splitter
            case "ME":
                self.calc = splitters.majority_err_splitter
            case "GI":
                self.calc = splitters.gini_splitter
            case "None":
                self.calc = splitters.info_gain_splitter
        # print(self.frame.to_numpy()[0])
        # print(self.medians)
        self.root = Node(self.frame.to_numpy(), max_depth, self.calc, weights)

    def predict(self, vec):
        """
        Predict an attributed using a vector of attributes.

        argumnts:
        vec: a vector of the same dimension of features as the test setjk
        """
        vect = copy.copy(vec)
        for i in range(len(vect)):
            if self.medians[i] is not None:
                vect[i] = 1 if vect[i] > self.medians[i] else 0
        # print(vect)
        return self.root.predict(vect)


class AdaBoost:
    """Constructor meant to build a strong classifier from weak ones."""

    def __init__(self, table, iterations, calculator):
        self.weak_classifiers = []
        self.data = pd.DataFrame(table)
        self.attr = self.data[self.data.columns[:-1]]
        self.labels = self.data[self.data.columns[-1]]
        self.weights = [1/len(table)]*len(table)
        for _ in range(iterations):
            tree = BuildTree(table, 1, calculator=calculator, self.weights)
            err = 0
            correct = []
            for i in range(self.labels.len()):
                if tree.predict(self.attr[i]) == self.labels[i]:
                    err += self.weights[i]
                    correct.append(1)
                else:
                    correct.append(-1)
            if err >= .5:
                return
            alpha = 1/2 * math.log(1/err - 1)
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] * \
                    math.exp(- alpha * correct[i])
            self.weights = np.norm(self.weights)
