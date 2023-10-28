"""File for Adaboosting class."""
import math

import pandas as pd
import numpy as np
import copy


def info_gain_splitter(column, weights):
    # print(len(weights))
    size = sum(weights)
    results = {}
    labels = 0
    for i in range(len(column)):
        if not results.__contains__(column[i]):
            results[column[i]] = 0
            labels += 1
        results[column[i]] += weights[i]
    entropy = 0
    if labels == 1:
        return 0
    for i in results.values():
        entropy += -i * math.log(i / size) / math.log(labels)
    # print(entropy)

    # print("entropy", entropy)
    return entropy


def weighted_gain(input_array, weights):
    """
    Use to calculate the best attribute to split for other methods.

    :param input_array: the table of inputs and labels
    :param gain_function: the function used to calculate gain
    :param splits: columns which have already been split
    :return: returns the column on which the data is to be split, according to the gain function
    """
    input_array = input_array.transpose()
    labelrow = input_array[len(input_array) - 1]
    information = []
    if len(set(labelrow)) == 1:
        return -1
    for row in range(len(input_array) - 1):
        this_row = input_array[row]
        weights_value = {}
        # print(len(weights))
        for column in range(len(this_row)):
            if not weights_value.__contains__(this_row[column]):
                weights_value[this_row[column]] = []
            weights_value[this_row[column]].append([labelrow[column], weights[column]])
        gain = 0
        for value in weights_value.values():
            attrs = []
            temp_weights = []
            for i in value:
                attrs.append(i[0])
                temp_weights.append(i[1])

            gain += info_gain_splitter(attrs, temp_weights)
        information.append(gain)
    # print(information)
    # print(information.index(min(information)))

    return information.index(min(information))


class Node:
    """Node class for constructing tree."""

    def __init__(self, input_array, max_depth, weights):
        """
        Take three arguments to create tree.

        input_array:array to build tree
        max_depth: max depth tree can build
        calculator: heuristic used to split data
        """
        # (print(len(weights)),)
        # print(len(input_array))
        self.result = None
        self.split_value = None
        self.children = {}
        self.input_array = input_array
        self.weights = weights
        if max_depth == 0:
            counter = {}
            results = self.input_array[:, -1]
            # print(results)
            # print(results)
            for i in range(len(results)):
                if not counter.__contains__(results[i]):
                    counter[results[i]] = 0
                counter[results[i]] += weights[i]
            maxvalue = 0
            for i, j in counter.items():
                # print(i)
                # print(j)
                # print("j", j)
                if j > maxvalue:
                    maxvalue = j
                    self.result = i
            # print("result", self.result)
            return
        self.split_value = weighted_gain(input_array, weights)
        if self.split_value == -1:
            self.result = self.input_array[:, -1][0]
            return
        frame = pd.DataFrame(input_array)
        weight_frame = pd.DataFrame({"weights": weights})
        frame = frame.join(weight_frame)

        # if self.median_value is None:
        groups = frame.groupby(self.split_value)
        # split_value_values = set(input_array[:, self.split_value])
        for i, j in groups:
            one_group = j[j.columns[:-1]]
            these_weights = j[j.columns[-1]]
            self.children[i] = Node(
                one_group.to_numpy(), max_depth - 1, these_weights.to_numpy()
            )

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

    def __init__(self, input_array, max_depth, weights):
        """
        Build your tree using this.

        args:
        input_array: any input array with last values to be calcuated.
        max_depth: the max height of tree; may terminated before reached
        calculator: heuristic used to calucate split column;
        E for entropy, GI for gini and
        """
        # weights_frame = pd.DataFrame({"weights": weights})
        self.frame = pd.DataFrame(data=input_array, columns=None)
        self.medians = []
        self.weights = weights
        for column in self.frame.columns:
            if pd.api.types.is_numeric_dtype(self.frame[column]):
                median_value = self.frame[column].median()
                self.medians.append(median_value)
                self.frame[column] = (self.frame[column] > median_value).astype(int)
            else:
                self.medians.append(None)
        self.medians
        # print(self.medians)
        # print(self.frame)
        self.root = Node(self.frame.to_numpy(), max_depth, self.weights)

    def predict(self, vec):
        """
        Predict an attributed using a vector of attributes.

        argumnts:
        vec: a vector of the same dimension of features as the test setjk
        """
        vec = copy.copy(vec)

        for i in range(len(vec)):
            if self.medians[i] is not None:
                vec[i] = 1 if vec[i] > self.medians[i] else 0
        # print(vec)
        # print(vec[11])

        return self.root.predict(vec)


class AdaBoost:
    """Constructor meant to build a strong classifier from weak ones."""

    def __init__(self, table, iterations):
        self.weak_classifiers = []
        self.data = pd.DataFrame(table)
        self.attr = self.data[self.data.columns[:-1]].to_numpy()
        # print(self.data)
        self.labels = self.data[self.data.columns[-1]].to_numpy()
        self.weights = [1 / len(table)] * len(table)
        self.trees = []
        for _ in range(iterations):
            # print(self.weights)
            # print(np.linalg.norm(self.weights))
            # print(self.labels)
            tree = BuildTree(table, 1, self.weights)
            self.trees.append(tree)
            err = 0
            totalerr = 0
            count = 0
            # print(err)
            correct = []
            prediction = []
            new_weights = self.weights
            for i in range(len(self.labels)):
                prediction.append(tree.predict(self.attr[i]))
                count += 1
                # print(tree.predict(self.attr[i]))
                if tree.predict(self.attr[i]) != self.labels[i]:
                    # print(self.weights[i])
                    totalerr += 1
                    err += self.weights[i]
                    correct.append(-1)
                else:
                    correct.append(1)
            # print("err:", err)
            if err >= 0.5:
                return
            if err == 0:
                return
            alpha = 1 / 2 * math.log(1 / err - 1)
            self.weak_classifiers.append(alpha)
            for i in range(len(new_weights)):
                new_weights[i] = new_weights[i] * math.exp(-alpha * correct[i])
            norm = sum(new_weights)

            new_weights = [i / norm for i in new_weights]
            self.weights = new_weights

    def predict(self, vector):
        results = {}
        for i in range(len(self.trees)):
            tree = self.trees[i]
            prediction = tree.predict(vector)
            if not results.__contains__(prediction):
                results[prediction] = 0
            results[prediction] += self.weak_classifiers[i]
        maxvalue = None
        maximum = 0
        for (
            i,
            j,
        ) in results.items():
            if j > maximum:
                maximum = j
                maxvalue = i
        return maxvalue
