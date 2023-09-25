import splitters
import pandas as pd
import numpy as np


def weighted_gain(inputArray, gain_function):
    """
    :param inputArray: the table of inputs and labels
    :param gain_function: the function used to calculate gain
    :param splits: columns which have already been split
    :return: returns the column on which the data is to be split, according to the gain function
    """
    inputArray = inputArray.transpose()
    labelrow = inputArray[len(inputArray)-1]
    information = []
    medianRow = []
    if len(set(labelrow)) == 1:
        return -1, None
    for row in range(len(inputArray)-1):
        columnvars = {}
        medianValue = None
        entropysum = 0
        if inputArray[row][0].strip('-').isnumeric():
            # entropysum = 1
            # information.append(entropysum)
            # continue

            introw = []
            for i in inputArray[row]:
                introw.append(int(i))
            # for i in range(len(inputArray[row])):
            #     introw.append(int(inputArray[i]))
            medianValue = np.median(introw)
            columnvars[0] = []
            columnvars[1] = []

            for column in range(len(introw)):
                indicator = 0
                if int(inputArray[row][column]) > medianValue:
                    indicator += 1
                columnvars[indicator].append(labelrow[column])
        else:
            for column in range(len(inputArray[row])):
                if not columnvars.__contains__(inputArray[row][column]) or inputArray[row][column] == "unknown":
                    columnvars[inputArray[row][column]] = []
                columnvars[inputArray[row][column]].append(labelrow[column])
        medianRow.append(medianValue)
        for i in columnvars:
            subentropy = gain_function(columnvars[i])
            entropysum += len(columnvars[i])/len(labelrow)*subentropy
        information.append(entropysum)
    indexofmin = information.index(min(information))
    return indexofmin, medianRow[indexofmin]


class node:
    def __init__(self, inputArray, maxDepth, calculator):
        self.result = None
        self.splitValue = None
        self.children = {}
        self.inputArray = inputArray
        if maxDepth == 0:
            counter = {}
            for i in inputArray[:, -1]:
                if not counter.__contains__(i):
                    counter[i] = 0
                counter[i] += 1
            maxvalue = 0
            for i in counter:
                if counter[i] > maxvalue:
                    maxvalue = counter[i]
                    self.result = i
            return
        self.splitValue, self.medianValue = weighted_gain(
            inputArray, calculator)
        if self.splitValue == -1:
            self.result = self.inputArray[:, -1][0]
            return
        frame = pd.DataFrame(inputArray)
        if self.medianValue is None:
            groups = frame.groupby(self.splitValue)
            splitValueValues = set(inputArray[:, self.splitValue])
            for i in splitValueValues:
                arr = groups.get_group(i).to_numpy()
                self.children[i] = node(arr, maxDepth - 1, calculator)
        else:

            group = frame.groupby(
                frame[self.splitValue] < str(self.medianValue))
            try:
                self.children[0] = node(
                    group.get_group(True).to_numpy(), maxDepth-1, calculator)
            except KeyError:
                self.children[0] = node(
                    group.get_group(False).to_numpy(), maxDepth-1, calculator)
            try:
                self.children[1] = node(group.get_group(
                    False).to_numpy(), maxDepth-1, calculator)
            except KeyError:
                self.children[1] = node(group.get_group(
                    True).to_numpy(), maxDepth-1, calculator)

    def predict(self, vector):
        if self.result is not None:
            return self.result
        if self.medianValue is not None:
            if int(vector[self.splitValue]) > self.medianValue:
                return self.children[1].predict(vector)
            return self.children[0].predict(vector)
        if not self.children.__contains__(vector[self.splitValue]):
            common = None
            frequency = {}
            maxValue = 0
            for i in self.inputArray[:, self.splitValue]:
                if not frequency.__contains__(i):
                    frequency[i] = 0
                frequency[i] += 1
            for i in frequency:
                if frequency[i] > maxValue:
                    maxValue = frequency[i]
                    common = i
            return self.children[common].predict(vector)
        return self.children[vector[self.splitValue]].predict(vector)


class build_tree:
    def __init__(self, inputArray, maxDepth, calculator):
        match calculator:
            case "E":
                self.calc = splitters.info_gain_splitter
            case "ME":
                self.calc = splitters.majority_err_splitter
            case "GI":
                self.calc = splitters.gini_splitter
            case "None":
                self.calc = splitters.info_gain_splitter
        self.root = node(inputArray, maxDepth, self.calc)

    def predict(self, vec):
        return self.root.predict(vec)

#
# table = np.array([["s", "h", "h", "w", "-"],
#                   ["s", "h", "h", "s", "-"],
#                   ["o", "h", "h", "w", "+"],
#                   ["r", "m", "h", "w", "+"],
#                   ["r", "c", "n", "w", "+"],
#                   ["r", "c", "n", "s", "-"],
#                   ["o", "c", "n", "s", "+"],
#                   ["s", "m", "h", "w", "-"],
#                   ["s", "c", "n", "w", "+"],
#                   ["r", "m", "n", "w", "+"],
#                   ["s", "m", "n", "s", "+"],
#                   ["o", "m", "h", "s", "+"],
#                   ["o", "h", "n", "w", "+"],
#                   ["r", "m", "h", "s", "-"],
#                   ["r", "m", "n", "w", "+"],
#                   ])
# tree = build_tree(table, 6, "E")
# print(tree.predict(["s", "m", "h", "s"]))
# print(tree.predict(["o", "m", "h", "s"]))
# df = pd.DataFrame(
#     table)
# groups = df.groupby(0)
# sunny = groups.get_group("s")
# rainy = groups.get_group("r")
# print(sunny)
#
# tree = build_tree(table, 5, "E")
