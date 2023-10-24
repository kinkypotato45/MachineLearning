from numpy.lib import median
import splitters
import pandas as pd
import numpy as np
import copy


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
    if len(set(labelrow)) == 1:
        return -1
    for row in range(len(inputArray)-1):
        columnvars = {}
        entropysum = 0
        for column in range(len(inputArray[row])):
            if not columnvars.__contains__(inputArray[row][column]):
                # or inputArray[row][column] == "unknown":
                columnvars[inputArray[row][column]] = []
            columnvars[inputArray[row][column]].append(labelrow[column])
        for i in columnvars:
            subentropy = gain_function(columnvars[i])
            entropysum += len(columnvars[i])/len(labelrow)*subentropy
        # print(gain_function(inputArray[len(inputArray)-1]) - entropysum)
        information.append(entropysum)
    indexofmin = information.index(min(information))
    return indexofmin


class node:
    def __init__(self, inputArray, maxDepth, calculator):
        self.result = None
        self.splitValue = None
        self.children = {}
        self.inputArray = inputArray
        if maxDepth == 0:
            # resultArr = inputArray[:, -1]
            # self.result = max(
            #     set(resultArr), key=inputArray[:, -1].count())
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
        self.splitValue = weighted_gain(
            inputArray, calculator)
        if self.splitValue == -1:
            self.result = self.inputArray[:, -1][0]
            return
        frame = pd.DataFrame(inputArray)
        # if self.medianValue is None:
        groups = frame.groupby(self.splitValue)
        splitValueValues = set(inputArray[:, self.splitValue])
        for i in splitValueValues:
            # arr = groups.get_group(i).to_numpy()
            self.children[i] = node(groups.get_group(
                i).to_numpy(), maxDepth - 1, calculator)

    def predict(self, vector):
        if self.result is not None:
            return self.result
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
        self.frame = pd.DataFrame(data=inputArray, columns=None)
        # print(self.frame)
        self.medians = []
        for column in self.frame.columns:
            if pd.api.types.is_numeric_dtype(self.frame[column]):
                medianValue = self.frame[column].median()
                self.frame[column] = (
                    self.frame[column] > medianValue).astype(int)
                self.medians.append(medianValue)        # match calculator:
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
        self.root = node(self.frame.to_numpy(), maxDepth, self.calc)

    def predict(self, vec):
        vect = copy.copy(vec)
        for i in range(len(vect)):
            if self.medians[i] is not None:
                vect[i] = 1 if vect[i] > self.medians[i] else 0
        # print(vect)
        return self.root.predict(vect)


#
table = np.array([["s", "h", "h", "w", "-"],
                  ["s", "h", "h", "s", "-"],
                  ["o", "h", "h", "w", "+"],
                  ["r", "m", "h", "w", "+"],
                  ["r", "c", "n", "w", "+"],
                  ["r", "c", "n", "s", "-"],
                  ["o", "c", "n", "s", "+"],
                  ["s", "m", "h", "w", "-"],
                  ["s", "c", "n", "w", "+"],
                  ["r", "m", "n", "w", "+"],
                  ["s", "m", "n", "s", "+"],
                  ["o", "m", "h", "s", "+"],
                  ["o", "h", "n", "w", "+"],
                  ["r", "m", "h", "s", "-"],
                  ["o", "m", "n", "w", "+"],


                  ])

# weighted_gain(table, splitters.majority_err_splitter)
# print(splitters.info_gain_splitter(table[:, 4]))
tree = build_tree(table, 4, "E")
print(tree.predict(["s", "m", "h", "s"]))
# print(tree.predict(["o", "m", "h", "s"]))
# df = pd.DataFrame(
#     table)
# groups = df.groupby(0)
# sunny = groups.get_group("s")
# rainy = groups.get_group("r")
# print(sunny)
#
# tree = build_tree(table, 5, "E")
