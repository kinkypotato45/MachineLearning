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
    labeldict = set()
    labelscount = 0
    for i in inputArray[len(inputArray) - 1]:
        if not labeldict.__contains__(i):
            labeldict.add(i)
    labels = len(labeldict)
    labelrow = inputArray[len(inputArray)-1]
    print(gain_function(inputArray[len(inputArray)-1]))
    information = []
    if labelscount == 1:
        return -1, [], labels
    for row in range(0, len(inputArray)-1):
        columnvars = dict()
        for column in range(0, len(inputArray[row])):

            if not columnvars.__contains__(inputArray[row][column]):
                columnvars[inputArray[row][column]] = []
            columnvars[inputArray[row][column]].append(labelrow[column])
        entropysum = 0
        for i in columnvars:
            subentropy = gain_function(columnvars[i])
            entropysum += len(columnvars[i])/len(labelrow)*subentropy
        information.append(entropysum)
        print(gain_function(inputArray[len(inputArray)-1]) - entropysum)
    indexOfMin = information.index(min(information))
    return indexOfMin


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
                  ["r", "m", "n", "w", "+"],
                  ])
df = pd.DataFrame(
    table, columns=['outlook', 'temperature', 'humidity', 'wind', 'play'])
groups = df.groupby(['outlook'])
sunny = groups.get_group("s")
rainy = groups.get_group("r")

print(sunny)
# print(weighted_gain(df.to_numpy(), splitters.gini_splitter, ))
print(weighted_gain(rainy.to_numpy(), splitters.info_gain_splitter))
# sunnyhumid = sunny.groupby(['humidity'])
# print(weighted_gain(rainy.to_numpy(), splitters.gini_splitter))
# print(sunnyhumid.get_group("n"))
