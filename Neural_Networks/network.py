import numpy as np
import pandas as pd
import math
import random

# random.seed(10)


def sigmoid(x):
    return 1/(1 + math.exp(-x))


class NN:

    def __init__(self, width, depth, data, weightgen):
        # self.X = data[:,-1]
        # self.Y=data[:-1]
        start = data.shape[1] - 1
        self.data = data
        self.hidden = []
        for i in range(depth):
            layer = [Node(sigmoid, value=1)]
            # This is the bias node
            for j in range(width-1):
                layer.append(Node(sigmoid))
            self.hidden.append(layer)
            if i > 0:
                for parent_node in self.hidden[i-1]:
                    for j in range(1, width):
                        child = self.hidden[i][j]
                        child.addParent(
                            parent_node, weight=weightgen())
                        parent_node.addChild(child)

        self.start_layer = [Node(sigmoid, value=1)]
        for _ in range(start):
            self.start_layer.append(Node(sigmoid))
        for parent_node in self.start_layer:
            for i in range(1, width):
                child = self.hidden[0][i]
                child.addParent(parent_node, weight=weightgen())
                parent_node.addChild(child)
                # return

        # self.end_layer = []
        # for _ in range(end):
        #     self.end_layer.append(Node(lambda a: a))
        self.end = Node(lambda a: a)
        for parent_node in self.hidden[depth-1]:
            self.end.addParent(parent_node, weight=weightgen())
            parent_node.addChild(self.end)

    def forwardPass(self, weights):
        for i in range(1, len(self.start_layer)):
            self.start_layer[i].value = weights[i-1]
        for row in self.hidden:
            for node in row:
                node.calculate()

        self.end.calculate()
        return self.end.value

    def backProp(self, row, rate):
        memo = {}
        end = self.forwardPass(row[: -1])
        L = round(end) - row[-1]
        # print("L", L)
        memo[self.end] = L
        nodeweights = {}
        temp = []
        for parent in self.end.parents:
            partial = L * parent.value
            temp.append(partial)
            memo[parent] = partial
        nodeweights[self.end] = temp

        for i in range(1, len(self.hidden) + 1):
            j = 0
            for thisnode in self.hidden[-i]:
                temp = []
                incomplete = memo[thisnode] * \
                    thisnode.value * (1-thisnode.value)
                for parent in thisnode.parents:
                    partial = parent.value * incomplete
                    temp.append(partial)
                    if memo.__contains__(parent):
                        memo[parent] += partial
                    else:
                        memo[parent] = partial
                nodeweights[thisnode] = temp
                j += 1
        # print(nodeweights)
        for node, weight in nodeweights.items():
            node.weights = node.weights - np.multiply(rate, weight)

    def train(self, epochs, r, d):
        for i in range(epochs):
            rate = r/(1 + r * i/d)
            data = self.data.sample(frac=1).to_numpy()
            for row in data:
                # print(row)
                self.backProp(row, rate)
            # print(data)

    def predict(self, row):
        return round(self.forwardPass(row))
        # if self.end.value > .5:
        #     return 1
        # return 0


class Node:
    def __init__(self, function, value=None):
        self.parents = []
        self.children = []
        self.weights = []
        self.value = value
        self.function = function

    def addParent(self, node, weight):
        self.parents.append(node)
        self.weights.append(weight)

    def addChild(self, node):
        self.children.append(node)

    def calculate(self):
        if not self.parents:
            return

        summation = 0
        for i in range(len(self.parents)):
            summation += self.parents[i].value * self.weights[i]
        self.value = self.function(summation)
