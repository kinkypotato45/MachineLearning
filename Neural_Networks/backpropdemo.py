import numpy as np
import pandas as pd
import math


def sigmoid(x):
    return 1/(1 + math.exp(-x))


class NN:

    def __init__(self, width, depth, start,):
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
                        child.addParent(parent_node, weight=1)
                        parent_node.addChild(child)

        self.start_layer = [Node(sigmoid, value=1)]
        for _ in range(start):
            self.start_layer.append(Node(sigmoid))
        for parent_node in self.start_layer:
            for i in range(1, width):
                child = self.hidden[0][i]
                child.addParent(parent_node, weight=1)
                parent_node.addChild(child)
                # return

                # print(self.hidden)
        # self.end_layer = []
        # for _ in range(end):
        #     self.end_layer.append(Node(lambda a: a))
        self.end = Node(lambda a: a)
        for parent_node in self.hidden[depth-1]:
            self.end.addParent(parent_node, weight=1)
            parent_node.addChild(self.end)

    def forwardPass(self, weights):
        for i in range(1, len(self.start_layer)):
            self.start_layer[i].value = weights[i-1]
        for row in self.hidden:
            for node in row:
                node.calculate()
        self.end.calculate()

    def backProp(self, y, rate):
        memo = {}
        weights = {}
        L = self.end.value - y
        memo[self.end] = L
        nodeweights = {}
        temp = []
        for parent in self.end.parents:
            partial = L * parent.value
            temp.append(partial)
            memo[parent] = partial
        nodeweights[self.end] = temp
        print("layer 3: y weights partials:", temp)

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
                print("layer ", len(self.hidden) + 1 - i,
                      " node ", j, " partials:", temp)
                j += 1
            # for thisnode in self.hidden[-i-1]:

                # print(i)
                # print(thisnode.value)
                # if not nodeweights.__contains__(node):
                # nodeweights[node] = []
                # incomplete = memo[node]*node.value*(1-node.value)
                # for parent in node.parents:
                #     partial = incomplete * parent.value
                #     print(partial)
                #     nodeweights[node].append(partial)


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
        # print(self.weights)

    def addChild(self, node):
        self.children.append(node)

    def calculate(self):
        if self.value is None:
            summation = 0
            for i in range(len(self.parents)):
                summation += self.parents[i].value * self.weights[i]
            self.value = self.function(summation)
            # print(self.value)


network = NN(3, 2, 2, )
network.hidden[0][1].weights = [-1, -2, -3]
network.hidden[0][2].weights = [1, 2, 3]
network.hidden[1][1].weights = [-1, -2, -3]
network.hidden[1][2].weights = [1, 2, 3]
network.end.weights = [-1, 2, -1.5]

network.forwardPass([1, 1])
network.backProp(1, 0)
