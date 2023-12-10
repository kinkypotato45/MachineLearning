import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

dataset = np.loadtxt("data/train.csv", delimiter=",")
# print(dataset)
X = torch.tensor(dataset[:, 0:4], dtype=torch.float32)
Y = torch.tensor(dataset[:, 4], dtype=torch.float32)
print(X)
print(Y)
