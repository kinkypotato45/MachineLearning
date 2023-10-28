from numpy import matrix
import pandas as pd
from LMS_regression import LMS_regression

frame = pd.read_csv("data/concrete/train.csv", header=None)
# print(frame)

regression = LMS_regression(matrix=frame.to_numpy(), learning_rate=0.01)
print("MSE before grad descent: r = .01", regression.MSE())
totals = 0
while regression.changedist > 10**-6:
    regression.descend(steps=25)
    totals += 25
    print(totals, regression.MSE(), regression.changedist)
# print("for Batched gradient descent: r = .01, w =", regression.weights, "converges")
print("MSE after descent:", regression.MSE())

print(regression.weights)
# regression2 = LMS_regression(matrix=frame.to_numpy(), sample_size=1, learning_rate=0.01)
# print("Stochastic MSE before grad descent", regression2.MSE())
# regression2.descend()
# print("for Stochastic gradient descent: r = .01, w =", regression2.weights, "converges")
# print("MSE after descent:", regression2.MSE())
