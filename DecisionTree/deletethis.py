import pandas as pd
import treebuilder

frame = pd.read_csv("bank/train.csv")
test = pd.read_csv("bank/test.csv")
# print(frame)
testing = test[test.columns[:-1]].to_numpy()
testing_results = test[test.columns[-1]].to_numpy()

train_results = frame[frame.columns[-1]].to_numpy()
training = frame[frame.columns[:-1]].to_numpy()

tree = treebuilder.BuildTree(frame, 1, "E")
Count = 0
for i in range(len(testing)):
    if tree.predict(testing[i]) == testing_results[i]:
        Count += 1

print(Count / len(training))
