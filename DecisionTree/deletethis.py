import treebuilder
import pandas as pd

frame = pd.read_csv("bank-7/train.csv")
test = pd.read_csv('bank-7/test.csv')
# print(frame)
testing = test[test.columns[:-1]].to_numpy()
testing_results = test[test.columns[-1]].to_numpy()

train_results = frame[frame.columns[-1]].to_numpy()
training = frame[frame.columns[:-1]].to_numpy()

tree = treebuilder.build_tree(frame, 2, 'E')
count = 0
for i in range(len(training)):
    if tree.predict(training[i]) == train_results[i]:
        count += 1

print(count / len(training))
