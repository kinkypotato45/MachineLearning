import pandas as pd
import adaboost

testing = pd.read_csv("data/bank/test.csv", header=None)
training = pd.read_csv("data/bank/train.csv", header=None)

training_attr = training[training.columns[:-1]].to_numpy()
training_values = training[training.columns[-1]].to_numpy()


testing_attr = testing[testing.columns[:-1]].to_numpy()
testing_values = testing[testing.columns[-1]].to_numpy()

# boost = AdaBoost(training, 100)
# boost.predict(testing_attr[0])
min_it = 100
max_it = 501
steps = 10
print("Adaboosting: iterations, training errors, testing errors")

for i in range(10, 200, 10):
    # print(i)
    boosting = adaboost.AdaBoost(training, i)
    train_count = 0
    train_correct = 0
    test_count = 0
    test_correct = 0
    for j in range(len(training_attr)):
        train_count += 1
        if boosting.predict(training_attr[j]) == training_values[j]:
            train_correct += 1

    for j in range(len(testing_attr)):
        test_count += 1
        if boosting.predict(testing_attr[j]) == testing_values[j]:
            test_correct += 1
    print(i, train_correct / train_count, test_correct / test_count)
