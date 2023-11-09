"""
Test file for perceptron
"""
import pandas as pd
from perceptron import Perceptron, VotedPerceptron, AveragePerceptron


if __name__ == "__main__":
    frame = pd.read_csv("bank-note-1/test.csv", header=None)
    ave = AveragePerceptron(frame)
    vote = VotedPerceptron(frame)
    perc = Perceptron(frame)
    perc.learn(0.25, 10)
    ave.learn(0.25, 10)
    vote.learn(0.25, 10)
    frame = frame.to_numpy()
    aveAccuracy = 0
    voteAccuracy = 0
    percAccuracy = 0
    count = 0
    for row in frame:
        x = row[:-1]
        y = row[-1]
        if ave.predict(x) != y:
            aveAccuracy += 1
        if vote.predict(x) != y:
            voteAccuracy += 1
        if perc.predict(x) != y:
            percAccuracy += 1

        count += 1
    print("Base Perceptron error, r = .25, epochs = 10:", percAccuracy / count)
    print("Voted Perceptron error, r = .25, epochs = 10:", voteAccuracy / count)
    print("Average Perceptron error, r = .25, epochs = 10:", aveAccuracy / count)
    print(
        "Base Perceptron final weight vector, r =.25, epochs = 10:",
        perc.returnWeights(),
    )
    print("voted Perceptron final weight vectors and counts, r =.25, epochs = 10:")
    vects = vote.returnWeights()
    for weight, count in vects:
        print(weight, count)
    print(
        "Averaged Perceptron final weight vector, r =.25, epochs = 10:",
        ave.returnWeights(),
    )
