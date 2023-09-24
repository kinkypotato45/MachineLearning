import math


def majority_err_splitter(column):
    counts = dict()
    labels = 0
    for i in column:
        if not counts.__contains__(i):
            counts[i] = 0
            labels += 1
        counts[i] += 1
    maximum = 0
    if labels == 1:
        return 0
    for i in counts:
        if counts[i] > maximum:
            maximum = counts[i]
    return 1 - maximum/len(column)


def gini_splitter(column):
    counts = dict()
    labels = 0
    for i in column:
        if not counts.__contains__(i):
            counts[i] = 0
            labels += 1
        counts[i] += 1
    elements = len(column)
    y_gini = 1
    if labels == 1:
        return 0
    for i in counts:
        y_gini -= (counts[i]/elements)**2
    return y_gini


def info_gain_splitter(column):
    n = 0
    results = dict()
    labels = 0
    for i in column:
        n += 1
        if not results.__contains__(i):
            results[i] = 0
            labels += 1
        results[i] += 1
    entropy = 0

    if labels == 1:
        return 0

    for i in results:
        entropy += -results[i]/n * \
            (math.log(results[i]/n)/math.log(labels))
    return entropy
