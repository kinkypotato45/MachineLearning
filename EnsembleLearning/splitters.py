import math

# def majority_err_splitter(column):
#     counts = dict()
#     if len(column) == 0:
#         return 1
#     labels = 0
#     for i in column:
#         if not counts.__contains__(i):
#             counts[i] = 0
#             labels += 1
#         counts[i] += 1
#     maximum = 0
#     if labels == 1:
#         return 0
#     for i in counts:
#         if counts[i] > maximum:
#             maximum = counts[i]
#     return 1 - maximum / len(column)

# def gini_splitter(column):
#     counts = dict()
#     labels = 0
#     for i in column:
#         if not counts.__contains__(i):
#             counts[i] = 0
#             labels += 1
#         counts[i] += 1
#     elements = len(column)
#     y_gini = 1
#     if labels == 1:
#         return 0
#     for i in counts:
#         y_gini -= (counts[i] / elements) ** 2
#     return y_gini


def info_gain_splitter(column, weights):
    size = 0
    results = {}
    labels = 0
    for i in range(len(column)):
        size += weights[i]
        if not results.__contains__(column[i]):
            results[i] = 0
            labels += 1
        results[i] += weights[i]
    entropy = 0

    if labels == 1:
        return 0

    for i in results:
        entropy += -results[i] / size * \
            (math.log(results[i] / size) / math.log(labels))
    return entropy
