import numpy as np
import matplotlib.pyplot as plt


def f1_score(y_true, y_pred):
    """ returns f1_score of binary classification task with true labels y_true and predicted labels y_pred """
    true_positive = sum([true and pred for true, pred in zip(y_true, y_pred)])
    false_positive = sum([not true and pred for true, pred in zip(y_true, y_pred)])
    false_negative = sum([true and not pred for true, pred in zip(y_true, y_pred)])
    recall = true_positive/(true_positive + false_negative)
    precision = true_positive/(true_positive + false_positive)

    return 2*recall*precision/(recall + precision)


def rmse(y_true, y_pred):
    diff = y_true - y_pred
    return np.sqrt((diff @ diff)/len(diff))


def visualize_results(k_list, scores, metric_name, title, path):
    """ plot a results graph of cross validation scores """
    pass
