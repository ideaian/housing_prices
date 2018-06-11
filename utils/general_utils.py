from sklearn import metrics
import numpy as np

def print_metrics(y, y_pred):
    print("Sqrt mse: {}".format(np.sqrt(metrics.mean_squared_error(y, y_pred))))
    print("Mean absolute error: {}".format(
        metrics.mean_absolute_error(y, y_pred)))
    print("R2 score: {}".format(metrics.r2_score(y, y_pred)))
    print("Absolute mean relative error: {}".format(
        abs_mean_relative_error(y, y_pred)))


def abs_mean_relative_error(y, y_pred):
    return np.mean(np.abs(y - y_pred) / (y))

