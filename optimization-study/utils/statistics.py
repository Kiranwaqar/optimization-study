import numpy as np


def mean_std(values):
    return np.mean(values), np.std(values)


def confidence_interval(values):
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values)
    ci = 1.96 * std / np.sqrt(len(values))  # 95% CI
    return mean, ci