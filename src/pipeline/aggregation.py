import numpy as np

def trim_mean(values, proportion_to_cut=0.1):
    values = np.array(values)
    values = np.sort(values)
    n = len(values)
    k = int(n * proportion_to_cut)
    if 2 * k >= n:
        return float(np.mean(values))
    return float(np.mean(values[k : n - k]))

def summarize_emotions(prob_matrix, labels):
    means = prob_matrix.mean(axis=0)
    return dict(zip(labels, means.tolist()))
