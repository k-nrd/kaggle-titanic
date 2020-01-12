import numpy as np


def sigmoid(z: np.array) -> np.array:
    g = 1 / (1 + np.exp(-z))
    return g


def classify(theta: np.array, X: np.array):
    p = sigmoid(X.dot(theta))
    p[p >= 0.5] = 1
    p[p < 0.5] = 0

    return p.astype(int)
