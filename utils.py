import numpy as np


def sigmoid(z: np.array) -> np.array:
    return 1 / (1 + np.exp(-z))


def sigmoidGradient(z: np.array) -> np.array:
    return sigmoid(z) * (1 - sigmoid(z))


def classify(theta: np.array, X: np.array) -> np.array:
    p = sigmoid(X.dot(theta))
    p[p >= 0.5] = 1
    p[p < 0.5] = 0

    return p.astype(int)
