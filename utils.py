import numpy as np


def sigmoid(z: np.array) -> np.array:
    return 1 / (1 + np.exp(-z))


def paddingOnes(X: np.array) -> np.array:
    ones = np.array([[1 for _ in range(len(X))]])
    return np.concatenate((ones.T, X), axis=1)
