from typing import Tuple
import numpy as np
from utils import sigmoid


def forwardprop(theta1: np.array, theta2: np.array, X: np.array) -> Tuple:
    z2 = np.dot(X, theta1.T)
    a2 = sigmoid(z2)
    ones = np.array([[1 for _ in range(len(a2))]])
    a2 = np.concatenate((ones.T, a2), axis=1)

    z3 = np.dot(a2, theta2.T)
    a3 = sigmoid(z3)

    return a3
