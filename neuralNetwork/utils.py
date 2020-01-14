from typing import Tuple
import numpy as np

from utils import sigmoid


def setupVars(X, y):
    m = X.shape[0]
    n = X.shape[1]
    y = y.reshape((len(y), 1)) if len(y.shape) == 1 else y
    k = y.shape[1]
    return (m, n, k, y)


def meltParams(theta1: np.array, theta2: np.array) -> np:
    return np.concatenate((theta1.ravel(), theta2.ravel()))


def reshapeParams(nnParams: np.array, n: int, hiddenLayerSize: int, k: int) -> Tuple:
    st1 = hiddenLayerSize * n
    theta1 = nnParams[:st1].reshape((hiddenLayerSize, n))
    theta2 = nnParams[st1:].reshape((k, hiddenLayerSize + 1))
    return (theta1, theta2)


def sigmoidGradient(z: np.array) -> np.array:
    return sigmoid(z) * (1 - sigmoid(z))


def initializeWeights(m: int, n: int) -> np.array:
    return np.random.rand(m, n) * 2 * 0.12 - 0.12
