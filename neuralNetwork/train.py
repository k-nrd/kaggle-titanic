from typing import Callable
import numpy as np
from scipy.optimize import fmin_cg, fmin_ncg

from neuralNetwork.functions import forwardProp
from neuralNetwork.utils import initializeWeights, meltParams, setupVars, reshapeParams


def train(
    X: np.array,
    y: np.array,
    hiddenLayerSize: int,
    costFunction: Callable,
    gradient: Callable,
    lmbda: float = 0,
) -> np.array:
    (m, n, k, y) = setupVars(X, y)
    initialTheta1 = initializeWeights(n, hiddenLayerSize)
    initialTheta2 = initializeWeights(hiddenLayerSize + 1, k)
    initialNNParams = meltParams(initialTheta1, initialTheta2)

    resCG = fmin_cg(
        costFunction,
        initialNNParams,
        gradient,
        (hiddenLayerSize, X, y, lmbda),
        maxiter=1000,
    )

    print("CG optimization results:\n", resCG)
    print(
        "--------------------------------------------------------------------------------"
    )

    return resCG


def classify(nnParams: np.array, hiddenLayerSize: int, k: int, X: np.array) -> np.array:
    n = X.shape[1]

    (theta1, theta2) = reshapeParams(nnParams, n, hiddenLayerSize, k)
    (h, _) = forwardProp(theta1, theta2, X)

    if k == 1:
        h = h.ravel()
        h[h >= 0.5] = 1
        h[h < 0.5] = 0
        return h.astype(int)

    maxValues = np.argmax(h, axis=1) + 1
    return maxValues.astype(int)
