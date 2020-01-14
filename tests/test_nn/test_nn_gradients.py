from typing import Callable
import numpy as np
import pandas as pd
from neuralNetwork.functions import (
    costFunction,
    regularizedBackprop,
)
from neuralNetwork.utils import initializeWeights
from utils import paddingOnes


def numericalGradient(
    costFunction: Callable, nnParams: np.array, epsilon: float = 0.0001
) -> np.array:
    num_grad = np.zeros(nnParams.shape)
    perturb = np.zeros(nnParams.shape)
    for i in range(len(nnParams)):
        perturb[i] = epsilon
        loss1 = costFunction(nnParams - perturb)
        loss2 = costFunction(nnParams + perturb)
        num_grad[i] = (loss2 - loss1) / (2 * epsilon)
        perturb[i] = 0
    return num_grad


def checkNNGradients(lmbda=0):
    n = 3
    sl2 = 5
    k = 3
    m = 5

    theta1 = initializeWeights(sl2, n + 1)
    theta2 = initializeWeights(k, sl2 + 1)
    # reusing initializeWeights to generate X
    X = initializeWeights(m, n)
    X = paddingOnes(X)
    y = [i % k for i in range(m)]
    y = pd.get_dummies(y).to_numpy()

    nnParams = np.concatenate((theta1.ravel(), theta2.ravel()))

    def checkCostFunction(nnParams: np.array) -> float:
        return costFunction(nnParams, hiddenLayerSize=sl2, X=X, y=y, lmbda=lmbda)

    grad = regularizedBackprop(nnParams, sl2, X, y, lmbda)
    num_grad = numericalGradient(checkCostFunction, nnParams)

    print(np.c_[num_grad, grad])
    print("The above columns should be very similar.")
    print("Left: Numerical Gradient, Right: Analytical Gradient")

    # Assuming epsilon = 0.0001
    diff = np.linalg.norm(num_grad - grad) / np.linalg.norm(num_grad + grad)
    print(
        "If the backprop implementation is correct, relative difference shoud be < 1e-9."
    )
    print(f"Relative difference: {diff}")
