from typing import Tuple
import numpy as np
from utils import sigmoid, paddingOnes
from neuralNetwork.utils import (
    meltParams,
    sigmoidGradient,
    setupVars,
    reshapeParams,
)


def forwardProp(theta1: np.array, theta2: np.array, X: np.array) -> Tuple:
    z2 = np.dot(X, theta1.T)
    a2 = sigmoid(z2)
    a2 = paddingOnes(a2)

    z3 = np.dot(a2, theta2.T)
    a3 = sigmoid(z3)

    return (a3, a2)


def costFunction(
    nnParams: np.array,
    hiddenLayerSize: int,
    X: np.array,
    y: np.array,
    lmbda: float = 0,
) -> float:
    # Setup useful variables
    (m, n, k, y) = setupVars(X, y)
    (theta1, theta2) = reshapeParams(nnParams, n, hiddenLayerSize, k)
    (a3, _) = forwardProp(theta1, theta2, X)

    y_vec = y.ravel()
    h_vec = a3.ravel()

    J = ((-y_vec.dot(np.log(h_vec))) - ((1 - y_vec).dot(np.log(1 - h_vec)))) / m

    if lmbda > 0:
        multiplier = lmbda / (2 * m)
        t1Reg = theta1[:, 1:].ravel()
        t2Reg = theta2[:, 1:].ravel()
        J += multiplier * (t1Reg.dot(t1Reg) + t2Reg.dot(t2Reg))

    print(f"Current J: {J}")
    return J


def regularizedBackprop(nnParams, hiddenLayerSize, X, y, lmbda=0) -> np.array:
    (m, n, k, y) = setupVars(X, y)
    (theta1, theta2) = reshapeParams(nnParams, n, hiddenLayerSize, k)
    (a3, a2) = forwardProp(theta1, theta2, X)

    d3 = a3 - y

    z2 = X.dot(theta1.T)
    d2 = d3.dot(theta2)
    d2 = d2[:, 1:] * sigmoidGradient(z2)

    theta1_grad = (X.T.dot(d2)).T / m
    theta2_grad = (a2.T.dot(d3)).T / m
    if lmbda > 0:
        theta1_grad[:, 1:] = np.add(theta1_grad[:, 1:], (lmbda / m) * theta1[:, 1:])
        theta2_grad[:, 1:] = np.add(theta2_grad[:, 1:], (lmbda / m) * theta2[:, 1:])

    return meltParams(theta1_grad, theta2_grad)
