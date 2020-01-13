import numpy as np
from neuralNetwork.forwardprop import forwardprop


def nnCostFunction(
    nnParams: np.array, hiddenLayerSize: int, X: np.array, y: np.array, lmbda: float,
) -> float:
    # Setup useful variables
    m = X.shape[0]
    n = X.shape[1]
    k = y.shape[1]

    sl2 = n * hiddenLayerSize
    theta1_vec = nnParams[:sl2]
    theta2_vec = nnParams[sl2:]

    theta1 = theta1_vec.reshape((hiddenLayerSize, n))
    theta2 = theta2_vec.reshape((k, hiddenLayerSize + 1))
    a3 = forwardprop(theta1, theta2, X)

    y_vec = y.ravel()
    h_vec = a3.ravel()

    J = ((-y_vec.dot(np.log(h_vec))) - ((1 - y_vec).dot(np.log(1 - h_vec)))) / m

    if lmbda > 0:
        multiplier = lmbda / (2 * m)
        summation = theta1_vec.dot(theta1_vec) + theta2_vec.dot(theta2_vec)
        J += multiplier * summation

    return J
