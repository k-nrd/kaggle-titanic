import numpy as np
from .utils import sigmoid


def lrCostFunction(theta: np.array, X: np.array, y: np.array, lmbda: float) -> float:
    # get # of examples
    m = len(X)

    # calculate h(X)
    h = sigmoid(X.dot(theta))

    # calculate cost term and regularization term of cost function
    cost = (np.dot(-y, np.log(h)) - np.dot(1 - y, np.log(1 - h))) / m
    cost_reg_term = lmbda * np.power(theta[1:], 2).sum() / (2 * m)

    # calculate cost J
    J = cost + cost_reg_term

    print("Cost term: ", cost)
    print("Reg term: ", cost_reg_term)
    print("Cost J: ", J)
    print(
        "----------------------------------------------------------------------------------"
    )

    return J


def gradient(theta: np.array, X: np.array, y: np.array, lmbda: float) -> np.array:
    # get # of examples
    m = len(X)

    # calculate h(X)
    h = sigmoid(X.dot(theta))

    # calculate regularized gradient grad
    grad = np.dot((h - y), X) / m
    grad_reg_term = (lmbda * theta[1:].T) / m
    grad[1:] = grad[1:] + grad_reg_term

    return grad
