from typing import Callable
import numpy as np
from scipy.optimize import minimize


def train(
    X: np.array, y: np.array, costFunction: Callable, gradient: Callable, lmbda: float
) -> np.array:
    initial_theta = np.zeros(X.shape[1])

    res = minimize(
        costFunction,
        initial_theta,
        args=(X, y, lmbda),
        method="BFGS",
        jac=gradient,
        options={"dist": True, "maxiter": 500},
    )

    print("Optimization results:\n", res)
    print(
        "--------------------------------------------------------------------------------"
    )

    return res.x
