import numpy as np
import pandas as pd
from scipy.io import loadmat

from neuralNetwork.costFunction import nnCostFunction
from utils import sigmoidGradient

data = loadmat("./tests/test_nn/ex4data1.mat")

X = data["X"]
ones = np.array([[1 for _ in range(len(X))]])
X = np.concatenate((ones.T, X), axis=1)

y = data["y"].reshape(-1)
y = pd.get_dummies(y).to_numpy()

params = loadmat("./tests/test_nn/ex4weights.mat")
theta1 = params["Theta1"]
theta2 = params["Theta2"]

theta1 = theta1.reshape(-1)
theta2 = theta2.reshape(-1)
nnParams = np.concatenate((theta1, theta2))

input("Testing cost function without regularization... [PRESS ANY KEY]")
J = nnCostFunction(nnParams, 25, X, y, 0)
print("Test cost without regularization: ", J)
print("Expected: approx. 0.287629\n")

input("Testing cost function with regularization... [PRESS ANY KEY]")
J = nnCostFunction(nnParams, 25, X, y, 1)
print("Test cost with regularization: ", J)
print("Expected: approx. 0.383770\n")

input("Testing sigmoid gradient... [PRESS ANY KEY]")
g = sigmoidGradient(np.array([[-1, -0.5, 0, 0.5, 1]]))
print("Computed sigmoid gradient: ", g[0])
print("Expected: [0.1967 0.235 0.25 0.235 0.1967]\n")
