import numpy as np
import pandas as pd
from scipy.io import loadmat

from neuralNetwork.functions import (
    costFunction,
    regularizedBackprop,
)
from neuralNetwork.train import train, classify
from neuralNetwork.utils import meltParams, sigmoidGradient
from tests.test_nn.test_nn_gradients import checkNNGradients
from utils import paddingOnes

data = loadmat("./tests/test_nn/ex4data1.mat")

X = data["X"]
X = paddingOnes(X)

yVec = data["y"].reshape(-1)
y = pd.get_dummies(yVec).to_numpy()

params = loadmat("./tests/test_nn/ex4weights.mat")
theta1 = params["Theta1"]
theta2 = params["Theta2"]

nnParams = meltParams(theta1, theta2)

input("Testing cost function without regularization... [PRESS ANY KEY]")
J = costFunction(nnParams, 25, X, y, 0)
print(f"Test cost without regularization: {J}")
print("Expected: approx. 0.287629\n")

input("Testing cost function with regularization... [PRESS ANY KEY]")
J = costFunction(nnParams, 25, X, y, 1)
print(f"Test cost with regularization: {J}")
print("Expected: approx. 0.383770\n")

input("Testing sigmoid gradient... [PRESS ANY KEY]")
g = sigmoidGradient(np.array([[-1, -0.5, 0, 0.5, 1]]))
print(f"Computed sigmoid gradient: {g[0]}")
print("Expected: [0.1967 0.235 0.25 0.235 0.1967]\n")

input("Testing backpropagation... [PRESS ANY KEY]")
checkNNGradients()
print("\n")

input("Testing backpropagation with regularization... [PRESS ANY KEY]")
lmbda = 3
checkNNGradients(lmbda)
J = costFunction(nnParams, 25, X, y, 3)
print(f"Cost at lambda {lmbda}: {J}")
print(f"Expected at lambda {lmbda}: 0.576051\n")

input("Testing optimizing for theta... [PRESS ANY KEY]")
lmbda = 1
params = train(X, y, 25, costFunction, regularizedBackprop, lmbda)
pred = classify(params, 25, 10, X)
print(f"Training set accuracy {np.mean(pred == yVec) * 100}")
