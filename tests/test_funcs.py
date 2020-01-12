from pandas import read_csv
import numpy as np
from scipy.optimize import minimize
from costFunction import lrCostFunction, gradient
from predict import predict
from sigmoid import sigmoid

data = read_csv("test_data.txt").to_numpy()
X = data[:, [0, 1]]
y = data[:, 2]


initial_theta = np.array([0, 0, 0])
test_theta = np.array([-24, 0.2, 0.2])
lmbda = 0.1

print("Starting test...")
print("Printing input matrix and output vector")
print("X:\n", X[:5, :])
print("y:\n", y[:5].T)
print("-------------------------------------------------")

J = lrCostFunction(initial_theta, X, y, lmbda)
print("Cost at initial theta (zeros): ", J)
print("-------------------------------------------------")

J = lrCostFunction(test_theta, X, y, lmbda)
print("Cost at test theta: ", J)
print("-------------------------------------------------")

# print("Gradient: ", grad)
input("Program paused. Press ENTER to continue.")

res = minimize(
    lrCostFunction,
    initial_theta,
    args=(X, y, lmbda),
    method="BFGS",
    jac=gradient,
    options={"dist": True, "maxiter": 500},
)
theta = res.x
print("Optimization results:\n", res)
print("-------------------------------------------------")
J = lrCostFunction(theta, X, y, lmbda)
print("Cost at optimal theta: ", J)

prob = sigmoid(np.array([[1, 45, 85]]).dot(theta))
print(
    f"\nFor a student with scores 45 and 85, we predict an admission probability of {prob}."
)
print("Expected value: 0.775 +/- 0.002\n")


p = predict(theta, X)
print("Train Accuracy: ", np.mean(p == y) * 100)
print("Expected accuracy (approx): 89.0")
