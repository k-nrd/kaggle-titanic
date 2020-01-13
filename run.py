import pandas as pd
import numpy as np

from prepare import prepFeatureMatrix
from regularizedLogistic.costFunction import lrCostFunction, gradient as grad
from regularizedLogistic.train import train
from utils import classify

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

y = train_data["Survived"].to_numpy()
train_X = prepFeatureMatrix(train_data)
test_X = prepFeatureMatrix(test_data)
test_ids = pd.DataFrame(test_data["PassengerId"])

input("Dataset prepared. Press anything to continue.")

theta = train(train_X, y, lrCostFunction, grad, 10)

train_p = classify(theta, train_X)
print("Train Accuracy: ", np.mean(train_p == y) * 100)
print(
    "--------------------------------------------------------------------------------"
)

test_p = classify(theta, test_X)
result = test_ids.join(pd.DataFrame(test_p, columns=["Survived"]))
result.to_csv("result.csv", index=False)
