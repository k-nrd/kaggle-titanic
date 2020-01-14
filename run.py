import pandas as pd
import numpy as np

from prepare import prepFeatureMatrix
import regularizedLogistic as rl
import neuralNetwork as nn

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

y = train_data["Survived"].to_numpy()
train_X = prepFeatureMatrix(train_data)
test_X = prepFeatureMatrix(test_data)
test_ids = pd.DataFrame(test_data["PassengerId"])

input("Dataset prepared. Press anything to continue.")

lrTheta = rl.train(train_X, y, rl.costFunction, rl.gradient, 10)
lrTrainP = rl.classify(lrTheta, train_X)

nnParams = nn.train(train_X, y, 25, nn.costFunction, nn.regularizedBackprop, 1)
nnTrainP = nn.classify(nnParams, 25, 1, train_X)
pd.DataFrame(nnTrainP).to_csv("nn_train_p.csv", index=False)
print(f"shape y: {y.shape}")
print("Logistic Regression Train Accuracy: ", np.mean(lrTrainP == y) * 100)
print("Neural Network Train Accuracy: ", np.mean(nnTrainP == y) * 100)
print(
    "--------------------------------------------------------------------------------"
)

# testP = rl.classify(lrTheta, test_X)
testP = nn.classify(nnParams, 25, 1, test_X)

result = test_ids.join(pd.DataFrame(testP, columns=["Survived"]))
result.to_csv("result.csv", index=False)
