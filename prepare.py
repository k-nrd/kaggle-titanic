import pandas as pd
import numpy as np


def prepFeatureMatrix(dataset: pd.DataFrame) -> np.array:
    cols = ["Age", "SibSp", "Parch", "Fare"]
    X = dataset[cols]

    gender_dummies = pd.get_dummies(dataset["Sex"])
    embarked_dummies = pd.get_dummies(dataset["Embarked"])
    pclass_dummies = pd.get_dummies(dataset["Pclass"])

    X = X.join(pclass_dummies).join(embarked_dummies).join(gender_dummies)
    X = X.fillna(X.mean()).to_numpy()

    # get X length, prepare X with padded 1s (x0)
    ones = np.array([[1 for _ in range(len(X))]])
    X = np.concatenate((ones.T, X), axis=1)

    return X
