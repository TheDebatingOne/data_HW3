import numpy as np
from sklearn.model_selection import KFold
import pandas as pd

np.random.seed(2)


def add_noise(data):
    """
    :param data: dataset as np.array of shape (n, d) with n observations and d features
    :return: data + noise, where noise~N(0,0.001^2)
    """
    noise = np.random.normal(loc=0, scale=0.001, size=data.shape)
    return data + noise


def get_folds():
    """
    :return: sklearn KFold object that defines a specific partition to folds
    """
    return KFold(n_splits=5, shuffle=True, random_state=34)


def read_data(path):
    return pd.read_csv(path)


def adjust_labels(y):
    y["season"] = y["season"].map(lambda x: x // 2)


class StandardScaler:
    def __init__(self):
        """ object instantiation """
        self.means = None
        self.stds = None

    def fit(self, X):
        """ fit scaler by learning mean and standard deviation per feature """
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)

    def transform(self, X):
        """ transform X by learned mean and standard deviation, and return it """
        for feature, mean, std in zip(X, self.means, self.stds):
            X[feature] = (X[feature] - mean)/std

    def fit_transform(self, X):
        """ fit scaler by learning mean and standard deviation per feature, and then transform X """
        self.fit(X)
        self.transform(X)


"""data = [[1, 200, 3], [10, 20, 30], [10, 20, 30]]
df = pd.DataFrame(data)
print(df)
SS = StandardScaler()
SS.fit_transform(df)
print(df)"""
