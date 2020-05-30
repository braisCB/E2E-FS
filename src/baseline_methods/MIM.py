from sklearn.feature_selection import mutual_info_classif
import numpy as np


class MIM:

    def __init__(self, n_features_to_select=10, normalize=True):
        self.n_features_to_select = n_features_to_select
        self.score = None
        self.ranking = None
        self.normalize = normalize

    def fit(self, X, y):
        new_X = X.copy()
        if self.normalize:
            new_X -= new_X.min(axis=0)
            new_X /= new_X.max(axis=0)
        self.score = mutual_info_classif(new_X, y)
        self.ranking = np.argsort(-self.score, 0)

    def transform(self, X):
        new_X = X[:, self.ranking[:self.n_features_to_select]]
        return new_X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)