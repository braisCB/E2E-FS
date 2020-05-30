from skrebate.relieff import ReliefF as skReliefF
import numpy as np


class ReliefF:

    def __init__(self, n_features_to_select=10, n_neighbors=10, normalize=True):
        self.n_features_to_select = n_features_to_select
        self.score = None
        self.ranking = None
        self.n_neighbors = n_neighbors
        self.normalize = normalize

    def fit(self, X, y):
        new_X = X.copy()
        if self.normalize:
            new_X -= new_X.min(axis=0)
            new_X /= new_X.max(axis=0)
        model = skReliefF(n_neighbors=self.n_neighbors).fit(new_X, y)
        self.score = model.feature_importances_
        self.ranking = np.argsort(-self.score, 0)

    def transform(self, X):
        new_X = X[:, self.ranking[:self.n_features_to_select]]
        return new_X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)