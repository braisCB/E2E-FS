from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
import numpy as np


class SVMRFE:

    def __init__(self, n_features_to_select=10, normalize=True, step=.001):
        self.n_features_to_select = n_features_to_select
        self.score = None
        self.ranking = None
        self.normalize = normalize
        self.step = step

    def fit(self, X, y):
        new_X = X.copy()
        if self.normalize:
            new_X -= new_X.min(axis=0)
            new_X /= new_X.max(axis=0)
        estimator = LinearSVC(C=1., class_weight='balanced')
        print('nfeatures', self.n_features_to_select)
        self.selector = RFE(estimator, n_features_to_select=self.n_features_to_select, step=self.step).fit(new_X, y)
        self.score = 1. / self.selector.ranking_
        self.ranking = self.selector.ranking_
        print((self.ranking == 1).sum())

    def transform(self, X):
        new_X = X[:, self.ranking[:self.n_features_to_select]]
        return new_X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)