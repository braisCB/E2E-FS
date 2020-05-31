import numpy as np
import matlab.engine
import os


class Matlab_InfFS_Fast:

    def __init__(self, n_features_to_select=10, alpha=.5, supervised=False, normalize=True, matlab_engine=None):
        self.n_features_to_select = n_features_to_select
        self.score = None
        self.ranking = None
        self.alpha = alpha
        self.supervised = supervised
        self.normalize = normalize
        self.matlab_engine = matlab_engine

    def fit(self, X, y):
        new_X = X.copy()
        if self.normalize:
            new_X -= new_X.min(axis=0)
            new_X /= new_X.max(axis=0)
        if self.matlab_engine is None:
            self.matlab_engine = matlab.engine.start_matlab()
        new_X = matlab.double(new_X.tolist())
        new_y = y.copy()
        new_y = matlab.double(new_y.reshape((-1, 1)).tolist())
        self.matlab_engine.addpath(os.path.dirname(os.path.realpath(__file__)), nargout=0)
        self.ranking, self.score = self.matlab_engine.infFS_fast(new_X, new_y, self.alpha, 0, nargout=2)
        self.ranking = np.asarray(self.ranking, dtype=int).flatten() - 1
        self.score = np.asarray(self.score).flatten()

    def transform(self, X):
        new_X = X[:, self.ranking[:self.n_features_to_select]]
        return new_X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
