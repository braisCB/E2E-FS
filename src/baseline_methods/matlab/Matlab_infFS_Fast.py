import numpy as np
import matlab.engine
import os


class InfFS:

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


class ILFS:

    def __init__(self, n_features_to_select=10, TT=6, balance=False, normalize=True, matlab_engine=None):
        self.n_features_to_select = n_features_to_select
        self.score = None
        self.ranking = None
        self.balance = balance
        self.TT = TT
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
        self.ranking, self.score = self.matlab_engine.ILFS(new_X, new_y, self.TT, nargout=2)
        self.ranking = np.asarray(self.ranking, dtype=int).flatten() - 1
        self.score = np.asarray(self.score).flatten()

    def transform(self, X):
        new_X = X[:, self.ranking[:self.n_features_to_select]]
        return new_X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class RecursiveILFS:

    def __init__(self, n_features_to_select=10, TT=6, gamma=.5, balance=False, normalize=True):
        self.n_features_to_select = n_features_to_select
        self.score = None
        self.ranking = None
        self.gamma = gamma
        self.balance = balance
        self.TT = TT
        self.normalize = normalize

    def fit(self, X, y):
        new_X = X.copy()
        if self.normalize:
            new_X -= new_X.min(axis=0)
            new_X /= new_X.max(axis=0)
        self.ranking = np.arange(new_X.shape[-1]).astype(int)
        n_features = self.ranking.shape[0]
        while n_features > 5:
            try:
                score = ILFS_score(new_X[:, self.ranking[:n_features]], y, self.TT, self.balance)
                self.ranking[:n_features] = self.ranking[:n_features][np.argsort(-score, 0)]
                n_features = int(n_features * self.gamma)
            except:
                break

    def transform(self, X):
        new_X = X[:, self.ranking[:self.n_features_to_select]]
        return new_X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
