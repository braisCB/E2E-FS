import numpy as np
from scipy.stats import spearmanr
from scipy.linalg import eigh, solve


def get_infFS_score(X, y, alpha, supervised):
    if supervised:
        labels = np.unique(y)
        s_n = X[y == labels[0]]
        s_p = X[y == labels[1]]
        mu_s_n = np.mean(s_n, axis=0, keepdims=True)
        mu_s_p = np.mean(s_p, axis=0, keepdims=True)
        priors_corr = np.square(mu_s_p - mu_s_n)
        st = np.var(s_p, axis=0, keepdims=True) + np.var(s_n, axis=0, keepdims=True)
        st[st == 0.] = 10000.
        corr = priors_corr / st
        corr = corr.T @ corr
        corr -= corr.min()
        corr /= corr.max()
    else:
        corr, _ = spearmanr(X)
        corr[np.isinf(corr)] = 0.
        corr[np.isnan(corr)] = 0.
        corr = 1. - np.abs(corr)

    std_rows = np.std(X, axis=0, keepdims=True)
    sigma = np.maximum(std_rows, std_rows.T)
    sigma -= sigma.min()
    sigma /= sigma.max()
    sigma[np.isinf(sigma)] = 0.
    sigma[np.isnan(sigma)] = 0.

    A = alpha*sigma + (1-alpha)*corr

    factor = 0.99
    I = np.eye(A.shape[0], dtype=np.float64)
    # w = np.linalg.eigvals(A)
    w = eigh(A, eigvals_only=True, eigvals=(len(A)-1, len(A)-1))
    # print(np.abs(w.max()))
    r = factor / np.abs(w.max())

    S = solve(I - r * A, I, overwrite_b=True) - I
    # S = np.linalg.inv(I - r * A) - I
    score = S.sum(axis=1)
    print(score.min())
    return score


class InfFS:

    def __init__(self, n_features_to_select=10, alpha=.5, supervised=False, normalize=True):
        self.n_features_to_select = n_features_to_select
        self.score = None
        self.ranking = None
        self.alpha = alpha
        self.supervised = supervised
        self.normalize = normalize

    def fit(self, X, y):
        new_X = X.copy()
        if self.normalize:
            new_X -= new_X.min(axis=0)
            new_X /= new_X.max(axis=0)
        self.score = get_infFS_score(new_X, y, self.alpha, self.supervised)
        self.ranking = np.argsort(-self.score, 0)

    def transform(self, X):
        new_X = X[:, self.ranking[:self.n_features_to_select]]
        return new_X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class RecursiveInfFS:

    def __init__(self, n_features_to_select=10, alpha=.2, gamma=.5, supervised=False, normalize=True):
        self.n_features_to_select = n_features_to_select
        self.score = None
        self.ranking = None
        self.alpha = alpha
        self.supervised = supervised
        self.gamma = gamma
        self.normalize = normalize

    def fit(self, X, y):
        new_X = X.copy()
        if self.normalize:
            new_X -= new_X.min(axis=0)
            new_X /= new_X.max(axis=0)
        self.ranking = np.arange(new_X.shape[-1]).astype(int)
        n_features = self.ranking.shape[0]
        while n_features > 5:
            score = get_infFS_score(new_X[:, self.ranking[:n_features]], y, self.alpha, self.supervised)
            self.ranking[:n_features] = self.ranking[:n_features][np.argsort(-score, 0)]
            n_features = int(n_features * self.gamma)

    def transform(self, X):
        new_X = X[:, self.ranking[:self.n_features_to_select]]
        return new_X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
