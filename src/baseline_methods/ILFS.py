import numpy as np
from scipy.linalg import eigh, solve


eps = np.finfo(float).eps


def ILFS_score(X, Y, TT=6, balance=False):
    A = LearningGraphWeights(X, Y, TT, balance)
    factor = 0.99

    I = np.eye(len(A))
    # w, _ = np.linalg.eig(A)
    w = eigh(A, eigvals_only=True, eigvals=(len(A)-1, len(A)-1))
    rho = w.max()
    r = factor / rho
    y = I - ( r * A )

    # S = solve(y, I, overwrite_b=True) - I
    S = np.linalg.inv(y) - I
    score = S.sum(axis=1)
    return score


def LearningGraphWeights(train_x, train_y, TT, balance):

    params_iter = 15
    numSamp, numFeat = train_x.shape

    num_classes = len(np.unique(train_y))
    unique_class_labels = np.sort(np.unique(train_y))[::-1]
    right_labels = 999 * np.ones_like(train_y, dtype=int)

    c = []
    mu_c = []
    std_c = []
    c_freq = []
    for i, label in enumerate(unique_class_labels):
        right_labels[train_y == label] = i
        c.append(train_x[train_y == label])
        mu_c.append(np.mean(c[-1], axis=0))
        std_c.append(np.std(c[-1], axis=0))
        c_freq.append(len(c))

    hist_weights = np.ones_like(right_labels)
    if balance:
        c_freq = np.array(c_freq)
        c_freq = c_freq.max() / c_freq
        hist_weights = c_freq[right_labels]
        numSamp = hist_weights.sum()

    mu_c = np.array(mu_c)
    std_c = np.array(std_c)

    std_c[std_c < 1000000*eps] = 1.
    numFactors = 2

    token = list(range(TT))
    numTokens = len(token)

    tokenFeatMatrix = np.zeros((numTokens,numFeat))
    priors_sep_scores = np.zeros((numFeat,))

    for f in range(numFeat):
        d = (train_x[:,f][:, None] - mu_c[:, f]) ** 2. / (std_c[:, f] ** 2.).sum()
        prob_class_est = np.abs(d) / np.maximum(1e-10, np.abs(d).sum(axis=1, keepdims=True))
        prob_class = prob_class_est[np.arange(len(prob_class_est)), right_labels.flatten()]
        if not np.isnan(np.sum(prob_class)) or np.isinf(np.sum(prob_class)):
            try:
                bins = np.linspace(prob_class.min(), prob_class.max() - eps, TT).tolist()
                bins.append(2. * bins[-1] - bins[-2])
                tokenFeatMatrix[:,f] = np.histogram(
                    prob_class,
                    bins,
                    weights=hist_weights
                )[0]
            except:
                tokenFeatMatrix[:, f] = numSamp / numTokens
        priors_sep_scores[f] = np.sum(tokenFeatMatrix[-1 - int(round(TT*0.35)):,f]) / numSamp

    priors_sep_scores *= 100

    priors_sep_scores[np.isnan(priors_sep_scores)] = 1
    priors_sep_scores[np.isinf(priors_sep_scores)] = 1

    prob_token_factor = np.array([np.linspace(5000,1,numTokens), np.linspace(1,5000,numTokens)])
    prob_token_factor /= prob_token_factor.sum(axis=1, keepdims=True)

    prob_factor_feat = np.zeros((numFactors,numFeat))
    prob_factor_feat[0] = priors_sep_scores
    prob_factor_feat[1] = 100. - priors_sep_scores
    prob_factor_feat /= prob_factor_feat.sum(axis=0, keepdims=True)

    prob_token_feat = np.zeros((numTokens, numFeat))
    for z in range(numFactors):
        prob_token_feat += prob_token_factor[z][:, None] * prob_factor_feat[z, :]

    prob_factor_token_feat = np.zeros((numFactors, numTokens, numFeat))

    lls = []
    for ii in range(params_iter):

        # 'E-step'
        for z in range(numFactors):
            prob_factor_token_feat[z] = prob_token_factor[z][:, None] * prob_factor_feat[z] * \
                                        tokenFeatMatrix / np.maximum(1e-12, prob_token_feat)

        # 'M-step'
        # 'update p(z|d)'
        prob_token_factor = prob_factor_token_feat.sum(axis=2)
        prob_token_factor[prob_token_factor == 0.] = 1e-12
        prob_token_factor /= np.maximum(1e-12, prob_token_factor.sum(axis=0, keepdims=True))

        # 'update p(w|z)')
        prob_factor_feat = prob_factor_token_feat.sum(axis=1)
        prob_factor_feat[prob_factor_feat == 0.] = 1e-12
        prob_factor_feat /= np.maximum(1e-12, prob_factor_feat.sum(axis=1, keepdims=True))

        # update p(d, w) and calculate likelihood
        prob_token_feat[:] = 0
        for z in range(numFactors):
            prob_token_feat += prob_token_factor[z][:, None] * prob_factor_feat[z]

        ll = (tokenFeatMatrix * np.log(prob_token_feat)).sum()

        lls.append(ll)

        # print('iter : ', ii, ' - Log-likelihood : ', ll)

        if len(lls) > 1 and lls[-1] - lls[-2] < 1e-6:
            break

    factor_representing_relevancy = 0

    G = prob_factor_feat[factor_representing_relevancy,:][:, None] @ prob_factor_feat[factor_representing_relevancy,:][None, :]
    return G


class ILFS:

    def __init__(self, n_features_to_select=10, TT=6, balance=False, normalize=True):
        self.n_features_to_select = n_features_to_select
        self.score = None
        self.ranking = None
        self.balance = balance
        self.TT = TT
        self.normalize = normalize

    def fit(self, X, y):
        new_X = X.copy()
        if self.normalize:
            new_X -= new_X.min(axis=0)
            new_X /= new_X.max(axis=0)
        self.score = ILFS_score(new_X, y, self.TT, self.balance)
        self.ranking = np.argsort(self.score)[::-1]

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
