from src.baseline_methods.SFS_DFS import saliency_function, SFS as SFS_class
from tensorflow.keras import backend as K


class iSFS:

    def __init__(self, model_func, n_features_to_select=10, normalize=False,
                 loss='hinge', gamma=.9):
        self.model_func = model_func
        self.n_features_to_select = n_features_to_select
        self.score = None
        self.ranking = None
        self.normalize = normalize
        self.loss = loss
        self.gamma = gamma

    def fit(self, X, y=None, rank_kwargs=None, **fit_kwargs):
        fit_kwargs = fit_kwargs or {}
        rank_kwargs = rank_kwargs or {}
        def create_model(input_shape, **kwargs):
            model = self.model_func(input_shape, **kwargs)
            model.saliency = saliency_function.get_saliency(self.loss, model)
            return model

        new_X = X.copy()
        if self.normalize:
            new_X -= new_X.min(axis=0)
            new_X /= new_X.max(axis=0)
        self.ranking = SFS_class.get_rank('sfs', data=new_X, label=y, model_func=create_model, model_kwargs={},
                                          fit_kwargs=fit_kwargs,
                                          rank_kwargs={'gamma': self.gamma, 'epsilon': self.n_features_to_select},
                                          saliency_kwargs={}, **rank_kwargs)
        self.score = 1. / (self.ranking + 1)

    def transform(self, X):
        new_X = X[:, self.ranking[:self.n_features_to_select]]
        return new_X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def __del__(self):
        if hasattr(self, 'model'):
            del self.model
            K.clear_session()


class SFS(iSFS):

    def __init__(self, **kwargs):
        super(SFS, self).__init__(**kwargs)
        self.gamma = 0.
