from sklearn.base import TransformerMixin
import numpy as np
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_distances

# convierte la matriz binaria de ngramas en una matriz de distancias
class BinaryToDistanceTransformer(TransformerMixin):
    def __init__(self, _norm='l2', _metric='euclidean'):
        self.norm = _norm
        self.metric = _metric

    def transform(self, X, y=None, **fit_params):
        Xdense = np.matrix(X).astype('float')
        X_scaled = preprocessing.scale(Xdense)
        X_normalized = preprocessing.normalize(X_scaled, norm=self.norm)
        return pairwise_distances(X_normalized, metric=self.metric)

    def fit(self, X, y=None, **fit_params):
        return self


# filtra la matriz de ngramas por los tweets que tengan arriba de un numero mágico de ngramas
class FiltroNGramasTransformer(TransformerMixin):
    def __init__(self, numMagico, vectorizer):
        self.numMagico = numMagico
        self.map_index_after_cleaning = {}
        self.vectorizer = vectorizer

    def transform(self, X, y=None, **fit_params):



        self.Xclean = np.zeros((1, X.shape[1]))
        for i in range(0, X.shape[0]):
            # keep sample with size at least numMagico ngramas
            if X[i].sum() >= self.numMagico:
                self.Xclean = np.vstack([self.Xclean, X[i].toarray()])
                self.map_index_after_cleaning[self.Xclean.shape[0] - 2] = i

        self.Xclean = self.Xclean[1:, ]
        return self.Xclean

    def fit(self, X, y=None, **fit_params):
        return self