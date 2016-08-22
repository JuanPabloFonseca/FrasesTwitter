from sklearn.base import TransformerMixin
import numpy as np
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_distances
import re

# convierte la matriz binaria de ngramas en una matriz de distancias
class BinaryToDistanceTransformer(TransformerMixin):
    def __init__(self, _norm='l2', _metric='euclidean'):
        self.norm = _norm
        self.metric = _metric

    def transform(self, X, y=None, **fit_params):

        # Xdense = np.matrix(X).astype('float')

        X_scaled = preprocessing.scale(X) # hace copia de los datos, posible optimizacion con copy=False, requiere sparce matrix

        X_normalized = preprocessing.normalize(X_scaled, norm=self.norm)

        X_dense = np.matrix(X_normalized).astype('float')
        return pairwise_distances(X_dense, metric=self.metric)

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
        #quitar ngramas repetidos (iguales exceptuando el orden, '#' y '@')
        #print(self.Xclean.shape[1])
        inv_map = {v: k for k, v in self.vectorizer.vocabulary_.items()}
        i = 0
        indices=[m for m in range(self.Xclean.shape[1])]
        while i < self.Xclean.shape[1] - 1:
            j=i+1
            while j < self.Xclean.shape[1]:
                n1=re.sub('[@#]', '', inv_map[indices[i]])
                n2=re.sub('[@#]', '', inv_map[indices[j]])
                if len(n1.split()) == len(n2.split()) and set(n1.split()) == set(n2.split()):
                    #print("{} = {}".format(inv_map[indices[i]],inv_map[indices[j]]))
                    for k in range(self.Xclean.shape[0]):
                        self.Xclean[k][i] += self.Xclean[k][j]
                    self.Xclean = np.delete(self.Xclean, j, 1)
                    indices.remove(indices[j])
                    j -= 1

                #else:
                    #print("{} != {}".format(inv_map[indices[i]],inv_map[indices[j]]))
                j+=1
            i+=1
            #print(self.Xclean.shape[1])
        #print(self.Xclean.shape[1])

        self.inv_map={ i : inv_map[indices[i]] for i in range(len(indices)) }
        return self.Xclean

    def fit(self, X, y=None, **fit_params):
        return self