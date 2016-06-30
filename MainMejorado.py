import LDA_cluster
import ObtenerTweets
import LimpiarTweets
from sklearn.feature_extraction.text import TfidfVectorizer

import ML_MejoraAlBaseline as ml

import numpy
import pandas as pd
# import statsmodels
import pylab


def principal():
    #s= ["política,economía","otros,entretenimiento","política","música,entretenimiento","música","economía,política"]
    #s=pd.Series(s)
    #print(s.str.get_dummies(sep=','))

    [topicos,listaTweets]=obtenerTopicosYTweets() #obtiene tópicos

    tweetsLimpios=LimpiarTweets.limpiarTexto(listaTweets)

    X=obtenerTfIdf(tweetsLimpios)


    ml.clasificadores_supervisados(X, topicos)

    # X = word2vec

    ml.clasificadores_supervisados(X, topicos)


def obtenerTopicosYTweets():
    clasificacion = []
    tweets = []
    with open('tass/tass_2015/tweetsTopic.txt') as f:
        for line in f:
            contenido = line.split('\\\\\\',1)
            clasificacion.append(contenido[0])
            tweets.append(contenido[1])

    topicos = pd.Series(clasificacion)
    tablaTopicos = topicos.str.get_dummies(sep=',')  # genera la matriz binaria

    return [tablaTopicos, tweets]

def obtenerTfIdf(datos):
    docs = []
    for t in datos:
        docs.append(', '.join(str(x) for x in t))
    # Use tf-idf features
    tfidf_vectorizer = TfidfVectorizer(min_df=2, lowercase=False, encoding='utf8mb4')

    # Calcula idf mediante log base 2 de (1 + N/n_i), donde N es el total de tweets, y n_i es el numero de documentos donde aparece la palabra
    tfidf = tfidf_vectorizer.fit_transform(docs)
    print(tfidf.shape)

if __name__ == '__main__':
    principal()