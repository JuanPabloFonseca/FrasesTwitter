import LDA_cluster
import ObtenerTweets
import LimpiarTweets

import ML_MejoraAlBaseline as ml

import numpy
import pandas as pd
# import statsmodels
import pylab


def principal():
    #s= ["política,economía","otros,entretenimiento","política","música,entretenimiento","música","economía,política"]
    #s=pd.Series(s)
    #print(s.str.get_dummies(sep=','))

    TT=obtenerTopicosTweets() #obtiene tópicos


    print(TT[0].loc[[0]]) # topics de tweet 0
    print(TT[1][0])  # tweet 0

    X = tfidf(TT[1])

    ml.clasificadores_supervisados(X, TT[0])

    X = word2vec

    ml.clasificadores_supervisados(X, TT[0])

def obtenerTopicosTweets():
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



if __name__ == '__main__':
    principal()