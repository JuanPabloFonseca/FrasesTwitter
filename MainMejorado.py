#!/usr/bin/python3.5

import LDA_cluster
import ObtenerTweets
import LimpiarTweets
from sklearn.feature_extraction.text import TfidfVectorizer

import ML_MejoraAlBaseline as ml

import numpy
import pandas as pd
# import statsmodels
import pylab

from sklearn.cross_validation import train_test_split

tfidf_vectorizer = TfidfVectorizer(min_df=2, lowercase=False, encoding='utf8mb4')

def principal():
    #s= ["política,economía","otros,entretenimiento","política","música,entretenimiento","música","economía,política"]
    #s=pd.Series(s)
    #print(s.str.get_dummies(sep=','))

    [topicosTrain,listaTweets]=obtenerTopicosYTweets('train') #obtiene tópicos

    # X_train, X_test, y_train, y_test = train_test_split(listaTweets, topicos, test_size=0.25, random_state=42)

    tweetsLimpios=LimpiarTweets.limpiarTexto(listaTweets)
    X = obtenerTfIdf(tweetsLimpios)

    # X = StandardScaler().fit_transform(X)

    [topicosTest, listaTweetsTest] = obtenerTopicosYTweets('test')  # obtiene tópicos
    tweetsLimpiosTest = LimpiarTweets.limpiarTexto(listaTweetsTest)
    X_te = calcularTFIDF(tweetsLimpiosTest)

    ml.clasificadores_supervisados(X, topicosTrain, X_te, topicosTest)

    # X = word2vec
    # ml.clasificadores_supervisados(X, topicos)

def obtenerTopicosYTweets(str):
    clasificacion = []
    tweets = []

    if(str == 'train'):
        archivo = 'tass/tass_2015/tweetsTopic.txt'
    else:
        archivo = 'tass/TASSTest.txt'
    with open(archivo) as f:
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

    # Calcula idf mediante log base 2 de (1 + N/n_i), donde N es el total de tweets, y n_i es el numero de documentos donde aparece la palabra
    tfidf = tfidf_vectorizer.fit_transform(docs)
    return tfidf

def calcularTFIDF(datos):
    docs = []
    for t in datos:
        docs.append(', '.join(str(x) for x in t))
    tfidf = tfidf_vectorizer.transform(docs)
    return tfidf

if __name__ == '__main__':
    principal()