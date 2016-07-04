import LDA_cluster
import ObtenerTweets
import LimpiarTweets

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

import ML_MejoraAlBaseline as ml

import numpy
import pandas as pd
import statsmodels.api as sm
import pylab

import time



tfidf_vectorizer = TfidfVectorizer(min_df=2, lowercase=False, encoding='utf8mb4')

def principal():
    [topicosTrain,listaTweets]=obtenerTopicosYTweets('train') #obtiene tópicos

    tweetsLimpios=LimpiarTweets.limpiarTexto(listaTweets)
    X_tr = obtenerTfIdf(tweetsLimpios)
    start = time.time()

    pca = PCA(n_components=0.90)
    Xpca = pca.fit_transform(X_tr.toarray())

    end = time.time()
    print("tiempo fit_transform TRAIN PCA: ", end - start)
    print("Reduccion de {0} a {1}".format(X_tr.shape[1], len(pca.explained_variance_ratio_)))

    [topicosTest, listaTweetsTest] = obtenerTopicosYTweets('test')  # obtiene tópicos
    tweetsLimpiosTest = LimpiarTweets.limpiarTexto(listaTweetsTest)
    X_te = calcularTFIDF(tweetsLimpiosTest)


    start = time.time()
    X_te_pca = pca.transform(X_te.toarray())
    end = time.time()
    print("tiempo transform TEST PCA: ", end - start)

    ml.clasificadores_supervisados(Xpca, topicosTrain, X_te_pca, topicosTest)

    #ml.clasificadores_supervisados(X, topicosTrain, X_te, topicosTest)


    #REGRESIÓN LOGÍSTICA. Para eso se junta la matriz binaria de tópicos (pandas.core.frame.DataFrame)
    #                                     y la matriz de los scores tfidf (scipy.sparse.csr.csr_matrix)


    print("Regresion logistica")
    Xdf = pd.DataFrame(Xpca)          #pasas a DataFrame

    print(Xdf.shape)

    RLdf= topicosTrain
    RLdf=pd.concat([RLdf,Xdf],axis=1) #juntas matriz binaria de tópicos y Xdf (tfidf con PCA)
    print(RLdf.shape)
    print(RLdf.head())


    train_cols = RLdf.columns[10:]
    #revisar si son o no demasiadas columnas para la regresión logística

    print("ahora se va a hacer logit")
    logit = sm.Logit(RLdf['cine'], RLdf[train_cols])
    RLcine = logit.fit()
    print(RLcine.summary())
    '''logit = sm.Logit(RLdf['deportes'], RLdf[train_cols])
    RLdeportes = logit.fit()
    logit = sm.Logit(RLdf['econom?a'], RLdf[train_cols])
    RLeco = logit.fit()
    logit = sm.Logit(RLdf['entretenimiento'], RLdf[train_cols])
    RLentret = logit.fit()
    logit = sm.Logit(RLdf['f?tbol'], RLdf[train_cols])
    RLfutbol = logit.fit()
    logit = sm.Logit(RLdf['literatura'], RLdf[train_cols])
    RLlite = logit.fit()
    logit = sm.Logit(RLdf['m?sica'], RLdf[train_cols])
    RLmusica = logit.fit()
    logit = sm.Logit(RLdf['otros'], RLdf[train_cols])
    RLotros = logit.fit()
    logit = sm.Logit(RLdf['pol?tica'], RLdf[train_cols])
    RLpolitica = logit.fit()
    logit = sm.Logit(RLdf['tecnolog?a'], RLdf[train_cols])
    RLtecno = logit.fit()'''


    X_te_df=pd.DataFrame(X_te_pca)

    print(RLcine.predict(X_te_df[:])) #debería imprimir 1s y 0s según la predicción

    # X = word2vec
    # ml.clasificadores_supervisados(X, topicos)

def obtenerTopicosYTweets(str):
    clasificacion = []
    tweets = []

    if(str == 'train'):
        archivo = 'tass/TASSTrain.txt'
    else:
        archivo = 'tass/TASSTest1k.txt'
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