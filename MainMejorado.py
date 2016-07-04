import LDA_cluster
import ObtenerTweets
import LimpiarTweets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model

import ML_MejoraAlBaseline as ml

import numpy
import pandas as pd
import pylab

import time
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split

tfidf_vectorizer = TfidfVectorizer(max_features=4000,min_df=2, lowercase=False, encoding='utf8mb4',ngram_range=(1,2))

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

    #REGRESIÓN LOGÍSTICA. Para eso se junta la matriz binaria de tópicos (pandas.core.frame.DataFrame)
    #                                     y la matriz de los scores tfidf (scipy.sparse.csr.csr_matrix)

    Xarr=X_tr.toarray()                  #pasas a ndarray


    #esto en caso de necesitar PCA
    '''print("Se va a hacer pca")
    pca=PCA(n_components=0.8)         #PCA al 80% (determinar porcentaje según logit lo requiera)
    Xpca=pca.fit_transform(Xarr)      #reduces dimensionalidad
    Xdf = pd.DataFrame(Xpca)          #pasas a DataFrame
    print("Ya se hizo pca")
    print(Xdf.shape)'''

    #esto en caso de no utilizar PCA
    Xdf=pd.DataFrame(Xarr)

    RLdf= topicosTrain
    RLdf=pd.concat([RLdf,Xdf],axis=1) #juntas matriz binaria de tópicos y Xdf (tfidf con PCA)


    #revisar si son o no demasiadas columnas para la regresión logística

    X_te_arr = X_te.toarray()
    X_te_df = pd.DataFrame(X_te_arr)

    # esto en caso de haber hecho PCA
    # X_te_pca=pca.transform(X_te_arr)
    # X_te_df=pd.DataFrame(X_te_pca)

    print("Se va a hacer Regresión Logística")
    PredDataFrame=regresionLogistica(RLdf,X_te_df)
    print(PredDataFrame)
    print("presiento que no está funcionando bien... fin de la regresión logística")

    # X = word2vec
    # ml.clasificadores_supervisados(X, topicos)

def regresionLogistica(RLdf,X_te_df):
    train_cols = RLdf.columns[10:]
    logit = linear_model.LogisticRegression()
    RLcine = logit.fit(X=RLdf[train_cols], y=RLdf['cine'])
    RLdeportes = logit.fit(X=RLdf[train_cols], y=RLdf['deportes'])
    RLeco = logit.fit(X=RLdf[train_cols], y=RLdf['economía'])
    RLentret = logit.fit(X=RLdf[train_cols], y=RLdf['entretenimiento'])
    RLfutbol = logit.fit(X=RLdf[train_cols], y=RLdf['fútbol'])
    RLlite = logit.fit(X=RLdf[train_cols], y=RLdf['literatura'])
    RLmusica = logit.fit(X=RLdf[train_cols], y=RLdf['música'])
    RLotros = logit.fit(X=RLdf[train_cols], y=RLdf['otros'])
    RLpolitica = logit.fit(X=RLdf[train_cols], y=RLdf['política'])
    RLtecno = logit.fit(X=RLdf[train_cols], y=RLdf['tecnología'])

    P0 = pd.DataFrame(RLcine.predict(X=X_te_df[:]))
    P1 = pd.DataFrame(RLdeportes.predict(X=X_te_df[:]))
    P2 = pd.DataFrame(RLeco.predict(X=X_te_df[:]))
    P3 = pd.DataFrame(RLentret.predict(X=X_te_df[:]))
    P4 = pd.DataFrame(RLfutbol.predict(X=X_te_df[:]))
    P5 = pd.DataFrame(RLlite.predict(X=X_te_df[:]))
    P6 = pd.DataFrame(RLmusica.predict(X=X_te_df[:]))
    P7 = pd.DataFrame(RLotros.predict(X=X_te_df[:]))
    P8 = pd.DataFrame(RLpolitica.predict(X=X_te_df[:]))
    P9 = pd.DataFrame(RLtecno.predict(X=X_te_df[:]))
    PredDataFrame = pd.concat([P0, P1, P2, P3, P4, P5, P6, P7, P8, P9], axis=1)
    return PredDataFrame

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