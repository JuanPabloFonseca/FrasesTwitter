import LDA_cluster
import ObtenerTweets

import LimpiarTweets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn.metrics import precision_recall_fscore_support

import ML_MejoraAlBaseline as ml

import numpy as np
import pandas as pd
import pylab

import time
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split

tfidf_vectorizer = TfidfVectorizer(max_features=4000,min_df=4, lowercase=False, encoding='utf8mb4',ngram_range=(2,5))

def principal():
    [topicosTrain,listaTweets]=obtenerTopicosYTweets('train') #obtiene tópicos

    tweetsLimpios=LimpiarTweets.limpiarTexto(listaTweets)
    X_tr = obtenerTfIdf(tweetsLimpios)
    start = time.time()
    print(X_tr.shape)

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
    Xdf=pd.DataFrame(Xarr) #pasas a dataframe

    RLdf= topicosTrain
    RLdf=pd.concat([RLdf,Xdf],axis=1) #juntas matriz binaria de tópicos y Xdf (tfidf con PCA)



    #revisar si son o no demasiadas columnas para la regresión logística

    X_te_arr = X_te.toarray()
    X_te_df = pd.DataFrame(X_te_arr) #pasas a DataFrame

    # esto en caso de haber hecho PCA
    # X_te_pca=pca.transform(X_te_arr)
    # X_te_df=pd.DataFrame(X_te_pca)

    print("Se va a hacer Regresión Logística")
    PredDataFrame=regresionLogistica(RLdf,X_te_df,topicosTest)
    #print(PredDataFrame)



    # X = word2vec
    # ml.clasificadores_supervisados(X, topicos)

def regresionLogistica(RLdf,X_te_df,topicosTest):
    train_cols = RLdf.columns[10:]
    logit = linear_model.LogisticRegression()

    logit.fit(X=RLdf[train_cols], y=RLdf['cine'])
    P0 = pd.DataFrame(logit.predict(X=X_te_df))
    #logit.fit(X=RLdf[train_cols], y=RLdf['deportes']) no hay de deportes

    logit.fit(X=RLdf[train_cols], y=RLdf['economía'])
    P2 = pd.DataFrame(logit.predict(X=X_te_df))
    logit.fit(X=RLdf[train_cols], y=RLdf['entretenimiento'])
    P3 = pd.DataFrame(logit.predict(X=X_te_df))
    logit.fit(X=RLdf[train_cols], y=RLdf['fútbol'])
    P4 = pd.DataFrame(logit.predict(X=X_te_df))
    logit.fit(X=RLdf[train_cols], y=RLdf['literatura'])
    P5 = pd.DataFrame(logit.predict(X=X_te_df))
    logit.fit(X=RLdf[train_cols], y=RLdf['música'])
    P6 = pd.DataFrame(logit.predict(X=X_te_df))
    logit.fit(X=RLdf[train_cols], y=RLdf['otros'])
    P7 = pd.DataFrame(logit.predict(X=X_te_df))
    logit.fit(X=RLdf[train_cols], y=RLdf['política'])
    P8 = pd.DataFrame(logit.predict(X=X_te_df))
    logit.fit(X=RLdf[train_cols], y=RLdf['tecnología'])
    P9 = pd.DataFrame(logit.predict(X=X_te_df))


    pr0, rc0, fb0, su0 = precision_recall_fscore_support(topicosTest['cine'], P0, average='binary')
    print("Cine: Precision {0}, Recall {1}, F1 {2}".format(pr0, rc0, fb0))
    pr2, rc2, fb2, su2 = precision_recall_fscore_support(topicosTest['economía'], P2, average='binary')
    print("Economía: Precision {0}, Recall {1}, F1 {2}".format(pr2, rc2, fb2))
    pr3, rc3, fb3, su3 = precision_recall_fscore_support(topicosTest['entretenimiento'], P3, average='binary')
    print("Entretenimiento: Precision {0}, Recall {1}, F1 {2}".format(pr3, rc3, fb3))
    pr4, rc4, fb4, su4 = precision_recall_fscore_support(topicosTest['fútbol'], P4, average='binary')
    print("Fútbol: Precision {0}, Recall {1}, F1 {2}".format(pr4, rc4, fb4))
    pr5, rc5, fb5, su5 = precision_recall_fscore_support(topicosTest['literatura'], P5, average='binary')
    print("Literatura: Precision {0}, Recall {1}, F1 {2}".format(pr5, rc5, fb5))
    pr6, rc6, fb6, su6 = precision_recall_fscore_support(topicosTest['música'], P6, average='binary')
    print("Música: Precision {0}, Recall {1}, F1 {2}".format(pr6, rc6, fb6))
    pr7, rc7, fb7, su7 = precision_recall_fscore_support(topicosTest['otros'], P7, average='binary')
    print("Otros: Precision {0}, Recall {1}, F1 {2}".format(pr7, rc7, fb7))
    pr8, rc8, fb8, su8 = precision_recall_fscore_support(topicosTest['política'], P8, average='binary')
    print("Política: Precision {0}, Recall {1}, F1 {2}".format(pr8, rc8, fb8))
    pr9, rc9, fb9, su9 = precision_recall_fscore_support(topicosTest['tecnología'], P9, average='binary')
    print("Tecnología: Precision {0}, Recall {1}, F1 {2}".format(pr9, rc9, fb9))

    P1=pd.DataFrame(np.zeros((1000,1), dtype=np.int))#no hay ninguno del tema de deportes

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