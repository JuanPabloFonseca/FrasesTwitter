import LDA_cluster
import ObtenerTweets

import LimpiarTweets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn.metrics import precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix

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

    #primer intento:
    #PredDataFrame=regresionLogistica(RLdf,X_te_df,topicosTest)

    #primer intento optimizado:
    #PDF=regLogOptimizada1(tweetsLimpios,topicosTrain,tweetsLimpiosTest,topicosTest)

    #segundo intento optimizado:
    PredDF=regLogOptimizada2(RLdf,X_te_df,topicosTest)
    print("\nA continuación la matriz de confusión: ")
    print(calcular_matriz_confusión(topicosTest,PredDF))
    print("Hay algo muy mal aquí...")

    # X = word2vec
    # ml.clasificadores_supervisados(X, topicos)

def regresionLogistica(RLdf,X_te_df,topicosTest):

    model = OneVsRestClassifier(linear_model.LogisticRegression())
    model.fit(X=RLdf[RLdf.columns[10:]], y=RLdf[RLdf.columns[:10]])
    PredTrain = pd.DataFrame(model.predict(RLdf[RLdf.columns[10:]]))

    print("\nDel train: ")
    for i in range(0, 10):
        prt, rct, fbt, sut = precision_recall_fscore_support(RLdf[RLdf.columns[:10]].iloc[:, i], PredTrain[i], average='binary')
        print("{0}: Precision {1}, Recall {2}, F1 {3}".format(RLdf.columns[i], prt, rct, fbt))



    PredDF = pd.DataFrame(model.predict(X_te_df))

    print("\nDel test: ")
    pr0, rc0, fb0, su0 = precision_recall_fscore_support(topicosTest['cine'], PredDF[0], average='binary')
    print("Cine: Precision {0}, Recall {1}, F1 {2}".format(pr0, rc0, fb0))
    for i in range(1,9):
        pr, rc, fb, su = precision_recall_fscore_support(topicosTest.iloc[:,i], PredDF[i+1], average='binary')
        print("{0}: Precision {1}, Recall {2}, F1 {3}".format(topicosTest.columns[i],pr, rc, fb))

    PredDF.columns = ["cine", "deportes", "economía", "entretenimiento", "fútbol", "literatura",
                "música", "otros", "política", "tecnología"]

    return PredDF

def regLogOptimizada1(tTrain,topicTrain,tTest,topicTest):#primer intento

    #esto es increiblemente ineficiente y pierde el punto????? creo que sí
    mins=[3,4,5,3,4,4,4,2,1,4]#se vio que estos valores para min_df optimizaban el F1 de cada topico en train
    PDF=pd.DataFrame()
    for i in range(10):
        tfidf_vectorizer = TfidfVectorizer(max_features=4000, min_df=mins[i], lowercase=False, encoding='utf8mb4', ngram_range=(2, 5))
        X_train=pd.DataFrame(obtenerTfIdfParticular(tTrain,tfidf_vectorizer).toarray())
        logReg=linear_model.LogisticRegression()
        logReg.fit(X=X_train,y=topicTrain[topicTrain.columns[i]])
        PredTrain=pd.DataFrame(logReg.predict(X_train))
        PDF=pd.concat([PDF,PredTrain],axis=1)
        prt, rct, fbt, sut = precision_recall_fscore_support(topicTrain[topicTrain.columns[i]], PredTrain, average='binary')
        print("{0}: Precision {1}, Recall {2}, F1 {3}".format(topicTrain.columns[i], prt, rct, fbt))


    #falta evaluar en test con esos modelos... pero no lo voy a hacer porque esto es increiblemente ineficiente

    return PDF


def regLogOptimizada2(RLdf,X_te_df,topicosTest):
    cs=[0.86, 0.01, 0.97, 0.99, 0.84, 0.01, 0.99, 1.0, 0.92, 0.01]

    #este código comentado fue para encontrar las mejores cs para cada tópico, y son las que están aquí arriba
    '''for i in range(10):
        fbmax=0
        jmax=0.01
        for j in range (1,101,1):
            logReg=linear_model.LogisticRegression(C=j/100)
            logReg.fit(X=RLdf[RLdf.columns[10:]], y=RLdf[RLdf.columns[i]])
            PredTrain=pd.DataFrame(logReg.predict(X=RLdf[RLdf.columns[10:]]))
            prt, rct, fbt, sut = precision_recall_fscore_support(RLdf[RLdf.columns[i]], PredTrain, average='binary')
            if fbt > fbmax:
                fbmax=fbt
                jmax=j/100
        cs[i]=jmax
    print(cs)'''

    #se obtienen las mejores regresiones logísticas para el conjunto de entrenamiento (SIN ver al conjunto de prueba)
    print("\nDel train: ")
    PDF = pd.DataFrame()
    logReg=[]
    for i in range(10):
        logReg.append(linear_model.LogisticRegression(C=cs[i]))
        logReg[i].fit(X=RLdf[RLdf.columns[10:]], y=RLdf[RLdf.columns[i]])
        PredTrain = pd.DataFrame(logReg[i].predict(X=RLdf[RLdf.columns[10:]]))
        PDF = pd.concat([PDF, PredTrain], axis=1)
        prt, rct, fbt, sut = precision_recall_fscore_support(RLdf[RLdf.columns[i]], PredTrain, average='binary')
        print("{0}: Precision {1}, Recall {2}, F1 {3}".format(RLdf.columns[i], prt, rct, fbt))


    #luego, se ajustan los datos de prueba a las regresiones:
    print("\nDel test: ")
    PredDF = pd.DataFrame()
    PredTest = pd.DataFrame(logReg[0].predict(X_te_df))
    PredDF = pd.concat([PredDF, PredTest], axis=1)
    pr0, rc0, fb0, su0 = precision_recall_fscore_support(topicosTest['cine'], PredDF, average='binary')
    print("Cine: Precision {0}, Recall {1}, F1 {2}".format(pr0, rc0, fb0))
    PredDF = pd.concat([PredDF, pd.DataFrame(np.zeros((PredDF.shape[0],1),dtype=np.int))], axis=1)
    for i in range(1, 9):
        PredTest = pd.DataFrame(logReg[i+1].predict(X_te_df))
        PredDF = pd.concat([PredDF, PredTest], axis=1)
        pr, rc, fb, su = precision_recall_fscore_support(topicosTest.iloc[:, i], PredDF.iloc[:,i + 1], average='binary')
        print("{0}: Precision {1}, Recall {2}, F1 {3}".format(topicosTest.columns[i], pr, rc, fb))

    PredDF.columns = ["cine","deportes","economía","entretenimiento","fútbol","literatura",
                                             "música","otros","política","tecnología"]
    return PredDF


def calcular_matriz_confusión(topicos_true,topicos_pred):
    y_true=[]
    y_pred=[]
    for i in range(topicos_true.shape[0]):
        y_true.append(topicos_true.iloc[i,:].argmax())
        y_pred.append(topicos_pred.iloc[i,:].argmax())#REVISAR ESTO!!

    cm=confusion_matrix(y_true, y_pred, labels=["cine","deportes","economía","entretenimiento","fútbol","literatura",
                                             "música","otros","política","tecnología"])
    cm=pd.DataFrame(cm,index=["cin","dep","eco","ent","fút","lit","mús","otr","pol","tec"],
                     columns=["cin","dep","eco","ent","fút","lit","mús","otr","pol","tec"])
    return cm

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


def obtenerTfIdfParticular(datos,vect):
    docs = []
    for t in datos:
        docs.append(', '.join(str(x) for x in t))
    # Use tf-idf features

    # Calcula idf mediante log base 2 de (1 + N/n_i), donde N es el total de tweets, y n_i es el numero de documentos donde aparece la palabra
    tfidf = vect.fit_transform(docs)
    return tfidf


def calcularTFIDF(datos):
    docs = []
    for t in datos:
        docs.append(', '.join(str(x) for x in t))
    tfidf = tfidf_vectorizer.transform(docs)
    return tfidf

if __name__ == '__main__':
    principal()