#!/usr/bin/python3.5

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from LimpiarTweets import limpiarTextoTweet, quitarAcentos, quitarEmoticons, tokens2daClust
from snow_datatransformer import BinaryToDistanceTransformer, FiltroNGramasTransformer
from collections import Counter
from misStopWords import creaStopWords
from datetime import datetime
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial import distance
import fastcluster
import re
import time


import matplotlib.pyplot as plt

def mostrarNTweetsCluster(N, data_transform, indL):
    # Mostrar los n tweets de cada cluster
    idx_clusts = sorted([(l, k) for k, l in enumerate(indL)], key=lambda x: x[0])
    n_c = {}
    for x in idx_clusts:
        if not x[0] in n_c.keys() :
            print("Tweets Cluster {0}".format(x[0]))
            n_c[x[0]]=0

        if n_c[x[0]] >= N:
            continue
        else:
            n_c[x[0]] += 1
            print(tweets_cluster[ventana][data_transform.named_steps['filtrar'].map_index_after_cleaning.get(x[1])])


def mostrarNTweetsCluster2(N, data_transform, indL, indL2): # N = num tweets por cada cluster original
    # Mostrar los n tweets de cada cluster
    idx_clusts = sorted([(l, k) for k, l in enumerate(indL)], key=lambda x: x[0])
    idx_clusts2 = sorted([(l, k) for k, l in enumerate(indL2)], key=lambda y: y[0])
    n_c = {}
    for y in idx_clusts2:
        if not y[0] in n_c.keys() :
            print("Tweets Cluster {0}".format(y[0]))
            n_c[y[0]]=0


        tuitscluster=[x[1] for x in idx_clusts if x[0] == y[1] + 1]
        for k in range(min(N,len(tuitscluster))):
            print(tweets_cluster[ventana][data_transform.named_steps['filtrar'].map_index_after_cleaning.get(tuitscluster[k])])
        print(" ")

def mostrarNGramas2(indL, indL2, centroides, cuenta, inv_map): # sólo muestra los ngramas de cada cluster (2da clusterizacion)
    # Mostrar los n tweets de cada cluster
    idx_clusts = sorted([(l, k) for k, l in enumerate(indL)], key=lambda x: x[0])   # lista indexada de los los clusters originales (índice empieza en 1) vs los tweets (índice empieza en 0)
    idx_clusts2 = sorted([(l, k) for k, l in enumerate(indL2)], key=lambda y: y[0]) # lista indexada de los los clusters nuevos (índice empieza en 1) vs los clusters originales (índice empieza en 0)
    frecuencia_ngramas_total = [-1] * len(set([idx_clusts2[i][0] for i in range(len(idx_clusts2))])) # se guarda la frecuencia de los ngramas en TODOS los clusters nuevos (separado por cluster nuevo)

    ngramas_centroides = [] #ngramas correspondientes a los centroides originales
    for c in range(len(centroides)):
        ngramas_centroides.append([centroides[c][i]*cuenta[c] for i in range(len(centroides[c]))]) # al final del for, ngramas_centroides tiene la frecuencia de los ngramas

    for y in idx_clusts2:
        if frecuencia_ngramas_total[y[0]-1] == -1 : #si todavía no hay nada del cluster y[0] - 1
            frecuencia_ngramas_total[y[0] - 1] = [0]*len(inv_map)
        frecuencia_ngramas_total[y[0] - 1] = [frecuencia_ngramas_total[y[0] - 1][i] + ngramas_centroides[y[1]][i] for i in range(len(frecuencia_ngramas_total[y[0] - 1]))]

    lista_ngramas_por_cluster=[{} for i in range(len(frecuencia_ngramas_total))]
    for f in range(len(frecuencia_ngramas_total)):
        print("Ngramas del cluster {0}: ".format(f+1))
        for ng in range(len(frecuencia_ngramas_total[f])):
            if frecuencia_ngramas_total[f][ng] > 0:
                lista_ngramas_por_cluster[f][inv_map[ng]]=frecuencia_ngramas_total[f][ng]
        lista_ngramas_por_cluster[f] = sorted(lista_ngramas_por_cluster[f].items(), key=lambda x: x[1], reverse=True)
        print(lista_ngramas_por_cluster[f])

    return lista_ngramas_por_cluster # es una lista de listas de ngramas por cluster (nuevo)



def mostrarNGramas(Xclean, freqTwCl, indL, inv_map):

    # Xclean = data_transform.named_steps['filtrar'].Xclean
    # inv_map = data_transform.named_steps['filtrar'].inv_map

    # obtención del (los) ngrama(s) MÁS repetido(s) en cada cluster
    # print(inv_map)
    main_ngram_in_cluster = [-1] * len(freqTwCl)


    num_ngram_total = [-1] * len(freqTwCl) # guarda la frecuencia de los ngramas PARA TODOS LOS CLUSTERS.
    cuenta = [0] * len(freqTwCl)
    # centroide = []
    centroide = np.zeros((len(freqTwCl), Xclean.shape[1]))


    for clust in range(len(freqTwCl)):
        num_ngram = {}  # [0] * data_transform.named_steps['filtrar'].Xclean.shape[1]

        # num_ngram guarda la frecuencia de los ngramas para el cluster clust.

        # centroide.append([0] * Xclean.shape[1])
        cont = 0
        # tweets_del_cluster = []
        for tweet in range(Xclean.shape[0]):
            if indL[tweet] == clust + 1:
                # tweets_del_cluster.append(tweet)
                cont += 1
                for i in range(Xclean.shape[1]):
                    centroide[clust][i] += Xclean[tweet][i]
                    if len(inv_map) > 0:
                        if inv_map[i] in num_ngram.keys():
                            num_ngram[inv_map[i]] += data_transform.named_steps['filtrar'].Xclean[tweet][i]
                        elif data_transform.named_steps['filtrar'].Xclean[tweet][i] > 0:
                            num_ngram[inv_map[i]] = data_transform.named_steps['filtrar'].Xclean[tweet][i]
        if len(inv_map) > 0:
            num_ngram = sorted(num_ngram.items(), key=lambda x: x[1], reverse=True)

        centroide[clust] = [(x / cont) for x in centroide[clust]]
    # centroide = np.matrix(centroide)

        # cercano = -1
        # distancia = 10000000

        # print(tweets_del_cluster)
        # for k in range(len(tweets_del_cluster)):
        #    actual = tweets_del_cluster[k]
        #    distActual = distance.euclidean(centroide[clust], data_transform.named_steps['filtrar'].Xclean[actual, :])
        #    if distActual < distancia:
        #        cercano = actual
        #        distancia = distActual
        #print("\n{} Tweets en Cluster {}".format(cont, clust + 1))



        # main_ngram_in_cluster[clust] = [cont, tweets_cluster[ventana][data_transform.named_steps['filtrar'].map_index_after_cleaning.get(cercano)], num_ngram]
        # print("CENTROIDE {}".format(centroide[clust]))
        # print("Num. Tweets {}, un tweet ejemplar: \"{}\" y ngramas {}".format(cont,
        #      tweets_cluster[ventana][data_transform.named_steps['filtrar'].map_index_after_cleaning.get(cercano)], num_ngram))


        num_ngram_total[clust] = num_ngram  # se guarda num_ngram. Al final, num_ngram_total tiene la info de TODOS los clusters
        cuenta[clust]=cont

    return [centroide, num_ngram_total, cuenta] # se regresa num_ngram_total, porque num_ngram sólo sería para el último cluster y queremos para todos

def clusterDelTweet(tw,centroides,cnt):
    tw = limpiarTextoTweet(tw, stop_words)
    inv_map = data_transform.named_steps['filtrar'].inv_map
    vectortweet = [0]*len(inv_map)
    for ng in range(len(inv_map)):
        nglim = re.sub('[@#]', '', inv_map[ng])
        ngsp = nglim.split()
        for p in range(len(tw)-len(ngsp)+1):
            pal = [re.sub('[@#]', '', tw[p+pa]) for pa in range(len(ngsp))]
            if set(ngsp) == set(pal):
                vectortweet[ng] += 1

    clusterBasura = np.argmax(cnt)
    cercano =- 1
    dist = 1000000000
    for c in range(len(centroides)):
        if c != clusterBasura:
            distActual = distance.euclidean(centroides[c], vectortweet)
            if distActual < dist:
                dist = distActual
                cercano = c

    # regresa el número de cluster al que el tweet "pertenece" (recordando que la numeración empieza desde 1)
    return cercano+1

def imprimirDendogramas(X):
    hclust_methods = ["ward", "median", "centroid", "weighted", "single", "complete", "average"]

    iris_dendlist = []
    show_leaf_counts = True
    # for i in hclust_methods:
    #    #hclust(d_iris, method = hclust_methods[i])
    #   iris_dendlist.append()
    # names(iris_dendlist) = hclust_methods
    # iris_dendlist
    # par(mfrow = c(4,2))
    # for i in range(7):

    #    ddata = iris_dendlist[i]
    #    plt.subplot(ddata, i, figsize=(6, 5))

    # fig, axes = plt.subplots(nrows=3, ncols=3)
    fig = plt.figure(1, figsize=(20, 10))
    fig.suptitle('Corona, modelo y pan. [1,3] ngramas. Distancia euclideana.', fontsize=20)
    # plt.clf()

    plt.subplot(3, 3, 1)
    plt.title(hclust_methods[0])
    hc_iris = fastcluster.linkage(X, method=hclust_methods[0])
    sch.dendrogram(hc_iris,
       color_threshold=1,
       # p=6,
       # truncate_mode='lastp',
       show_leaf_counts=show_leaf_counts)

    plt.subplot(3, 3, 2)
    plt.title(hclust_methods[1])
    hc_iris = fastcluster.linkage(X, method=hclust_methods[1])
    sch.dendrogram(hc_iris,
       color_threshold=1,
       # p=6,
       # truncate_mode='lastp',
       show_leaf_counts=show_leaf_counts)

    plt.subplot(3, 3, 3)
    plt.title(hclust_methods[2])
    hc_iris = fastcluster.linkage(X, method=hclust_methods[2])
    sch.dendrogram(hc_iris,
       color_threshold=1,
       # p=6,
       # truncate_mode='lastp',
       show_leaf_counts=show_leaf_counts)

    plt.subplot(3, 3, 4)
    plt.title(hclust_methods[3])
    hc_iris = fastcluster.linkage(X, method=hclust_methods[3])
    sch.dendrogram(hc_iris,
       color_threshold=1,
       # p=6,
       # truncate_mode='lastp',
       show_leaf_counts=show_leaf_counts)

    plt.subplot(3, 3, 5)
    plt.title(hclust_methods[4])
    hc_iris = fastcluster.linkage(X, method=hclust_methods[4])
    sch.dendrogram(hc_iris,
       color_threshold=1,
       # p=6,
       # truncate_mode='lastp',
       show_leaf_counts=show_leaf_counts)

    plt.subplot(3, 3, 6)
    plt.title(hclust_methods[5])
    hc_iris = fastcluster.linkage(X, method=hclust_methods[5])
    sch.dendrogram(hc_iris,
       color_threshold=1,
       # p=6,
       # truncate_mode='lastp',
       show_leaf_counts=show_leaf_counts)

    plt.subplot(3, 3, 7)
    plt.title(hclust_methods[6])
    hc_iris = fastcluster.linkage(X, method=hclust_methods[6])
    sch.dendrogram(hc_iris,
       color_threshold=1,
       # p=6,
       # truncate_mode='lastp',
       show_leaf_counts=show_leaf_counts)

    # plt.figure(iris_dendlist[[i]], axes = False, horiz = True)
    show_leaf_counts = True

    # plt.title("Dendrogram")


    # plt.figure(1, figsize=(6, 5))


if __name__ == "__main__":

    # ########## PARAMETROS

    time_window_mins = 14400.0
    n_documentos_maximos = 5
    factor_frecuencia = 0.001
    num_ngrams_in_tweet = 3
    minimo_usuarios = 3
    minimo_hashtags = 3
    ngrama_minimo = 2
    ngrama_maximo = 4

    ### ngramas

    # ########## LEER ARCHIVO

    #corpus = []
    total_tweets_norepetidos = []
    tid_to_raw_tweet = {}
    tid_to_urls_window_corpus = {}
    tids_window_corpus = []

    stop_words = creaStopWords()

    # faltan ventanas de tweets
    ventana = 0
    tweet_unixtime_old = -1
    ventanas = []
    ventanas.append([])
    tweets_cluster = []
    tweets_cluster.append([])
    archivo = open('PanCoronaModelojsons/todos_tws.txt')
    start = time.time()
    for line in archivo:
        contenido = line.split('\\\\\\\\\\\\')
        tweet_gmttime = datetime.strptime(contenido[0][1:], '%a %b %d %H:%M:%S %z %Y')  # contenido[0]
        tweet_unixtime = tweet_gmttime.timestamp()
        tweet_id = contenido[1]
        text = contenido[2][:-2]

        users = re.findall("@[^\s]+", text)
        hashtags = re.findall("#[^\s]+", text)

        # no considerar tweets repetidos
        tw = re.sub("(#|@)([^\s]+)", ' ', text)
        tw = limpiarTextoTweet(tw, stop_words)
        if tw not in total_tweets_norepetidos:
            # no esta repetido
            total_tweets_norepetidos.append(tw)
        else:
            # esta repetido
            continue

        features = limpiarTextoTweet(text, stop_words)
        tweet_bag = ""
        try:
            for feature in features:
                tweet_bag += feature + " , "
        except:
            pass

        if tweet_unixtime_old == -1:
            tweet_unixtime_old = tweet_unixtime

        # #while this condition holds we are within the given size time window
        if (tweet_unixtime - tweet_unixtime_old) < time_window_mins * 60:
            if len(users) < minimo_usuarios and len(hashtags) < minimo_hashtags and len(
                    features) > 3:
                tweet_bag = tweet_bag[:-1]
                ventanas[ventana].append(tweet_bag)
                tweets_cluster[ventana].append(text)
        else:
            # incrementar ventana
            ventana += 1
            ventanas.append([])
            tweets_cluster.append([])
            tweet_unixtime_old = tweet_unixtime
    end = time.time()
    print("tiempo leer ventanas: {} seg".format(end - start))

    print("Ventanas: {}".format(len(ventanas)))
    for ventana in range(len(ventanas)):
        print("VENTANA {}".format(ventana))

        # ########## PIPELINE
        max_freq = max(int(len(ventanas[ventana]) * factor_frecuencia), n_documentos_maximos)

        vect = CountVectorizer(tokenizer=limpiarTextoTweet, binary=True, min_df=max_freq, ngram_range=(2, 3))


        data_transform = Pipeline([('counts', vect),
                                   ('filtrar', FiltroNGramasTransformer(numMagico=3, vectorizer=vect)),
                                   ('matrizdist', BinaryToDistanceTransformer(_norm='l2',_metric='euclidean'))])


        # ######## CLUSTERING

        start = time.time()
        X = data_transform.fit_transform(ventanas[ventana])
        end = time.time()
        print("tiempo pipeline: {} seg".format(end - start))
        print("Tweets ventana {} vs limpios {}".format(len(ventanas[ventana]), X.shape[0]))

        imprimirDendogramas(X)

        dt = 0.1
        print("AVERAGE")
        start = time.time()

        L = fastcluster.linkage(X, method='average')
        T = sch.to_tree(L)
        print("hclust cut threshold:", T.dist * dt)
        indL = sch.fcluster(L, T.dist * dt, 'distance')
        freqTwCl = Counter(indL)
        end = time.time()
        print("tiempo clustering: {} seg".format(end - start))
        # mostrarNTweetsCluster(3, data_transform, indL)

        Xclean = data_transform.named_steps['filtrar'].Xclean
        inv_map = data_transform.named_steps['filtrar'].inv_map

         #muestra ngramas más repetidos por cluster
        # mng[0] es main_ngram_in_cluster, tiene info de # tweets por clust, tweet + repr. por clust, ngramas del clust
        # mng[1] son los centroides de los clusters
        mng = mostrarNGramas(Xclean, freqTwCl, indL, inv_map)
        centroides = mng[0]
        ngramas_centroides = mng[1]
        cuenta = mng[2] # número de tweets por cluster

        for i in range(centroides.shape[0]):
            cercano = 0
            distancia = distance.euclidean(centroides[i], Xclean[0, :])
            for k in range(1, len(Xclean)):
                distActual = distance.euclidean(centroides[i], Xclean[k, :])
                if distActual < distancia:
                    cercano = k
                    distancia = distActual
            tweet = tweets_cluster[ventana][data_transform.named_steps['filtrar'].map_index_after_cleaning.get(cercano)]
            print("\nEl tweet mas cercano del cluster {} es {}".format(i + 1, tweet))



        # cuenta = [mng[0][cl][0] for cl in range(len(mng[0]))] # rescato el num de tweets por cluster

        # segunda clusterización
        #
        # ngramas_clusters = [mng[0][cl][2] for cl in range(len(mng[0]))]
        # print(ngramas_clusters)
        # ngramas_clusters = [[ngramas_clusters[cl][i][0] for i in range(len(ngramas_clusters[cl]))] for cl in range(len(ngramas_clusters)) ]
        # print(ngramas_clusters)
        # ngr_clu=[]
        # for i in range(len(ngramas_clusters)):
        #     ngrama_string=''
        #     for j in range(len(ngramas_clusters[i])):
        #         ngrama_string+=ngramas_clusters[i][j]
        #         ngrama_string+=' , '
        #     ngr_clu.append(ngrama_string)
        # print(ngr_clu)
        #
        # vect2 = CountVectorizer(tokenizer=tokens2daClust, binary=True, min_df=1, ngram_range=(1,1))
        #
        # data_transform2 = Pipeline([('counts', vect2), ('filtrar', FiltroNGramasTransformer(numMagico=2, vectorizer=vect2)),
        #                                     ('matrizdist', BinaryToDistanceTransformer(_norm='l2', _metric='euclidean'))])
        # X2 = data_transform2.fit_transform(ngr_clu)

        data_transform2 = Pipeline([('matrizdist', BinaryToDistanceTransformer(_norm='l2', _metric='euclidean'))])
        X_centroides = data_transform2.fit_transform(centroides)

        L2 = fastcluster.linkage(X_centroides, method='ward')
        dt2 = 0.3 # variarle a éste
        T2 = sch.to_tree(L2)

        print("hclust cut threshold:", T2.dist * dt2)
        indL2 = sch.fcluster(L2, T2.dist * dt2, 'distance')
        freqTwCl2 = Counter(indL2)
        #hasta aquí creo que la 2da clust está bien

        res = mostrarNGramas(centroides, freqTwCl2, indL2 , [])

        print("NGRAMAS de los clusters nuevos: ")
        mostrarNGramas2( indL, indL2, centroides, cuenta, inv_map)

        nuevos_centroides = res[0]

        # cambiar por top n tweets
        for i in range(nuevos_centroides.shape[0]):
            cercano = 0
            distancia = distance.euclidean(nuevos_centroides[i], Xclean[0, :])
            for k in range(1, len(Xclean)):
                 distActual = distance.euclidean(nuevos_centroides[i], Xclean[k, :])
                 if distActual < distancia:
                    cercano = k
                    distancia = distActual
            tweet = tweets_cluster[ventana][data_transform.named_steps['filtrar'].map_index_after_cleaning.get(cercano)]
            print("\nEl tweet mas cercano del cluster {} es {}".format(i+1, tweet))
            # print("\nLos ngramas del tweet {} son {}".format(i, ngramas_centroides[i]))

        imprimirDendogramas(X_centroides)

        plt.show()
        # mostrarNTweetsCluster2(3, data_transform, indL, indL2)


        ## revisar diferencia en este, que falla:
        #mng2 = mostrarNGramas(data_transform2)

        #falta...


        # ver a qué cluster pertenece un nuevo tweet (ejemplo a continuación):

        # tuit = "Me ha gustado un video de @youtube."  # este no lo clasifica bien, sino que lo clasifica al cluster basura. SOLUCIÓN: no considerar al cluster basura.
        # c = clusterDelTweet(tuit, centroides, cuenta)
        # print("El nuevo tweet {} pertenece al cluster {}.\n".format(tuit, c))
        #
        # tuit = "La alianza pan prd se está llevando a cabo."
        # c = clusterDelTweet(tuit, centroides, cuenta)
        # print("El nuevo tweet {} pertenece al cluster {}.\n".format(tuit, c))
        #
        # tuit = "Se está construyendo un nuevo modelo económico que mejore al país."
        # c = clusterDelTweet(tuit, centroides, cuenta)
        # print("El nuevo tweet {} pertenece al cluster {}.\n".format(tuit, c))
        #
        # tuit = "Ya listos para la copa corona mx?"
        # c = clusterDelTweet(tuit, centroides, cuenta)
        # print("El nuevo tweet {} pertenece al cluster {}.\n".format(tuit, c))


