#!/usr/bin/python3.5

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from LimpiarTweets import limpiarTextoTweet, quitarAcentos, quitarEmoticons
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

def mostrarNTweetsCluster(N, data_transform, indL):
    #Mostrar los n tweets de cada cluster
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

def mostrarNGramas(data_transform):
    Xclean = data_transform.named_steps['filtrar'].Xclean

    inv_map = data_transform.named_steps['filtrar'].inv_map

    # obtención del (los) ngrama(s) MÁS repetido(s) en cada cluster
    print(inv_map)
    main_ngram_in_cluster = [-1] * len(freqTwCl)
    centroide = []
    for clust in range(len(freqTwCl)):
        num_ngram = {}  # [0] * data_transform.named_steps['filtrar'].Xclean.shape[1]
        centroide.append([0] * data_transform.named_steps['filtrar'].Xclean.shape[1])
        cont = 0
        tweets_del_cluster = []
        for tweet in range(data_transform.named_steps['filtrar'].Xclean.shape[0]):
            if indL[tweet] == clust + 1:
                tweets_del_cluster.append(tweet)
                cont += 1
                for i in range(data_transform.named_steps['filtrar'].Xclean.shape[1]):
                    centroide[clust][i] += data_transform.named_steps['filtrar'].Xclean[tweet][i]
                    if inv_map[i] in num_ngram.keys():
                        num_ngram[inv_map[i]] += data_transform.named_steps['filtrar'].Xclean[tweet][i]
                    elif data_transform.named_steps['filtrar'].Xclean[tweet][i] > 0:
                        num_ngram[inv_map[i]] = data_transform.named_steps['filtrar'].Xclean[tweet][i]
        print("\n{} Tweets en Cluster {}".format(cont, clust + 1))

        centroide[clust] = [(x / cont) for x in centroide[clust]]
        cercano = -1
        distancia = 10000000
        print(tweets_del_cluster)
        for k in range(len(tweets_del_cluster)):
            actual = tweets_del_cluster[k]
            distActual = distance.euclidean(centroide[clust], data_transform.named_steps['filtrar'].Xclean[actual, :])
            if distActual < distancia:
                cercano = actual
                distancia = distActual

        # print("CENTROIDE {}".format(centroide[clust]))
        print("Tweet más representativo: {}".format(
            tweets_cluster[ventana][data_transform.named_steps['filtrar'].map_index_after_cleaning.get(cercano)]))

        # maximos = (np.argwhere(num_ngram == np.amax(num_ngram))).flatten().tolist()


        num_ngram = sorted(num_ngram.items(), key=lambda x: x[1], reverse=True)

        main_ngram_in_cluster[clust] = num_ngram
        # for m in range(len(maximos)):
        #    main_ngram_in_cluster[clust].append(inv_map[maximos[m]])

    for i in range(len(main_ngram_in_cluster)):
        print("Ngrama(s) más repetido(s) en el cluster ", (i + 1), ": ", main_ngram_in_cluster[i])

    return centroide

def clusterDelTweet(tw,centroides):
    tw = limpiarTextoTweet(tw, stop_words)
    inv_map = data_transform.named_steps['filtrar'].inv_map
    vectortweet=[0]*len(inv_map)
    for ng in range(len(inv_map)):
        nglim=re.sub('[@#]', '',inv_map[ng])
        ngsp=nglim.split()
        for p in range(len(tw)-len(ngsp)+1):
            pal=[re.sub('[@#]', '',tw[p+pa]) for pa in range(len(ngsp))]
            if set(ngsp) == set(pal):
                vectortweet[ng]+=1

    cercano=0
    dist=distance.euclidean(centroides[cercano],vectortweet)
    for c in range(1,len(centroides)):
        distActual=distance.euclidean(centroides[c],vectortweet)
        if(distActual<dist):
            dist=distActual
            cercano=c

    #regresa el número de cluster al que el tweet "pertenece" (recordando que la numeración empieza desde 1)
    return (cercano+1)



if __name__ == "__main__":

    ########### PARAMETROS

    time_window_mins = 14400.0
    n_documentos_maximos = 5
    factor_frecuencia = 0.001
    num_ngrams_in_tweet = 3
    minimo_usuarios = 3
    minimo_hashtags = 3
    ngrama_minimo = 2
    ngrama_maximo = 4

    ### ngramas

    ########### LEER ARCHIVO

    #corpus = []
    total_tweets_norepetidos=[]
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

        ########### PIPELINE
        max_freq = max(int(len(ventanas[ventana]) * factor_frecuencia), n_documentos_maximos)


        vect = CountVectorizer(tokenizer=limpiarTextoTweet, binary=True, min_df=max_freq, ngram_range=(1, 3))

        data_transform = Pipeline([('counts', vect),
                                   ('filtrar', FiltroNGramasTransformer(numMagico=3,vectorizer=vect)),
                                   ('matrizdist', BinaryToDistanceTransformer(_norm='l2',_metric='euclidean'))])


        ######### CLUSTERING

        start = time.time()
        X = data_transform.fit_transform(ventanas[ventana])
        end = time.time()
        print("tiempo pipeline: {} seg".format(end - start))
        print("Tweets ventana {} vs limpios {}".format(len(ventanas[ventana]), X.shape[0]))

    dt = 0.5
    print("AVERAGE")
    start = time.time()
    L = fastcluster.linkage(X, method='average')
    T = sch.to_tree(L)
    print("hclust cut threshold:", T.dist * dt)
    indL = sch.fcluster(L, T.dist * dt, 'distance')
    freqTwCl = Counter(indL)
    end = time.time()
    print("tiempo clustering: {} seg".format(end - start))
    mostrarNTweetsCluster(3, data_transform, indL)
    centroides=mostrarNGramas(data_transform) #muestra ngramas más repetidos por cluster, y regresa los centroides de cada cluster

    #ver a qué cluster pertenece un nuevo tweet (ejemplo a continuación):
    tuit = "Me ha gustado un video de @youtube." #este no lo clasifica bien, sino que lo clasifica al cluster basura. SOLUCIÓN: no considerar al cluster basura.
    c=clusterDelTweet(tuit,centroides)
    print("El nuevo tweet {} pertenece al cluster {}.\n".format(tuit,c))

    tuit = "La alianza pan prd se está llevando a cabo."
    c = clusterDelTweet(tuit, centroides)
    print("El nuevo tweet {} pertenece al cluster {}.\n".format(tuit, c))

    tuit = "Se está construyendo un nuevo modelo económico que mejore al país."
    c = clusterDelTweet(tuit, centroides)
    print("El nuevo tweet {} pertenece al cluster {}.\n".format(tuit, c))

    tuit = "Ya listos para la copa corona mx?"
    c = clusterDelTweet(tuit, centroides)
    print("El nuevo tweet {} pertenece al cluster {}.\n".format(tuit, c))