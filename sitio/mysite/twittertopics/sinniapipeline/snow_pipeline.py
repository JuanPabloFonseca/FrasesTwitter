#!/usr/bin/python3.5

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from .LimpiarTweets import limpiarTextoTweet, quitarAcentos, quitarEmoticons
from .snow_datatransformer import BinaryToDistanceTransformer, FiltroNGramasTransformer
from collections import Counter
from .misStopWords import creaStopWords
from datetime import datetime
import numpy as np
import scipy.cluster.hierarchy as sch
import fastcluster
import re
import time

# def mostrarNTweetsCluster(N, data_transform, indL):
#     #Mostrar los n tweets de cada cluster
#     idx_clusts = sorted([(l, k) for k, l in enumerate(indL)], key=lambda x: x[0])
#     n_c = {}
#     for x in idx_clusts:
#         if not x[0] in n_c.keys() :
#             print("Tweets Cluster {0}".format(x[0]))
#             n_c[x[0]]=0
#
#         if n_c[x[0]] >= N:
#             continue
#         else:
#             n_c[x[0]] += 1
#             print(tweets_cluster[ventana][data_transform.named_steps['filtrar'].map_index_after_cleaning.get(x[1])])

def mostrarNGramas(data_transform, freqTwCl, indL):
    #obtención del (los) ngrama(s) más repetido(s) en cada cluster
    inv_map = {v: k for k, v in data_transform.named_steps['counts'].vocabulary_.items()}
    main_ngram_in_cluster=[-1]*len(freqTwCl)
    for clust in range(len(freqTwCl)):
        num_ngram = {} #[0] * data_transform.named_steps['filtrar'].Xclean.shape[1]
        cont=0
        for tweet in range(data_transform.named_steps['filtrar'].Xclean.shape[0]):
            if indL[tweet] == clust+1:
                cont+=1
                for i in range(data_transform.named_steps['filtrar'].Xclean.shape[1]):
                    if inv_map[i] in num_ngram.keys():
                        num_ngram[inv_map[i]]+=data_transform.named_steps['filtrar'].Xclean[tweet][i]
                    elif data_transform.named_steps['filtrar'].Xclean[tweet][i] > 0:
                        num_ngram[inv_map[i]]=data_transform.named_steps['filtrar'].Xclean[tweet][i]
        print("{} Tweets en Cluster {}".format(cont,clust+1))
        #print(num_ngram) #muestra las repeticiones de todos los ngramas por cada cluster
        # maximos = (np.argwhere(num_ngram == np.amax(num_ngram))).flatten().tolist()

        num_ngram = sorted(num_ngram.items(), key=lambda x:x[1], reverse=True)

        main_ngram_in_cluster[clust]= num_ngram
        # for m in range(len(maximos)):
        #    main_ngram_in_cluster[clust].append(inv_map[maximos[m]])
    #for i in range(len(main_ngram_in_cluster)):
    #    print("Ngrama(s) más repetido(s) en el cluster ", (i+1),": ",main_ngram_in_cluster[i])

    return main_ngram_in_cluster

    # sch.dendrogram(L)


def procesarArchivo(archivo, time_window_mins, n_documentos_maximos, factor_frecuencia, num_ngrams_in_tweet, minimo_usuarios, minimo_hashtags, ngrama_minimo, ngrama_maximo):
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
        vectorizer = CountVectorizer(tokenizer=limpiarTextoTweet, # binary=True,
                                                              min_df=max_freq, ngram_range=(ngrama_minimo, ngrama_maximo))
        data_transform = Pipeline([('counts', vectorizer),
                                   ('filtrar', FiltroNGramasTransformer(numMagico=num_ngrams_in_tweet, vectorizer=vectorizer)),
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

        # mostrarNTweetsCluster(3, data_transform, indL)
        return mostrarNGramas(data_transform,freqTwCl, indL)