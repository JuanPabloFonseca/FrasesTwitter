#!/usr/bin/python3.5

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from .LimpiarTweets import limpiarTextoTweet, quitarAcentos, quitarEmoticons
from .snow_datatransformer import BinaryToDistanceTransformer, FiltroNGramasTransformer
from collections import Counter
from .misStopWords import creaStopWords
from datetime import datetime
import numpy as np
from scipy.spatial import distance
import scipy.cluster.hierarchy as sch
import fastcluster
import re
import time

class pipeline:
    def mostrarNGramas(self, Xclean, inv_map, map_index_after_cleaning, freqTwCl, indL, tweets_cluster, ventana):
        #Xclean = data_transform.named_steps['filtrar'].Xclean
        #inv_map = data_transform.named_steps['filtrar'].inv_map

        # obtención del (los) ngrama(s) MÁS repetido(s) en cada cluster
        # print(inv_map)
        main_ngram_in_cluster = [-1] * len(freqTwCl)
        centroide = []
        for clust in range(len(freqTwCl)):
            num_ngram = {}  # [0] * data_transform.named_steps['filtrar'].Xclean.shape[1]
            centroide.append([0] * Xclean.shape[1])
            cont = 0
            tweets_del_cluster = []
            for tweet in range(Xclean.shape[0]):
                if indL[tweet] == clust + 1:
                    tweets_del_cluster.append(tweet)
                    cont += 1
                    for i in range(Xclean.shape[1]):
                        centroide[clust][i] += Xclean[tweet][i]
                        if inv_map[i] in num_ngram.keys():
                            num_ngram[inv_map[i]] += Xclean[tweet][i]
                        elif Xclean[tweet][i] > 0:
                            num_ngram[inv_map[i]] = Xclean[tweet][i]


            centroide[clust] = [(x / cont) for x in centroide[clust]]
            cercano = -1
            distancia = 10000000
            # print(tweets_del_cluster)
            for k in range(len(tweets_del_cluster)):
                actual = tweets_del_cluster[k]
                distActual = distance.euclidean(centroide[clust], Xclean[actual, :])
                if distActual < distancia:
                    cercano = actual
                    distancia = distActual

            # print("\n{} Tweets en Cluster {}".format(cont, clust + 1))


            num_ngram = sorted(num_ngram.items(), key=lambda x: x[1], reverse=True)
            main_ngram_in_cluster[clust] = [cont, tweets_cluster[ventana][map_index_after_cleaning.get(cercano)], num_ngram]
            # print("CENTROIDE {}".format(centroide[clust]))
            # print("Num. Tweets {}, tweet más representativo: {} y ngramas {}".format( cont,
            #    tweets_cluster[ventana][data_transform.named_steps['filtrar'].map_index_after_cleaning.get(cercano)]), num_ngram)
        return [main_ngram_in_cluster,centroide]


    def obtenerModelo(self, archivo, time_window_mins, n_documentos_maximos, factor_frecuencia, num_ngrams_in_tweet, minimo_usuarios, minimo_hashtags, ngrama_minimo, ngrama_maximo):
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

            # dt = 0.5
            # print("AVERAGE")

            start = time.time()
            LinkageMatrix = fastcluster.linkage(X, method='average')

            Xclean = data_transform.named_steps['filtrar'].Xclean
            inv_map = data_transform.named_steps['filtrar'].inv_map
            map_index_after_cleaning = data_transform.named_steps['filtrar'].map_index_after_cleaning

            return LinkageMatrix, tweets_cluster, Xclean, inv_map, map_index_after_cleaning



    def obtenerTopicos(self, LinkageMatrix, threshold, Xclean, inv_map, map_index_after_cleaning, tweets_cluster):
        T = sch.to_tree(LinkageMatrix)
        print("hclust cut threshold:", T.dist * threshold)
        indL = sch.fcluster(LinkageMatrix, T.dist * threshold, 'distance')
        freqTwCl = Counter(indL)
        end = time.time()
        # print("tiempo clustering: {} seg".format(end - start))

        # mostrarNTweetsCluster(3, data_transform, indL)
        mng=self.mostrarNGramas(Xclean, inv_map, map_index_after_cleaning,freqTwCl, indL, tweets_cluster,0)
        #mng[0] es main_ngram_in_cluster, tiene info de # tweets por clust, tweet + repr. por clust, ngramas del clust
        #mng[1] son los centroides de los clusters


        # DADO UN TWEET NUEVO, A QUÉ CLUSTER PERTENECE (3 ejemplos de "pan" a continuación):
        ##############################################
        # cuenta=[mng[0][cl][0] for cl in range(len(mng[0]))] # rescato el num de tweets por cluster
        #
        # tuit = "La alianza pan prd se está llevando a cabo."
        # c = clusterDelTweet(tuit, mng[1],cuenta,inv_map)
        # print("El nuevo tweet {} pertenece al cluster {}.\n".format(tuit, c))
        #
        # tuit = "Se está construyendo un nuevo modelo económico que mejore al país."
        # c = clusterDelTweet(tuit, mng[1],cuenta,inv_map)
        # print("El nuevo tweet {} pertenece al cluster {}.\n".format(tuit, c))
        #
        # tuit = "Ya listos para la copa corona mx?"
        # c = clusterDelTweet(tuit, mng[1],cuenta,inv_map)
        # print("El nuevo tweet {} pertenece al cluster {}.\n".format(tuit, c))
        ##############################################


        return mng[0] #regresa main_ngram_in_cluster



    def clusterDelTweet(tw,centroides,cnt,inv_map):
        stop_words = creaStopWords()
        tw = limpiarTextoTweet(tw, stop_words)
        vectortweet=[0]*len(inv_map)
        for ng in range(len(inv_map)):
            nglim=re.sub('[@#]', '',inv_map[ng])
            ngsp=nglim.split()
            for p in range(len(tw)-len(ngsp)+1):
                pal=[re.sub('[@#]', '',tw[p+pa]) for pa in range(len(ngsp))]
                if set(ngsp) == set(pal):
                    vectortweet[ng]+=1


        clusterBasura = np.argmax(cnt)
        cercano = -1
        dist = 1000000000
        for c in range(len(centroides)):
            if c != clusterBasura:
                distActual = distance.euclidean(centroides[c], vectortweet)
                if (distActual < dist):
                    dist = distActual
                    cercano = c

        #regresa el número de cluster al que el tweet "pertenece" (recordando que la numeración empieza desde 1)
        return (cercano+1)
