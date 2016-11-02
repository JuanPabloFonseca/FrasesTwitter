#!/usr/bin/python3.5

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from .LimpiarTweets import limpiarTextoTweet, quitarAcentos, quitarEmoticons, steamWord
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
    def contieneElemento(self, matriz, elemento):
        encontrado = False
        indice_encontrado = -1

        for k in range(len(matriz)):
            if elemento in matriz[k]:
                encontrado = True
                indice_encontrado = k
                break
        return encontrado, indice_encontrado

    def mostrarNGramas2(self, indL, indL2, centroides, cuenta, inv_map): # sólo muestra los ngramas de cada cluster (2da clusterizacion)
        # Mostrar los n tweets de cada cluster
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
            # print("Ngramas del cluster {0}: ".format(f+1))
            for ng in range(len(frecuencia_ngramas_total[f])):
                if frecuencia_ngramas_total[f][ng] > 0:
                    lista_ngramas_por_cluster[f][inv_map[ng]]=frecuencia_ngramas_total[f][ng]
            ## ANTES, sin agrupacion:
            ## lista_ngramas_por_cluster[f] = sorted(lista_ngramas_por_cluster[f].items(), key=lambda x: x[1], reverse=True)

            ### AGRUPACION DE NGRAMAS
            # ('agustin basave', 14.0), ('agustin basave renuncia', 14.0), ('renuncia agustin', 14.0) -> ('agustin basave renuncia', 14.0)
            # ('alianza pan') ('alianzas pan') -> ('(alianza OR alianzas) pan')

            print("Cluster {}, ngramas: {}".format(f, [l[0] for l in lista_ngramas_por_cluster[f].items()]))

            ngramas = [l[0] for l in lista_ngramas_por_cluster[f].items()]
            cantidades = [l[1] for l in lista_ngramas_por_cluster[f].items()]

            # agrupando conjuntos similares de ngramas
            indices_conjunto = []
            num_conjuntos = 1
            for i in range(len(ngramas)):
                if not self.contieneElemento(indices_conjunto, i)[0]:  # si el ngrama no existe en algún conjunto
                    indices_conjunto.append([])
                    indices_conjunto[num_conjuntos - 1].append(i)
                    num_conjuntos = num_conjuntos + 1

                for j in range(len(ngramas)):
                    if i != j:
                        if set([steamWord(x) for x in ngramas[i].split()]).issuperset(
                                set([steamWord(x) for x in ngramas[j].split()])):  # es super conjunto
                            encontrado, indice_encontrado = self.contieneElemento(indices_conjunto, i)
                            encontradoj, indice_encontradoj = self.contieneElemento(indices_conjunto, j)

                            j_estaSolo = len(indices_conjunto[indice_encontradoj]) == 1

                            if encontrado and not encontradoj:
                                indices_conjunto[indice_encontrado].append(j)
                            elif encontrado and encontradoj and j_estaSolo:
                                indices_conjunto[indice_encontrado].append(j)
                                indices_conjunto.remove(indices_conjunto[indice_encontradoj])
                                num_conjuntos = num_conjuntos - 1

            # hasta aqui indices_conjunto agrupa indices de palabras que pertenezcan a un mismo superconjunto
            superconjuntos = []
            for c in indices_conjunto:
                superconjunto = set()

                minimo = cantidades[c[0]]
                for i in c:
                    superconjunto = superconjunto.union(ngramas[i].split())
                    minimo = min(minimo, cantidades[i])  # se queda con la cuenta minima del conjunto

                # crear regla OR para palabras con igual raíz
                elementos = [[x] for x in list(superconjunto)]

                reglaConjunto = ''
                indices_yaAsociados = []
                for i in range(len(elementos)):
                    if not i in indices_yaAsociados:
                        for j in range(len(elementos)):
                            if i != j:
                                # del conjunto identifica palabras que tengan la misma raiz de palabra y las agrupa en un OR
                                # REVISAR ver la posibilidad de considerar agrupar conjugaciones del mismo verbo
                                if elementos[i][0][0] == elementos[j][0][0] and steamWord(
                                        elementos[i][0]) == steamWord(elementos[j][0]):
                                    elementos[i].append(elementos[j][0])
                                    indices_yaAsociados.append(j)

                # elimina la palabra que fue asociada por el OR
                temp = 0
                for i in indices_yaAsociados:
                    try:
                        elementos.remove(elementos[i-temp])
                        temp=temp+1
                    except IndexError:
                        print("Revisar elementos {}, indice a eliminar {}".format(elementos,i-temp))

                # stringtify de elementos, para mostrar la regla
                ands = ''
                for e in range(len(elementos)):
                    if len(elementos[e]) > 1:
                        ors = '('
                    else:
                        ors = ''
                    for l in range(len(elementos[e])):
                        ors = ors + elementos[e][l]
                        if l < len(elementos[e]) - 1:
                            ors = ors + ' OR '
                        else:
                            if len(elementos[e]) > 1:
                                ors = ors + ')'
                    ands = ands + ors
                    if e < len(elementos) - 1:
                        ands = ands + ' '

                superconjuntos.append((ands, minimo))
            lista_ngramas_por_cluster[f] = sorted(superconjuntos, key=lambda x: x[1], reverse=True)
            ## FINALIZA AGRUPACION

            # print(lista_ngramas_por_cluster[f])

        return lista_ngramas_por_cluster

    def mostrarNGramas(self, Xclean, inv_map, map_index_after_cleaning, freqTwCl, indL, tweets_cluster, ventana):

        main_ngram_in_cluster = [-1] * len(freqTwCl)
        num_ngram_total = [-1] * len(freqTwCl) # guarda la frecuencia de los ngramas PARA TODOS LOS CLUSTERS.
        cuenta = [0] * len(freqTwCl)
        centroide = np.zeros((len(freqTwCl), Xclean.shape[1]))


        for clust in range(len(freqTwCl)):
            num_ngram = {}  # [0] * data_transform.named_steps['filtrar'].Xclean.shape[1]
            cont = 0
            for tweet in range(Xclean.shape[0]):
                if indL[tweet] == clust + 1:
                    cont += 1
                    for i in range(Xclean.shape[1]):
                        centroide[clust][i] += Xclean[tweet][i]
                        if len(inv_map) > 0:
                            if inv_map[i] in num_ngram.keys():
                                num_ngram[inv_map[i]] += Xclean[tweet][i]
                            elif Xclean[tweet][i] > 0:
                                num_ngram[inv_map[i]] = Xclean[tweet][i]
            if len(inv_map) > 0:
                num_ngram = sorted(num_ngram.items(), key=lambda x: x[1], reverse=True)

            centroide[clust] = [(x / cont) for x in centroide[clust]]

            num_ngram_total[clust] = num_ngram  # se guarda num_ngram. Al final, num_ngram_total tiene la info de TODOS los clusters
            cuenta[clust]=cont
        return [centroide, cuenta]


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

            dt = 0.1
            print("Primera clusterizacion")
            start = time.time()

            # ward,
            L = fastcluster.linkage(X, method='average')
            T = sch.to_tree(L)
            indL = sch.fcluster(L, T.dist * dt, 'distance')
            freqTwCl = Counter(indL)
            Xclean = data_transform.named_steps['filtrar'].Xclean
            inv_map = data_transform.named_steps['filtrar'].inv_map
            map_index_after_cleaning = data_transform.named_steps['filtrar'].map_index_after_cleaning

            # se envia 2 veces xclean
            mng = self.mostrarNGramas(Xclean, inv_map, map_index_after_cleaning, freqTwCl, indL, tweets_cluster, ventana)

            centroides_primera = mng[0]
            cuenta = mng[1]

            data_transform2 = Pipeline([('matrizdist', BinaryToDistanceTransformer(_norm='l2', _metric='euclidean'))])
            X_centroides = data_transform2.fit_transform(centroides_primera)


            LinkageMatrix = fastcluster.linkage(X_centroides, method='ward')

            return LinkageMatrix, tweets_cluster, centroides_primera, Xclean, inv_map, map_index_after_cleaning, indL, cuenta



    def obtenerTopicos(self, LinkageMatrix, threshold, centroides_primera, Xclean, inv_map, map_index_after_cleaning, tweets_cluster, indL, cuenta):

        start = time.time()
        print("Segunda clusterizacion")

        T2 = sch.to_tree(LinkageMatrix)

        print("hclust cut threshold:", T2.dist * threshold)

        indL2 = sch.fcluster(LinkageMatrix, T2.dist * threshold, 'distance')
        freqTwCl2 = Counter(indL2)

        end = time.time()
        print("tiempo clustering: {} seg".format(end - start))


        # mostrarNTweetsCluster(3, data_transform, indL)
        res = self.mostrarNGramas(centroides_primera, inv_map, map_index_after_cleaning, freqTwCl2, indL2, tweets_cluster,0)

        lista_ngramas_cluster = self.mostrarNGramas2( indL, indL2, centroides_primera, cuenta, inv_map)

        main_ngram_in_cluster = [-1] * len(freqTwCl2)

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
            tweet = tweets_cluster[0][map_index_after_cleaning.get(cercano)]
            # print("\nEl tweet mas cercano del cluster {} es {}".format(i+1, tweet))
            main_ngram_in_cluster[i] = [cuenta[i], tweet, lista_ngramas_cluster[i]]

        #mng[0] es main_ngram_in_cluster, tiene info de # tweets por clust, tweet + repr. por clust, ngramas del clust
        #mng[1] son los centroides de los clusters

        return [main_ngram_in_cluster, nuevos_centroides] #regresa main_ngram_in_cluster



    def clusterDelTweet(self,tw,centroides,tweetsEnClusters,inv_map):
        stop_words = creaStopWords()
        tw = limpiarTextoTweet(tw, stop_words)
        vectortweet=[0]*len(inv_map)
        for ng in range(len(inv_map)):
            nglim=re.sub('[@#]', '',inv_map[ng])
            ngsp=nglim.split()
            for p in range(len(tw)-len(ngsp)+1):
                pal=[re.sub('[@#]', '',tw[p+pa]) for pa in range(len(ngsp))]
                if set(ngsp) == set(pal):
                    vectortweet[ng] += 1

        # clusterBasura = np.argmax(tweetsEnClusters)
        cercano = -1
        dist = 1000000000
        for c in range(len(centroides)):
        #    if c != clusterBasura:
            distActual = distance.euclidean(centroides[c], vectortweet)
            if (distActual < dist):
                dist = distActual
                cercano = c

        #regresa el número de cluster al que el tweet "pertenece" (recordando que la numeración empieza desde 1)
        return (cercano+1)