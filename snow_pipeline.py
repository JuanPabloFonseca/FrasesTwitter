#!/usr/bin/python3.5

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from LimpiarTweets import limpiarTextoTweet, quitarAcentos, quitarEmoticons, tokens2daClust, steamWord
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

def contieneElemento(matriz, elemento):
    encontrado = False
    indice_encontrado = -1

    for k in range(len(matriz)):
        if elemento in matriz[k]:
            encontrado = True
            indice_encontrado = k
            break
    return encontrado, indice_encontrado

#Solo se utiliza en la segunda clusterización, obtener los ngramas de los clusters de centroides
def mostrarNGramas2(indL2, centroides, cuenta, inv_map): # sólo muestra los ngramas de cada cluster (2da clusterizacion)
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
        #print("Ngramas del cluster {0}: ".format(f+1))
        for ng in range(len(frecuencia_ngramas_total[f])):
            if frecuencia_ngramas_total[f][ng] > 0:
                lista_ngramas_por_cluster[f][inv_map[ng]]=frecuencia_ngramas_total[f][ng]


        ### PRIMER INTENTO DE AGRUPACION
        # ERROR:
        # ['nacional prd', 'agustin basave dijo', 'prd agustin', 'agustin basave', 'alianza pan', 'basave dijo', 'pan prd', 'alianza pan prd', 'prd agustin basave']
        # [[0, 2, 6, 7, 8, 1, 3, 5, 4], [1, 2, 3, 5, 8], [4, 6, 7]]

        # filtro de ngramas, convertir
        # ('agustin basave', 14.0), ('agustin basave renuncia', 14.0), ('renuncia agustin', 14.0) -> ('agustin basave renuncia', 14.0)
        ngrama_mayor = []
        nueva_lista_ngramas = {}

        print("Cluster {}, ngramas: {}".format(f, [l[0] for l in lista_ngramas_por_cluster[f].items()]))

        ngramas = [l[0] for l in lista_ngramas_por_cluster[f].items()]
        cantidades = [l[1] for l in lista_ngramas_por_cluster[f].items()]

        # agrupando conjuntos similares de ngramas
        indices_conjunto = []
        num_conjuntos = 1
        for i in range(len(ngramas)):
            if not contieneElemento(indices_conjunto,i)[0]: # si el ngrama no existe en algún conjunto
                indices_conjunto.append([])
                indices_conjunto[num_conjuntos-1].append(i)
                num_conjuntos = num_conjuntos + 1

            for j in range(len(ngramas)):
                if i != j:
                    if set([steamWord(x) for x in ngramas[i].split()]).issuperset([steamWord(x) for x in ngramas[j].split()]): #es interseccion
                        encontrado, indice_encontrado = contieneElemento(indices_conjunto,i)
                        encontradoj, indice_encontradoj = contieneElemento(indices_conjunto, j)

                        j_estaSolo = len(indices_conjunto[indice_encontradoj])==1

                        if encontrado and not encontradoj:
                            indices_conjunto[indice_encontrado].append(j)
                        elif encontrado and encontradoj and j_estaSolo:
                            indices_conjunto[indice_encontrado].append(j)
                            indices_conjunto.remove(indices_conjunto[indice_encontradoj])
                            num_conjuntos = num_conjuntos - 1
                    # else: # no es interseccion, crear nuevo conjunto


        superconjuntos = []
        for c in indices_conjunto:
            superconjunto = set()

            minimo = cantidades[c[0]]
            for i in c:
                superconjunto = superconjunto.union(ngramas[i].split())
                minimo = min(minimo, cantidades[i])

            # crear regla OR para palabras con igual raíz
            elementos = [[x] for x in list(superconjunto)]

            reglaConjunto = ''
            indices_yaAsociados = []
            for i in range(len(elementos)):
                # nuevaRegla = elementos[i]
                if not i in indices_yaAsociados:
                    for j in range(len(elementos)):
                         if i!=j:
                             if elementos[i][0][0] == elementos[j][0][0] and steamWord(elementos[i][0]) == steamWord(elementos[j][0]):
                                elementos[i].append(elementos[j][0])
                                indices_yaAsociados.append(j)

            for i in indices_yaAsociados:
                elementos.remove(elementos[i])

            superconjuntos.append((elementos, minimo))

            ## FINALIZA INTENTO AGRUPACION

        lista_ngramas_por_cluster[f] = sorted(superconjuntos, key=lambda x: x[1], reverse=True)
        # print(lista_ngramas_por_cluster[f])

    return lista_ngramas_por_cluster # es una lista de listas de ngramas por cluster (nuevo)

def mostrarNGramas(Xclean, freqTwCl, indL, inv_map):
    num_ngram_total = [-1] * len(freqTwCl) # guarda la frecuencia de los ngramas PARA TODOS LOS CLUSTERS.
    cuenta = [0] * len(freqTwCl)
    centroide = np.zeros((len(freqTwCl), Xclean.shape[1]))
    for clust in range(len(freqTwCl)):
        num_ngram = {}  # [0] * data_transform.named_steps['filtrar'].Xclean.shape[1]
        cont = 0
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

def imprimirDendogramas(X, metodoDistancia):
    hclust_methods = ["ward", "median", "centroid", "weighted", "single", "complete", "average"]
    show_leaf_counts = True
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('3 temas. [2,3] ngramas. Distancia ' + metodoDistancia + '.', fontsize=20)
    # plt.clf()
    for a in range(7):
        plt.subplot(3, 3, a+1)
        plt.title(hclust_methods[a])
        hc_iris = fastcluster.linkage(X, method=hclust_methods[a])
        sch.dendrogram(hc_iris,
           color_threshold=1,
           # p=6,
           # truncate_mode='lastp',
           show_leaf_counts=show_leaf_counts)

# genera un query de acuerdo a los clusters de interés (y con la especificidad del threshold)
def generaQuery(noClusters,nuevos_ngramas, threshold):
    #primero revisa que threshold esté entre 0 y 1
    if (threshold > 1):
        threshold = 1
    if (threshold < 0):
        threshold = 0
    query = ""
    for i in range(len(nuevos_ngramas)):
        if (i+1) in noClusters:
            if query:
                query = query + "OR "
            query = query + "("
            for j in range(len(nuevos_ngramas[i])):
                if(nuevos_ngramas[i][j][1] >= nuevos_ngramas[i][0][1]*(1-threshold)):
                    ng=re.split("[ @#]",nuevos_ngramas[i][j][0])
                    ng=' '.join([x for x in ng if x])  # elimina strings vacíos
                    query = query + "\"" + ng + "\" "
                else:
                    break
            query = query + ") "
    return query


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
    archivo = open('PanCoronaModelojsons/3temas.txt')
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
            if len(users) < minimo_usuarios and len(hashtags) < minimo_hashtags and len(features) > 3:
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
        vect = CountVectorizer(tokenizer=limpiarTextoTweet, # binary=True,
                               min_df=max_freq, ngram_range=(2, 3))
        data_transform = Pipeline([('counts', vect),
                                   ('filtrar', FiltroNGramasTransformer(numMagico=3, vectorizer=vect)),
                                   ('matrizdist', BinaryToDistanceTransformer(_norm='l2',_metric='euclidean'))])

        # ######## PRIMERA CLUSTERIZACION
        print("\nPRIMERA CLUSTERIZACION\n")
        X = data_transform.fit_transform(ventanas[ventana])

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

        Xclean = data_transform.named_steps['filtrar'].Xclean
        inv_map = data_transform.named_steps['filtrar'].inv_map

        #muestra ngramas más repetidos por cluster
        # mng[0] es main_ngram_in_cluster, tiene info de # tweets por clust, tweet + repr. por clust, ngramas del clust
        # mng[1] son los centroides de los clusters
        mng = mostrarNGramas(Xclean, freqTwCl, indL, inv_map)
        centroides = mng[0]
        ngramas_centroides = mng[1]
        cuenta = mng[2] # número de tweets por cluster

        cercanos = np.zeros((centroides.shape[0], Xclean.shape[0]))
        for i in range(centroides.shape[0]):
            for k in range(0, len(Xclean)):
                distActual = distance.euclidean(centroides[i], Xclean[k, :])
                cercanos[i][k]=distActual

        n = 5
        for i in range(centroides.shape[0]):
            tweets_cerca = sorted([(l, k) for k, l in enumerate(cercanos[i])], key=lambda y: y[0])
            n_min = min(int(ngramas_centroides[i][0][1]), n)
            print("\nNgramas clusters {}: {}".format(i, ngramas_centroides[i]))
            for j in range(n_min):
                tweet = tweets_cluster[ventana][data_transform.named_steps['filtrar'].map_index_after_cleaning.get(tweets_cerca[j][1])]
                print("Tweet cercano al cluster {} es {}".format(i, tweet))

        #### SEGUNDA CLUSTERIZACION
        print("\nSEGUNDA CLUSTERIZACION\n")
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
        nuevos_ngramas = mostrarNGramas2(indL2, centroides, cuenta, inv_map)
        nuevos_centroides = res[0]
        
        # cambiar por top n tweets
        cercanos = np.zeros((nuevos_centroides.shape[0], Xclean.shape[0]))
        for i in range(nuevos_centroides.shape[0]):
            for k in range(0, len(Xclean)):
                distActual = distance.euclidean(nuevos_centroides[i], Xclean[k, :])
                cercanos[i][k] = distActual

        n = 5
        for i in range(nuevos_centroides.shape[0]):
            tweets_cerca = sorted([(l, k) for k, l in enumerate(cercanos[i])], key=lambda y: y[0])
            n_min = min(int(nuevos_ngramas[i][0][1]), n) # falta generar cuenta 2
            print("Ngramas clusters {}: {}".format(i, nuevos_ngramas[i]))
        #     for j in range(n_min):
        #         tweet = tweets_cluster[ventana][
        #             data_transform.named_steps['filtrar'].map_index_after_cleaning.get(tweets_cerca[j][1])]
        #         print("Tweet cercano al cluster {} es {}".format(i, tweet))

        imprimirDendogramas(X_centroides, metodoDistancia='euclideana')
        plt.show()

        # generación del query basado en los clusters de interés
        # eleccion = input("\n\nÉchale un vistazo a los clusters arriba mostrados. "
        #                  "¿Cuáles de ellos te interesan? Indícalo a continuación (separado con espacios):")
        # noClusters = eleccion.split()
        # for i in range(len(noClusters)):
        #     noClusters[i]=int(noClusters[i])
        #
        # threshold = 0.4  # qué tantos ngramas de cada cluster se quieren considerar (porcentaje de tweets que tienen ese ngrama) 1 = todos.
        # print("\nEl query de tu interés es el siguiente: ")
        # print(generaQuery(noClusters,nuevos_ngramas,threshold))


        # ver a qué cluster pertenece un nuevo tweet (ejemplo a continuación):
        # tuit = "Ya listos para la copa corona mx?"
        # c = clusterDelTweet(tuit, centroides, cuenta)
        # print("El nuevo tweet {} pertenece al cluster {}.\n".format(tuit, c))


## POS Tag un tweet
# from nltk.tag import StanfordPOSTagger
# st = StanfordPOSTagger('spanish-distsim.tagger')
# st.tag('El exámen está muy perro')
