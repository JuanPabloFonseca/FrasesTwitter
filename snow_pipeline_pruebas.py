#!/usr/bin/python3.5

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from LimpiarTweets import limpiarTextoTweet, quitarAcentos, quitarEmoticons, tokens2daClust
from snow_datatransformer_pruebas import BinaryToDistanceTransformer, FiltroNGramasTransformer
from collections import Counter
from misStopWords import creaStopWords
from datetime import datetime
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial import distance
import fastcluster
import re
import time
import ObtenerTweets
import LimpiarTweets
import LDA_cluster
from sklearn.metrics import normalized_mutual_info_score


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

def clasifModelo(indL, indL2, M): # N = num tweets por cada cluster original, M es el número original de tweets
    # Mostrar los n tweets de cada cluster
    idx_clusts = sorted([(l, k) for k, l in enumerate(indL)], key=lambda x: x[0])
    idx_clusts2 = sorted([(l, k) for k, l in enumerate(indL2)], key=lambda y: y[0])
    '''pr=sorted([id[1] for id in idx_clusts])
    faltan=[]
    for i in range(pr[-1]+1):
        if i not in pr:
            faltan.append(i)
    print("faltan {}".format(faltan))'''

    clas = [-1]*M
    for y in idx_clusts2:
        tuitscluster=[x[1] for x in idx_clusts if x[0] == y[1] + 1]
        for k in range(len(tuitscluster)):
            clas[tuitscluster[k]]=y[0]-1
    return clas

def mostrarNGramas2(indL, indL2, centroides, cuenta, inv_map): # sólo muestra los ngramas de cada cluster (2da clusterizacion)
    # Mostrar los n tweets de cada cluster
    # idx_clusts = sorted([(l, k) for k, l in enumerate(indL)], key=lambda x: x[0])   # lista indexada de los los clusters originales (índice empieza en 1) vs los tweets (índice empieza en 0)
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

    # plt.figure(iris_dendlist[[i]], axes = False, horiz = True)
    show_leaf_counts = True

    # plt.title("Dendrogram")

    # plt.show()
    # plt.figure(1, figsize=(6, 5))


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

def purity(c_original,c_modelo):
    purity = 0
    for i in range(len(set(c_modelo))):
        cnt=[0]*len(set(c_original))
        for j in range(len(c_modelo)):
            if c_modelo[j] == i:
                cnt[c_original[j]]=cnt[c_original[j]]+1
        purity = purity + max(cnt)
    purity = purity/len(c_modelo)
    return purity


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
    #archivo = open('PanCoronaModelojsons/todos_tws.txt')

    archiv = ObtenerTweets.obtenerTweetsArchivo('test')
    clas = LDA_cluster.clasifOriginal("test") #clasificación original de todos
    clas_o=[]

    i=0
    for line in archiv:
        text = line
        # tweet_gmttime = datetime.strptime(contenido[0][1:], '%a %b %d %H:%M:%S %z %Y')  # contenido[0]
        # tweet_unixtime = tweet_gmttime.timestamp()
        # tweet_id = contenido[1]

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

        #if tweet_unixtime_old == -1:
        #    tweet_unixtime_old = tweet_unixtime


        if len(users) < minimo_usuarios and len(hashtags) < minimo_hashtags and len(
                features) > 3:
            tweet_bag = tweet_bag[:-1]
            ventanas[ventana].append(tweet_bag)
            tweets_cluster[ventana].append(text)
            clas_o.append(clas[i])
        i=i+1





    print("Ventanas: {}".format(len(ventanas)))
    for ventana in range(len(ventanas)):
        print("VENTANA {}".format(ventana))

        # ########## PIPELINE
        max_freq = max(int(len(ventanas[ventana]) * factor_frecuencia), n_documentos_maximos)

        vect = CountVectorizer(tokenizer=limpiarTextoTweet, binary=True, min_df=max_freq, ngram_range=(2, 3))


        data_transform = Pipeline([('counts', vect),
                                   ('filtrar', FiltroNGramasTransformer(numMagico=3, vectorizer=vect,clas_o=clas_o)),
                                   ('matrizdist', BinaryToDistanceTransformer(p=1,_norm='l2',_metric='euclidean'))]) # p=1 es la primera cl.


        # ######## CLUSTERING
        resuu = data_transform.fit_transform(ventanas[ventana])
        X=resuu[0]
        clasif_original=resuu[1]
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

        data_transform2 = Pipeline([('matrizdist', BinaryToDistanceTransformer(p=2,_norm='l2', _metric='euclidean'))]) # p=2 es la segunda cl.
        X_centroides = data_transform2.fit_transform(centroides)

        L2 = fastcluster.linkage(X_centroides, method='ward')
        dt2 = 0.3 # variarle a éste
        '''
        dt2     purity     NMI
        0.1     46.3%      5.6%
        0.2     41.8%      2.3%
        0.3     39.0%      1.1%
        0.4     37.8%      1.1%
        0.5     37.3%      0.43%
        0.6     36.9%      0.2%
        0.7     36.9%      0.0052%
        0.8     36.9%      0.0052%
        0.9     36.9%      0.0052%

        vemos que  NMI  es  mejor que LDA, sin importar el nivel de corte del dendograma, y mejora conforme el corte es más general.
        vemos que purity es peor que LDA, sin importar el nivel de corte del dendograma, y empeora conforme el corte es más general.
        esto sólo se hizo con una BD (la llamada "test"), habría que hacer pruebas con otras BD's
        '''

        T2 = sch.to_tree(L2)

        print("hclust cut threshold:", T2.dist * dt2)
        indL2 = sch.fcluster(L2, T2.dist * dt2, 'distance')
        freqTwCl2 = Counter(indL2)
        #hasta aquí creo que la 2da clust está bien

        res = mostrarNGramas(centroides, freqTwCl2, indL2 , [])

        print("NGRAMAS de los clusters nuevos: ")
        nuevos_ngramas = mostrarNGramas2( indL, indL2, centroides, cuenta, inv_map)

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

        imprimirDendogramas(X_centroides, metodoDistancia='euclideana')

        plt.show()

        clasif_modelo = clasifModelo(indL,indL2,len(clasif_original)) # len(clasif_original) sólo nos da el número de tweets que sí pasaron los filtros iniciales

        print(clasif_original[-200:])
        print(clasif_modelo[-200:])


        print("dt2: {}".format(dt2))
        print("Purity: {0}".format(purity(clasif_original, clasif_modelo)))
        print("NMI: {0}".format(normalized_mutual_info_score(clasif_original, clasif_modelo)))

        # generación del query basado en los clusters de interés
        eleccion = input("\n\nÉchale un vistazo a los clusters arriba mostrados. "
                         "¿Cuáles de ellos te interesan? Indícalo a continuación (separado con espacios): ")
        noClusters = eleccion.split()
        for i in range(len(noClusters)):
            noClusters[i]=int(noClusters[i])

        # print(noClusters)
        # print(nuevos_ngramas)
        threshold = 0.8  # qué tantos ngramas de cada cluster se quieren considerar (porcentaje de tweets que tienen ese ngrama) 1 = todos.
        print("\nEl query de tu interés es el siguiente: ")
        print(generaQuery(noClusters,nuevos_ngramas,threshold))
        # mostrarNTweetsCluster2(3, data_transform, indL, indL2)




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



## POS Tag un tweet
# from nltk.tag import StanfordPOSTagger
# st = StanfordPOSTagger('spanish-distsim.tagger')
# st.tag('El exámen está muy perro')
