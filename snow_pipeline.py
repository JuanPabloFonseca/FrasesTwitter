from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from LimpiarTweets import limpiarTextoTweet, quitarAcentos, quitarEmoticons
from snow_datatransformer import BinaryToDistanceTransformer, FiltroNGramasTransformer
from misStopWords import creaStopWords
from datetime import datetime
import scipy.cluster.hierarchy as sch
import fastcluster
import re

########### PARAMETROS

time_window_mins = 30.0
n_documentos_maximos = 5
factor_frecuencia = 0.01
num_ngrams_in_tweet = 3

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
archivo = open('json/pan_timeordered.txt')
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
        if len(users) < 3 and len(hashtags) < 3 and len(
                features) > 3:
            tweet_bag = tweet_bag[:-1]
            ventanas[ventana].append(tweet_bag)
    else:
        # incrementar ventana
        ventana += 1
        ventanas.append([])
        tweet_unixtime_old = tweet_unixtime


for corpus in ventanas:
    ########### PIPELINE
    max_freq = max(int(len(corpus) * factor_frecuencia), n_documentos_maximos)
    data_transform = make_pipeline(
        CountVectorizer(tokenizer=limpiarTextoTweet, binary=True,
                        min_df=max_freq, ngram_range=(1, 3)),
        FiltroNGramasTransformer(numMagico=3),
        BinaryToDistanceTransformer(_norm='l2',_metric='euclidean')
    )

    ######### CLUSTERING
    datos = data_transform.fit_transform(corpus)
    L = fastcluster.linkage(datos, method='average')
    dt = 0.5
    print("hclust cut threshold:", dt)
    indL = sch.fcluster(L, dt * datos.max(), 'distance')
    sch.dendrogram(L)

# idx_clusts = sorted([(l, k) for k, l in enumerate(indL)], key=lambda x: x[0])