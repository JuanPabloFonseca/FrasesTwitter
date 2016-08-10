#!/usr/bin/python3.5

# e-tweets, we re-place the text of the re-tweet with the original text of the tweet that was re-tweete
# filter out tweets that have more than 2 user mentions or more than 2 hashtags, or less than 4 text tokens
# we create a (binary) tweet-term matrix, where we remove user mentions (but keep hashtags)
# vocabulary terms are only bi-grams and tri-grams, that occur in at least a number of tweets, where the minimum is set to 10 tweets, max(int(len(window corpus)*0.0025),10)
# we reduce this matrix to only the subset of rows containing at least 3 terms


__author__ = 'gifrim'

# What this code does:
# Given a Twitter stream in JSON-to-text format, the time window size in minutes (e.g., 15 minutes)
# and the output file name, extract top 10 topics detected in the time window

# Example run:
# python twitter-topics-from-json-text-stream.py json-to-text-stream-syria.json.txt 15 15mins-topics-syria-stream.txt > details_clusters_15mins_topics_syria-stream.txt


from collections import Counter
# import CMUTweetTagger
from datetime import datetime
import fastcluster
from itertools import cycle
import json
import nltk
import numpy as np
import re
# import requests
import os
import scipy.cluster.hierarchy as sch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import metrics
# from stemming.porter2 import stem
import string
import sys
import time
import pandas as pd


import LimpiarTweets as li
import misStopWords
# from nltk.tag import StanfordPOSTagger


import scipy.spatial.distance as ssd

# load stop words


# normalize text
    # replace urls, mentions, hashtags, caracteres especiales, emoticons por espacio vacío


# nltk tokenize, tokenizar elementos
'''Assumes its ok to remove user mentions and hashtags from tweet text (normalize_text), '''
'''since we extracted them already from the json object'''


'''Prepare features, where doc has terms separated by comma'''
# lo utiliza en main
def custom_tokenize_text(text):
    REGEX = re.compile(r",\s*")
    tokens = []
    for tok in REGEX.split(text):
        # if "@" not in tok and "#" not in tok:
        if "@" not in tok:
            # tokens.append(stem(tok.strip().lower()))
            tokens.append(tok.strip().lower())
    return tokens

def spam_tweet(text):
    if 'Jordan Bahrain Morocco Syria Qatar Oman Iraq Egypt United States' in text:
        return True

    if 'Some of you on my facebook are asking if it\'s me' in text:
        return True

    if '@kylieminogue please Kylie Follow Me, please' in text:
        return True

    if 'follow me please' in text:
        return True

    if 'please follow me' in text:
        return True

    return False


'''start main'''
if __name__ == "__main__":


    file_timeordered_tweets = open('json/pan_timeordered.txt')

    # time_window_mins = float(sys.argv[2])

    ########### PARAMETROS

    time_window_mins = 30.0
    n_documentos_maximos = 5
    factor_frecuencia = 0.01
    num_ngrams_in_tweet = 3

    ###########

    # file_timeordered_news = codecs.open(sys.argv[3], 'r', 'utf-8')
    # fout = codecs.open(sys.argv[3], 'w', 'utf-8')

    debug = 0
    stop_words = misStopWords.creaStopWords()

    # read tweets in time order and window them
    tweet_unixtime_old = -1
    # fout.write("time window size in mins: " + str(time_window_mins))
    tid_to_raw_tweet = {}
    window_corpus = []
    tid_to_urls_window_corpus = {}
    tids_window_corpus = []
    dfVocTimeWindows = {}
    t = 0
    ntweets = 0
    tweettotales = 0

    # st = StanfordPOSTagger('spanish-distsim.tagger')
    tweets_cluster = []

    #quitar tweets repetidos
    total_tweets_repetidos=[]
    esRepetido=[]

    # tweets_pos_tagged = []

    ## QUITAR TWEETS DUPLICADOS

    #	fout.write("\n--------------------start time window tweets--------------------\n")
    # efficient line-by-line read of big files
    for line in file_timeordered_tweets:
        # [tweet_unixtime, tweet_gmttime, tweet_id, text, hashtags, users, urls, media_urls, nfollowers, nfriends] = eval(
        #    line)
        contenido = line.split('\\\\\\\\\\\\')
        # datetime.strptime(fecha, '%a %b %d %H:%M:%S %z %Y')
        tweet_gmttime = datetime.strptime(contenido[0][1:], '%a %b %d %H:%M:%S %z %Y') # contenido[0]
        tweet_unixtime = tweet_gmttime.timestamp()
        tweet_id = contenido[1]
        text = contenido[2][:-2]
        media_urls = ""

        users = re.findall("@[^\s]+", text)
        hashtags = re.findall("#[^\s]+", text)

        if spam_tweet(text):
            continue


        tw = re.sub("(?<=^|(?<=[^a-zA-Z0-9-_\.]))(#|@)([A-Za-z]+[A-Za-z0-9]+)", ' ',
                        line.split('\\\\\\\\\\\\')[2][:-2])
        # quita hashtags y mentions
        tw = li.limpiarTextoTweet(tw, stop_words)
        if tw not in total_tweets_repetidos:
            total_tweets_repetidos.append(tw)
            # esRepetido.append(0)
        else:
            # esRepetido.append(1)
            continue


        if tweet_unixtime_old == -1:
            tweet_unixtime_old = tweet_unixtime

        # #while this condition holds we are within the given size time window
        if (tweet_unixtime - tweet_unixtime_old) < time_window_mins * 60:
            ntweets += 1

            # normalize text and tokenize
            features = li.limpiarTextoTweet(text, stop_words)
            tweet_bag = ""
            try:
                #for user in set(users):
                #    tweet_bag += user.lower() + " , "
                #for tag in set(hashtags):
                #    if tag.lower() not in stop_words:
                #        tweet_bag += tag.lower() + " , "

                # el codigo de arriba solo agrega menciones y hashtags porque el tokenizer las quitaba y en hashtags valida que no sean stop words
                for feature in features:
                    tweet_bag += feature + " , "
            except:
                # print "tweet_bag error!", tweet_bag, len(tweet_bag.split(","))
                pass

            # print tweet_bag.decode('utf-8')

            # la ultima condicion se ve extraña ¿que hace?
            if len(users) < 3 and len(hashtags) < 3 and len(features) > 3: # and len(tweet_bag.split(",")) > 4 and not str(features).upper() == str(features):
                tweet_bag = tweet_bag[:-1] # quita la ultima comma
                # fout.write(tweet_bag + "\n\n")
                window_corpus.append(tweet_bag)
                tids_window_corpus.append(tweet_id)
                tid_to_urls_window_corpus[tweet_id] = media_urls
                tid_to_raw_tweet[tweet_id] = text

                text = text.lower()
                text = re.sub("(https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]", ' ',
                              text)  # url_token
                text = li.quitarAcentos(text)
                text = li.quitarEmoticons(text)

                ## print(text)
                #sen = li.separaTokensMantienePuntuacion(text)
                #if (len(sen) > 0):
                #    tweets_cluster.append(sen)
                tweets_cluster.append(text)
                ## print(tokens)


                # print urls_window_corpus
        else:
            dtime = datetime.fromtimestamp(tweet_unixtime_old).strftime("%d-%m-%Y %H:%M")
            print("\nWindow Starts GMT Time:", dtime, "\n")
            tweet_unixtime_old = tweet_unixtime

            # increase window counter
            t += 1

            # first only cluster tweets


            max_freq = max(int(len(window_corpus) * factor_frecuencia), n_documentos_maximos)


            vectorizer = CountVectorizer(tokenizer=li.limpiarTextoTweet, binary=True,
                                         min_df=max_freq, ngram_range=(1, 3)) # min_df estaba en 10, habria que hacerlo dinamico

            # en los unigramas dejar solo los hashtags y menciones

            try:
                X = vectorizer.fit_transform(window_corpus)

            except:
                print("No se encontraron suficientes documentos para definir un tema")
                continue



            map_index_after_cleaning = {}
            Xclean = np.zeros((1, X.shape[1]))
            for i in range(0, X.shape[0]):
                # keep sample with size at least 5
                # estaba en 4 ### PARAMETRO A REVISAR, PIDE QUE EN UN DOCUMENTO APAREZCAN MINIMO 5 NGRAMAS
                if X[i].sum() >= num_ngrams_in_tweet:
                    Xclean = np.vstack([Xclean, X[i].toarray()])
                    map_index_after_cleaning[Xclean.shape[0] - 2] = i

            Xclean = Xclean[1:, ]

            # NO HAY NGRAMAS REPETIDOS EN SUFICIENTES DOCUMENTOS
            if(len(Xclean)==0 or Xclean.shape[0]<2):
                print("##########################    Otra iteracion, no hay suficientes n-gramas en documentos")
                # se requiere que el ngrama (bigrama o trigrama) aparezca al menos en 7 tweets sin razon alguna, habria que revisar ese numero, antes estaba en 10
                continue

            print("total tweets in window:", ntweets)

            print("X.shape:", X.shape)
            print("Xclean.shape:", Xclean.shape)

            X = Xclean
            Xdense = np.matrix(X).astype('float')
            X_scaled = preprocessing.scale(Xdense)
            X_normalized = preprocessing.normalize(X_scaled, norm='l2')

            vocX = vectorizer.get_feature_names()

            boost_entity = {}

            # print("Indentificando sustantivos con StanfordPOSTagger")
            # for ngram in vocX:
            #     ngramas = ngram.split(sep=' ')
            #     for tweet in tweets_cluster: # tal vez se pueda filtra un poco mas con map_index_after_cleaning
            #
            #         # si los ngramas se encuentran en el tweet,
            #         if len([x for x in ngramas if x in tweet]) == len(ngramas):
            #
            #
            #             x = [(x, tweet.index(x)) for x in ngramas]
            #             sorted_x = [y[0] for y in sorted(x, key=lambda z: z[1])]
            #
            #             # si mantienen el orden
            #             if sorted_x == ngramas: # if ngram in tweet:
            #                 tokens = st.tag(tweet) # REMOVES UNDERSCORES FROM TOKENS
            #
            #                 for term in ngramas:
            #                     # si el ngrama es un sustantivo, boost entity
            #                     if term in [x[0] for x in tokens if x[1].startswith('n')]:
            #                         if ngram.strip() in boost_entity.keys():
            #                             boost_entity[ngram.strip()] += 2.5
            #                         else:
            #                             boost_entity[ngram.strip()] = 2.5
            #                     else:
            #                         if ngram.strip() in boost_entity.keys():
            #                             boost_entity[ngram.strip()] += 1.0
            #                         else:
            #                             boost_entity[ngram.strip()] = 1.0
            #                 break

            print("boosted entities")
            print(boost_entity)


            dfX = X.sum(axis=0) # suma por columna de ngramas
            ## concatenar valores de ngramas en ventanas


            dfVoc = {}
            wdfVoc = {}
            boosted_wdfVoc = {}

            keys = vocX
            vals = dfX
            for k, v in zip(keys, vals):
                # condicion y suma agregado al codigo original,
                if k in dfVoc.keys():
                    dfVoc[k] += v
                else:
                    dfVoc[k] = v

            for k in dfVoc:
                try:
                    dfVocTimeWindows[k] += dfVoc[k]
                    avgdfVoc = (dfVocTimeWindows[k] - dfVoc[k]) / (t - 1)

                except:
                    dfVocTimeWindows[k] = dfVoc[k]
                    avgdfVoc = 0

                # dar mas peso a los nuevos ngramas, y menos peso si es que ya aparecieron en ventanas anteriores
                wdfVoc[k] = (dfVoc[k] + 1) / (np.log(avgdfVoc + 1) + 1)
                try:
                    boosted_wdfVoc[k] = wdfVoc[k] * boost_entity[k]
                except:
                    boosted_wdfVoc[k] = wdfVoc[k]

            print("sorted wdfVoc*boost_entity:")
            print(sorted(((v, k) for k, v in boosted_wdfVoc.items()), reverse=True))

            # distMatrix = pairwise_distances(X_normalized, metric='cosine') # revisamos y coseno no estaba funcionando como deberia

            distMatrix = pairwise_distances(X_normalized, metric='euclidean')

            # convert the redundant n*n square matrix form into a condensed nC2 array
            # sch.distance.pdist(X)
            # distArray = ssd.squareform(distMatrix)  # distArray[{n choose 2}-{n-i choose 2} + (j-i-1)] is the distance between points i and j

            # cluster tweets
            print("fastcluster, average, euclidean")
            L = fastcluster.linkage(distMatrix, method='average')

            dt = 0.5
            print("hclust cut threshold:", dt)

            indL = sch.fcluster(L, dt * distMatrix.max(), 'distance')

            idx_clusts = sorted([(l, k) for k, l in enumerate(indL)], key=lambda x: x[0])

            n_c = 0
            for x in idx_clusts:
                if n_c != x[0]:
                    print("Tweets Cluster {0}".format(x[0]))
                    n_c = x[0]
                print(tweets_cluster[map_index_after_cleaning.get(x[1])])
            # sch.dendrogram(L)

            freqTwCl = Counter(indL)
            print("n_clusters:", len(freqTwCl))
            print(freqTwCl)




            #obtención del (los) ngrama(s) más repetido(s) en cada cluster
            inv_map = {v: k for k, v in vectorizer.vocabulary_.items()}
            main_ngram_in_cluster=[-1]*len(freqTwCl)
            for clust in range(len(freqTwCl)):
                num_ngram = [0] * X.shape[1]
                cont=0
                for tweet in range(X.shape[0]):
                    if indL[tweet] == clust + 1:
                        cont+=1
                        for i in range(X.shape[1]):
                            num_ngram[i]+=X[tweet][i]
                print(num_ngram) #muestra las repeticiones de todos los ngramas por cada cluster
                maximos = (np.argwhere(num_ngram == np.amax(num_ngram))).flatten().tolist()

                main_ngram_in_cluster[clust]= []
                for m in range(len(maximos)):
                    main_ngram_in_cluster[clust].append(inv_map[maximos[m]])
            for i in range(len(main_ngram_in_cluster)):
                print("Ngrama(s) más repetido(s) en el cluster ", (i+1),": ",main_ngram_in_cluster[i])




                    ######################
                    ######################

            ## mostrar contenido de clusters


            npindL = np.array(indL)

            freq_th = max(n_documentos_maximos, int(X.shape[0] * factor_frecuencia))

            cluster_score = {}
            for clfreq in freqTwCl.most_common(50):
                cl = clfreq[0]
                freq = clfreq[1]
                cluster_score[cl] = 0
                if freq >= freq_th:

                    clidx = (npindL == cl).nonzero()[0].tolist()
                    cluster_centroid = X[clidx].sum(axis=0)

                    try:
                        cluster_tweet = vectorizer.inverse_transform(cluster_centroid)
                        for term in np.nditer(cluster_tweet):

                            try:
                                cluster_score[cl] = max(cluster_score[cl], boosted_wdfVoc[str(term).strip()])
                            except:
                                pass
                    except:
                        pass
                    cluster_score[cl] /= freq
                else:
                    continue # antes tenia break





            sorted_clusters = sorted(((v, k) for k, v in cluster_score.items()), reverse=True)
            print("sorted cluster_score:")
            print(sorted_clusters)

            ntopics = 20
            headline_corpus = []
            orig_headline_corpus = []
            headline_to_cluster = {}
            headline_to_tid = {}
            cluster_to_tids = {}
            for score, cl in sorted_clusters[:ntopics]:

                clidx = (npindL == cl).nonzero()[0].tolist()

                first_idx = map_index_after_cleaning[clidx[0]]
                keywords = window_corpus[first_idx]
                orig_headline_corpus.append(keywords)
                headline = ''
                for k in keywords.split(","):
                    if not '@' in k and not '#' in k:
                        headline += k + ","
                headline_corpus.append(headline[:-1])
                headline_to_cluster[headline[:-1]] = cl
                headline_to_tid[headline[:-1]] = tids_window_corpus[first_idx]
                tids = []
                for i in clidx:
                    idx = map_index_after_cleaning[i]
                    tids.append(tids_window_corpus[idx])
                cluster_to_tids[cl] = tids

            ## cluster headlines to avoid topic repetition
            headline_vectorizer = CountVectorizer(tokenizer=li.limpiarTextoTweet, binary=True, min_df=1,
                                                  ngram_range=(1, 1))

            # headline_vectorizer = TfidfVectorizer(tokenizer=custom_tokenize_text, min_df=1, ngram_range=(1,1))

            H = headline_vectorizer.fit_transform(headline_corpus)
            print("H.shape:", H.shape)
            vocH = headline_vectorizer.get_feature_names()


            Hdense = np.matrix(H.todense()).astype('float')

            distH = pairwise_distances(Hdense, metric='cosine')

            HL = fastcluster.linkage(distH, method='average')

            # NO HAY SUFICIENTES CLUSTERS, FALTA OBTENER HEADLINES DE UN SOLO CLUSTER
            if(len(distH)<2):
                print("Otra iteracion, no hay suficientes clusters")
                # aqui revisar porque solo se habla de un tema, pero no se muestra
                continue

            dtH = 1.0
            indHL = sch.fcluster(HL, dtH * distH.max(), 'distance')
            freqHCl = Counter(indHL)
            print("hclust cut threshold:", dtH)
            print("n_clusters:", len(freqHCl))
            print(freqHCl)

            npindHL = np.array(indHL)
            hcluster_score = {}
            for hclfreq in freqHCl.most_common(ntopics):
                hcl = hclfreq[0]
                hfreq = hclfreq[1]
                hcluster_score[hcl] = 0
                hclidx = (npindHL == hcl).nonzero()[0].tolist()
                for i in hclidx:
                    hcluster_score[hcl] = max(hcluster_score[hcl],
                                              cluster_score[headline_to_cluster[headline_corpus[i]]])
            sorted_hclusters = sorted(((v, k) for k, v in hcluster_score.items()), reverse=True)
            print("sorted hcluster_score:")
            print(sorted_hclusters)

            for hscore, hcl in sorted_hclusters[:10]:
                hclidx = (npindHL == hcl).nonzero()[0].tolist()
                clean_headline = ''
                raw_headline = ''
                keywords = ''
                tids_set = set()
                tids_list = []
                urls_list = []
                selected_raw_tweets_set = set()
                tids_cluster = []
                for i in hclidx:
                    clean_headline += headline_corpus[i].replace(",", " ") + "//"
                    keywords += orig_headline_corpus[i].lower() + ","
                    tid = headline_to_tid[headline_corpus[i]]
                    tids_set.add(tid)
                    raw_tweet = tid_to_raw_tweet[tid].replace("\n", ' ').replace("\t", ' ')
                    raw_tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pic\.twitter\.com/[^\s]+))', '', raw_tweet)
                    selected_raw_tweets_set.add(raw_tweet.strip())
                    tids_list.append(tid)
                    if tid_to_urls_window_corpus[tid]:
                        urls_list.append(tid_to_urls_window_corpus[tid])
                    for id in cluster_to_tids[headline_to_cluster[headline_corpus[i]]]:
                        tids_cluster.append(id)

                raw_headline = tid_to_raw_tweet[headline_to_tid[headline_corpus[hclidx[0]]]]
                raw_headline = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pic\.twitter\.com/[^\s]+))', '', raw_headline)
                raw_headline = raw_headline.replace("\n", ' ').replace("\t", ' ')
                keywords_list = str(sorted(list(set(keywords[:-1].split(",")))))[1:-1].replace('\'', '')

                # Select tweets with media urls
                # If need code to be more efficient, reduce the urls_list to size 1.
                for tid in tids_cluster:
                    if len(urls_list) < 1 and tid_to_urls_window_corpus[tid] and tid not in tids_set:
                        raw_tweet = tid_to_raw_tweet[tid].replace("\n", ' ').replace("\t",' ')
                        raw_tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pic\.twitter\.com/[^\s]+))', '',
                                           raw_tweet)
                        # fout.write("\ncluster tweet: " + raw_tweet)
                        if raw_tweet.strip() not in selected_raw_tweets_set:
                            tids_list.append(tid)
                            urls_list.append(tid_to_urls_window_corpus[tid])
                            selected_raw_tweets_set.add(raw_tweet.strip())

                try:
                    print("\n", clean_headline, raw_headline, tids_list)
                except:
                    pass

                urls_set = set()
                for url_list in urls_list:
                    for url in url_list:
                        urls_set.add(url)



                # fout.write(
                #    "\n" + str(dtime) + "\t" + raw_headline.decode('utf8', 'ignore') + "\t" + keywords_list.decode(
                #        'utf8', 'ignore') + "\t" + str(tids_list)[1:-1] + "\t" + str(list(urls_set))[1:-1][
                #                                                                 2:-1].replace('\'', '').replace(
                #        'uhttp', 'http'))


            # sys.exit()
            window_corpus = []
            tids_window_corpus = []
            tid_to_urls_window_corpus = {}
            tid_to_raw_tweet = {}
            ntweets = 0

            tweets_cluster = []

            if t == 4:
                dfVocTimeWindows = {}
                t = 0

                # fout.write("\n--------------------start time window tweets--------------------\n")
                # fout.write(line)

    file_timeordered_tweets.close()
    # fout.close()


## pensada como opcion en la matriz de distancias
def LevenshteinDistance(str1, str2):
  d=dict()
  for i in range(len(str1)+1):
     d[i]=dict()
     d[i][0]=i
  for i in range(len(str2)+1):
     d[0][i] = i
  for i in range(1, len(str1)+1):
     for j in range(1, len(str2)+1):
        d[i][j] = min(d[i][j-1]+1, d[i-1][j]+1, d[i-1][j-1]+(not str1[i-1] == str2[j-1]))
  return d[len(str1)][len(str2)]
