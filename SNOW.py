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


import LimpiarTweets as li
import misStopWords
from nltk.tag import StanfordPOSTagger

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
    time_window_mins = 15.0
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

    st = StanfordPOSTagger('spanish-distsim.tagger')
    tweets_pos_tagged = []

    #	fout.write("\n--------------------start time window tweets--------------------\n")
    # efficient line-by-line read of big files
    for line in file_timeordered_tweets:
        # [tweet_unixtime, tweet_gmttime, tweet_id, text, hashtags, users, urls, media_urls, nfollowers, nfriends] = eval(
        #    line)

        contenido = line.split('\\\\\\')
        # datetime.strptime(fecha, '%a %b %d %H:%M:%S %z %Y')
        tweet_gmttime = datetime.strptime(contenido[0][1:], '%a %b %d %H:%M:%S %z %Y') # contenido[0]
        tweet_unixtime = tweet_gmttime.timestamp()
        tweet_id = contenido[1]
        text = contenido[2]
        media_urls = ""

        users = re.findall("@[^\s]+", text)
        hashtags = re.findall("#[^\s]+", text)

        if spam_tweet(text):
            continue

        if tweet_unixtime_old == -1:
            tweet_unixtime_old = tweet_unixtime

        # #while this condition holds we are within the given size time window
        if (tweet_unixtime - tweet_unixtime_old) < time_window_mins * 60:
            ntweets += 1
            print(ntweets)

            # normalize text and tokenize
            features = li.limpiarTextoTweet(text, stop_words)
            tweet_bag = ""
            try:
                for user in set(users):
                    tweet_bag += user.lower() + ","
                for tag in set(hashtags):
                    if tag.lower() not in stop_words:
                        tweet_bag += tag.lower() + ","
                for feature in features:
                    tweet_bag += feature + ","
            except:
                # print "tweet_bag error!", tweet_bag, len(tweet_bag.split(","))
                pass

            # print tweet_bag.decode('utf-8')

            # la ultima condicion se ve extraña ¿que hace?
            if len(users) < 3 and len(hashtags) < 3 and len(features) > 3 and len(tweet_bag.split(",")) > 4 and not str(
                    features).upper() == str(features):
                tweet_bag = tweet_bag[:-1]
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

                print(text)
                sen = li.separaTokensMantienePuntuacion(text)
                if (len(sen) > 0):
                    tokens = st.tag(sen)
                    tweets_pos_tagged.append(tokens)
                    print(tokens)


                # print urls_window_corpus
        else:
            dtime = datetime.fromtimestamp(tweet_unixtime_old).strftime("%d-%m-%Y %H:%M")
            print("\nWindow Starts GMT Time:", dtime, "\n")
            tweet_unixtime_old = tweet_unixtime

            # increase window counter
            t += 1

            # first only cluster tweets

            vectorizer = CountVectorizer(tokenizer=custom_tokenize_text, binary=True,
                                         min_df=max(int(len(window_corpus) * 0.0025), 7), ngram_range=(2, 3)) # min_df estaba en 10, habria que hacerlo dinamico

            try:
                X = vectorizer.fit_transform(window_corpus)
            except:
                print("No se encontraron suficientes documentos para definir un tema")
                continue

            map_index_after_cleaning = {}
            Xclean = np.zeros((1, X.shape[1]))
            for i in range(0, X.shape[0]):
                # keep sample with size at least 5
                if X[i].sum() > 4: # estaba en 4 ### PARAMETRO
                    Xclean = np.vstack([Xclean, X[i].toarray()])
                    map_index_after_cleaning[Xclean.shape[0] - 2] = i

            Xclean = Xclean[1:, ]

            # NO HAY NGRAMAS REPETIDOS EN SUFICIENTES DOCUMENTOS
            if(len(Xclean)==0):
                print("Otra iteracion, no hay suficientes n-gramas en documentos")
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


            # si el ngrama tiene un noun boost
            for ngram in vocX:
                for term in ngram.split(sep=' '):
                    for pos_tweet in tweets_pos_tagged:
                        for pos_term in pos_tweet:
                            if term == pos_term[0] and pos_term[1].startswith('n'):
                                boost_entity[ngram.strip()] = 2.5
                            else:
                                boost_entity[ngram.strip()] = 1.0

            # limpiar cluster para siguiente iteracion
            tweets_pos_tagged = []

            ### CODIGO ORIGINAL _ TENIA POS TAGGING AFTER CLEANING
            ## pos_tokens = CMUTweetTagger.runtagger_parse([term.upper() for term in vocX])
            ## print("inicia POS tagger {0}".format(vocX))
            ## st = StanfordPOSTagger('spanish-distsim.tagger')
            ## pos_tokens = st.tag([term.upper().split(sep=' ') for term in vocX])

            # CMU (^) ^ <- noun.
            # Stanford NN noun
            #for l in pos_tokens:
            #    term = ''
            #    #for gr in range(0, len(l)):
            #    term += l[0].lower() #l[gr][0].lower() + " "
            #    if str(l[1]).startswith('n'):  # para CMU if "^" in str(l):
            #        boost_entity[term.strip()] = 2.5
            #    else:
            #        boost_entity[term.strip()] = 1.0

            dfX = X.sum(axis=0)

            dfVoc = {}
            wdfVoc = {}
            boosted_wdfVoc = {}

            keys = vocX

            vals = dfX
            repetidas = 0
            for k, v in zip(keys, vals):
                # condicion y suma agregado al codigo original,
                if k in dfVoc.keys():
                    dfVoc[k] += v
                    repetidas += 1
                else:
                    dfVoc[k] = v
            print(repetidas)
            for k in dfVoc:
                try:
                    dfVocTimeWindows[k] += dfVoc[k]
                    avgdfVoc = (dfVocTimeWindows[k] - dfVoc[k]) / (t - 1)

                except:
                    dfVocTimeWindows[k] = dfVoc[k]
                    avgdfVoc = 0

                wdfVoc[k] = (dfVoc[k] + 1) / (np.log(avgdfVoc + 1) + 1)
                try:
                    boosted_wdfVoc[k] = wdfVoc[k] * boost_entity[k]
                except:
                    boosted_wdfVoc[k] = wdfVoc[k]
            print("sorted wdfVoc*boost_entity:")
            print(sorted(((v, k) for k, v in boosted_wdfVoc.items()), reverse=True))

            distMatrix = pairwise_distances(X_normalized, metric='cosine')

            # cluster tweets
            print("fastcluster, average, cosine")
            L = fastcluster.linkage(distMatrix, method='average')

            dt = 0.5
            print("hclust cut threshold:", dt)

            indL = sch.fcluster(L, dt * distMatrix.max(), 'distance')

            freqTwCl = Counter(indL)
            print("n_clusters:", len(freqTwCl))
            print(freqTwCl)

            npindL = np.array(indL)

            freq_th = max(10, int(X.shape[0] * 0.0025))
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
                    break

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
            headline_vectorizer = CountVectorizer(tokenizer=custom_tokenize_text, binary=True, min_df=1,
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
                    print("\n", clean_headline)
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
            if t == 4:
                dfVocTimeWindows = {}
                t = 0

                # fout.write("\n--------------------start time window tweets--------------------\n")
                # fout.write(line)

    file_timeordered_tweets.close()
# fout.close()