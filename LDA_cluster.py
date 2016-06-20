#!/usr/bin/python
import MySQLdb

from itertools import chain
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import re

db = MySQLdb.connect(host="localhost", user="root", passwd="root", db="sinnia", charset='utf8')
# name of the data base # you must create a Cursor object. It will let
# you execute all the queries you need
cur = db.cursor()
# Use all the SQL you like
cur.execute("SELECT * FROM corona_csv limit 5000")

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('es')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

# list for tokenized documents in loop
texts = []

# loop through document list
# for i in doc_set:
for row in cur.fetchall():
    # clean and tokenize document string
    i = row[10]

    p = re.compile('(blue|white|red)')
    i = p.sub('url_token', i)

    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens NO APLICA POR NO ESTAR EN BASELINE
    # stopped_tokens = [i for i in tokens if not i in en_stop]

    # stem tokens NO APLICA POR NO ESTAR EN BASELINE
    # stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

    # TOKENIZAR URLS

    # add tokens to list
    texts.append(tokens)

db.close()

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=20)


# Prints the topics.
for top in ldamodel.print_topics():
  print top
print

# Assigns the topics to the documents in corpus
lda_corpus = ldamodel[corpus]

# Find the threshold, let's set the threshold to be 1/#clusters,
# To prove that the threshold is sane, we average the sum of all probabilities:
scores = list(chain(*[[score for topic_id,score in topic] \
                      for topic in [doc for doc in lda_corpus]]))
threshold = sum(scores)/len(scores)
print threshold
print

cluster1 = [j for i,j in zip(lda_corpus,texts) if i[0][1] > threshold]
cluster2 = [j for i,j in zip(lda_corpus,texts) if i[1][1] > threshold]
cluster3 = [j for i,j in zip(lda_corpus,texts) if i[2][1] > threshold]

print cluster1
print cluster2
print cluster3

# estudio estadistico mas que estudio formal

# Sinnia y TASS
# 1) obtener LDA sobre entrenamiento
# 2) Utilizando las combinaciones lineales, clasificar con el valor maximo tweets de validacion
# 3) obtener precision y recall sobre test

# TASS vs Resultados de TASS, para indicar porque LDA
