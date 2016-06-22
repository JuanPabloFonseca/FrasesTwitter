#!/usr/bin/python
import MySQLdb

from itertools import chain
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models

import gensim
import re
import time
import unicodedata
import operator


import json
import pandas as pd
from sqlalchemy import create_engine

def obtenerTweetsArchivo():
    data = []
    with open('/home/eduardomartinez/Documents/Sinnia/json/todos.json') as f:
        for line in f:
            data.append(json.loads(line))

    datos = []
    for row in data:
        datos.append(row['text'])
    return datos

def cargarTweetsEnDB():
    data = []
    with open('/home/eduardomartinez/Documents/Sinnia/json/corona.json') as f:
        for line in f:
            data.append(json.loads(line))
    datos = []
    for tweet in data:
        text = tweet['text']  # encode unicode_escape
        user_id = int(tweet['user']['id'])
        id = int(tweet['id'])
        datos.append({'status_text': text})
    engine = create_engine('mysql://root:root@localhost:3306/sinnia') # ?charset=utf8mb4
    pd.DataFrame(datos).to_sql('corona_csv', engine, if_exists='append') #replace genera la tabla, append utiliza la misma tabla


def obtenerDatosBD():
    db = MySQLdb.connect(host="localhost", user="root", passwd="root", db="sinnia", charset='utf8')
    # name of the data base # you must create a Cursor object. It will let you execute all the queries you need
    cur = db.cursor()
    # Use all the SQL you like
    cur.execute("SELECT status_text FROM corona_csv")
    datos = []
    for row in cur.fetchall():
        datos.append(row[0])
    db.close()
    return datos

### Regresa los tweets tokenizados por ejemplo
### [u'j0sedxx', u'comiendo', u'pan', u'dulc', u'momento', u'v']
def limpiarTexto(datos):
    tokenizer = RegexpTokenizer(r'\w+')
    # create Spanish stop words list
    en_stop = get_stop_words('es')
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
    # list for tokenized documents in loop
    texts = []
    # loop through document list
    # for i in doc_set:
    start = time.time()
    for row in datos:
        # clean and tokenize document string
        # a minusculas
        i = row.lower()

        # TOKENIZAR URLS, convertir URL en la palabra url_token
        p = re.compile("(https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]")
        raw = p.sub(' ', i)  # url_token

        # tokenizar mention
        # p = re.compile("@[A-Za-z0-9_]+")
        # raw = p.sub('mention', raw)

        tokens = tokenizer.tokenize(raw)

        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # Al inicio se mantuvieron porque comento SINNIA que no los quitaban
        # Pero se mantuvieron por obtener resultados como el siguiente
        # ('tiempo LDA: ', 42.264963150024414)
        # (0, u'0.057*url_token + 0.056*corona + 0.036*en + 0.021*el + 0.021*la + 0.017*de + 0.016*del + 0.016*a + 0.015*por + 0.014*y')
        # (1, u'0.061*url_token + 0.056*corona + 0.044*de + 0.035*la + 0.029*el + 0.022*en + 0.017*a + 0.015*su + 0.014*del + 0.009*por')
        # (2, u'0.063*corona + 0.048*la + 0.041*de + 0.025*que + 0.022*y + 0.021*url_token + 0.020*a + 0.019*una + 0.013*en + 0.012*no')

        # stem tokens NO APLICA POR NO ESTAR EN BASELINE
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

        # sin steeming
        # ('tiempo LDA: ', 26.731714963912964)
        # (0, u'0.092*url_token + 0.085*corona + 0.012*wilder + 0.008*peso + 0.008*szpilka + 0.007*pesado + 0.007*rey + 0.007*dakar + 0.007*vencer + 0.007*peterhansel')
        # (1, u'0.089*corona + 0.044*url_token + 0.006*suma + 0.006*dakar + 0.006*peterhansel + 0.006*duod\xe9cima + 0.006*si + 0.005*q + 0.004*quiero + 0.004*rt')
        # (2, u'0.087*corona + 0.051*url_token + 0.012*miss + 0.011*copa + 0.008*mx + 0.007*1 + 0.005*bien + 0.004*0 + 0.004*fin + 0.004*chivas')

        # add tokens to list
        texts.append(stemmed_tokens)
    end = time.time()
    print("tiempo Obtener-Limpieza-Tokenizar: ", end - start)
    return texts

def LDA(datos):
    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(datos)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in datos]

    # generate LDA model
    start = time.time()
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=20)
    end = time.time()
    print("tiempo LDA: ", end - start)

    # Prints the topics.
    for top in ldamodel.print_topics():
        print(top)
    print

    # Assigns the topics to the documents in corpus
    lda_corpus = ldamodel[corpus]

    # Find the threshold, let's set the threshold to be 1/#clusters,
    # To prove that the threshold is sane, we average the sum of all probabilities:
    scores = list(chain(*[[score for topic_id, score in topic] \
                          for topic in [doc for doc in lda_corpus]]))
    threshold = sum(scores) / len(scores)
    print(threshold)
    print

    cluster1 = [j for i, j in zip(lda_corpus, datos) if i[0][1] > i[1][1] and i[0][1] > i[2][1]]
    cluster2 = [j for i, j in zip(lda_corpus, datos) if i[1][1] > i[0][1] and i[1][1] > i[2][1]]
    cluster3 = [j for i, j in zip(lda_corpus, datos) if i[2][1] > i[0][1] and i[2][1] > i[1][1]]

    print(len(cluster1))
    print(len(cluster2))
    print(len(cluster3))

    #print('Muestra de los 10 tweets de cada cluster')
    #for i in range(0, 10):
    #    print(cluster1[i])
    #    print(cluster2[i])
    #    print(cluster3[i])
    #    print('i')



def contarPalabras(datos):
    # palabras = []
    contar = {}
    for row in datos:
        for palabra in row:
            # palabras.append(palabra)
            # The normal form KD (NFKD) will apply the compatibility decomposition,
            # i.e. replace all compatibility characters with their equivalents.
            palabrastr = unicodedata.normalize('NFKD', palabra).encode('ascii', 'ignore')

            #checar si no es mejor unicode_escape

            if contar.has_key(palabrastr):
                contar[palabrastr]+=1
            else:
                contar[palabrastr] = 1
    # ordenar por valor, descendente
    sorted_x = sorted(contar.items(), key=operator.itemgetter(1), reverse=True)
    print('Palabras principales por conteo de palabras: ')
    for i in range(0,10):
        print(i, sorted_x[i])


# en documento
# estudio estadistico mas que estudio formal

# en proyecto python
# Sinnia y TASS
# 1) obtener LDA sobre entrenamiento
# 2) Utilizando las combinaciones lineales, clasificar con el valor maximo tweets de validacion
# 3) obtener precision y recall sobre test

# TASS vs Resultados de TASS, para indicar porque LDA

def demo():
    datos = obtenerTweetsArchivo()
    datos = limpiarTexto(datos)
    LDA(datos)
    contarPalabras(datos)

    #cargarTweetsEnDB()

if __name__ == '__main__':
    demo()