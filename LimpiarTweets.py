#!/usr/bin/python3.5

from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import re
import time

import misStopWords

### Regresa los tweets tokenizados por ejemplo
### [u'j0sedxx', u'comiendo', u'pan', u'dulc', u'momento', u'v']
def limpiarTexto(datos):
    tokenizer = RegexpTokenizer(r'\w+')
    # create Spanish stop words list
    en_stop = misStopWords.creaStopWords()
    # Create p_stemmer of class PorterStemmer
    # p_stemmer = PorterStemmer()

    # list for tokenized documents in loop
    texts = []
    # loop through document list
    # for i in doc_set:
    start = time.time()
    for row in datos:
        # clean and tokenize document string
        # a minusculas
        raw = row.lower()

        # TOKENIZAR URLS, convertir URL en la palabra url_token
        p = re.compile("(https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]")
        raw = p.sub(' ', raw)  # url_token


        # quitar acentos
        p = re.compile("á")
        raw = p.sub('a', raw)
        p = re.compile("é")
        raw = p.sub('e', raw)
        p = re.compile("í")
        raw = p.sub('i', raw)
        p = re.compile("ó")
        raw = p.sub('o', raw)
        p = re.compile("ú")
        raw = p.sub('u', raw)

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
        #stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

        # add tokens to list
        texts.append(stopped_tokens)
    end = time.time()
    print("tiempo Obtener-Limpieza-Tokenizar: ", end - start)
    return texts