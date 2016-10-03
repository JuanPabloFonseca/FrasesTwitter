#!/usr/bin/python3.5

from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import re
import time

from .misStopWords import creaStopWords

### Regresa los tweets tokenizados
### convierte a minusculas, quita urls, quita acentos, quita stop words
### por ejemplo
### [u'j0sedxx', u'comiendo', u'pan', u'dulc', u'momento', u'v']
def limpiarTexto(datos):
    patterns = r'[#@][^\s]+|\d+\.\d*|\w+'
    tokenizer = RegexpTokenizer(patterns)
    # create Spanish stop words list
    en_stop = creaStopWords()
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

        raw = re.sub("(https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]", ' ', raw)  # url_token



        # quitar acentos

        raw = re.sub("á", 'a', raw)

        raw = re.sub("é", 'e', raw)

        raw = re.sub("í", 'i', raw)

        raw = re.sub("ó",'o', raw)

        raw = re.sub("ú",'u', raw)

        # tokenizar mention
        # p = re.compile("@[A-Za-z0-9_]+")
        # raw = p.sub('mention', raw)

        tokens = tokenizer.tokenize(raw)

        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop and len(i) > 1]
        # Al inicio se mantuvieron porque comento SINNIA que no los quitaban
        # Pero se mantuvieron por obtener resultados como el siguiente
        # ('tiempo LDA: ', 42.264963150024414)
        # (0, u'0.057*url_token + 0.056*corona + 0.036*en + 0.021*el + 0.021*la + 0.017*de + 0.016*del + 0.016*a + 0.015*por + 0.014*y')
        # (1, u'0.061*url_token + 0.056*corona + 0.044*de + 0.035*la + 0.029*el + 0.022*en + 0.017*a + 0.015*su + 0.014*del + 0.009*por')
        # (2, u'0.063*corona + 0.048*la + 0.041*de + 0.025*que + 0.022*y + 0.021*url_token + 0.020*a + 0.019*una + 0.013*en + 0.012*no')

        # stem tokens NO APLICA POR NO ESTAR EN BASELINE
        #stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]


        #PROCESAR EMOJIS

        # add tokens to list
        texts.append(stopped_tokens)
    end = time.time()
    print("tiempo Obtener-Limpieza-Tokenizar: ", end - start)
    return texts

def limpiarTextoTweet(tweet, stop_words=[]):

    patterns = r'[#@][^\s]+|\d+\.\d*|\w+'
    tokenizer = RegexpTokenizer(patterns)

    # create Spanish stop words list
    en_stop = stop_words
    # Create p_stemmer of class PorterStemmer
    # p_stemmer = PorterStemmer()

    # list for tokenized documents in loop
    texts = []
    # loop through document list
    # for i in doc_set:


    # clean and tokenize document string
    # a minusculas
    raw = tweet.lower()

    # TOKENIZAR URLS, convertir URL en la palabra url_token

    raw = re.sub("(https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]", ' ', raw)  # url_token

    raw = quitarAcentos(raw)

    raw = quitarEmoticons(raw)

    # tokenizar mention
    # p = re.compile("@[A-Za-z0-9_]+")
    # raw = p.sub('mention', raw)

    tokens = tokenizer.tokenize(raw) # tokeniza una mencion @algo como algo

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


    #PROCESAR EMOJIS

    # add tokens to list


    return stopped_tokens


def separaTokensMantienePuntuacion(texto):
    palabras = []
    palabra = ''
    i=0
    for c in texto:

        if c in " \n|": # estos separadores no se mantienen en el resultado
            if len(palabra)>0:
                palabras.append(palabra)
                palabra = ''
        elif i == len(texto)-1: # si es el final de la oracion
            palabra += c
            palabras.append(palabra)
            palabra = ''
        elif c in ".,¿¡!?:;*\"\'()": # estos separadores si se mantienen
            palabras.append(palabra)
            palabra = ''
            palabras.append(c)
        else:
            palabra += c

        i += 1
    return palabras



def quitarAcentos(text):
    text = re.sub(r"à|á", 'a', text)
    text = re.sub(r"è|é", 'e', text)
    text = re.sub(r"ì|í", 'i', text)
    text = re.sub(r"ò|ó", 'o', text)
    text = re.sub(r"ù|ú", 'u', text)
    return text

def quitarEmoticons(text):

    bien = ''
    for x in text:
        if ord(x)<=256:
            bien += x

    bien = re.sub(r"&gt;|&lt;(3+)?|\\n|(:|;)(\)|\(|D|\*)|&amp;|\*-\*|(x|X)(d|D)", ' ', bien)

    return bien

def steamWord(palabra):
    p_stemmer = PorterStemmer()
    return p_stemmer.stem(palabra)