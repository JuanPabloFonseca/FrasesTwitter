#!/usr/bin/python3.5

from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

import time
import operator

import ObtenerTweets
import LimpiarTweets

def print_top_words(model, feature_names, n_top_words):
    for topic_id, topic in enumerate(model.components_):
        print('\nTopic Nr.%d:' % int(topic_id + 1))
        print(''.join([feature_names[i] + ' ' + str(round(topic[i], 2))
              +' | ' for i in topic.argsort()[:-n_top_words - 1:-1]]))


# I discovered the lambda-formular connected to sklearns components_ in "Latent Dirichlet Allocation" Blei/Ng/Jordan (p.1007).
# There is no normalisation either, so I think the sklearn implementation is correct.
# In my case it was quiet interesting to get very high values for one topic.
# This also fitted well to the common meaning of those tokens of this topic.
# I think the differences in value height go together with the dirichlet distribution, so higher values mean that topics occure more often in the corpus.
# If I'm right, we actually lose information by normalising.
def NMF_sklearn(datos, n_topics, n_top_words):
    docs = []
    start = time.time()
    for t in datos:
        docs.append(', '.join(str(x) for x in t))
    # Use tf-idf features for NMF.
    print("Extracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(min_df=2, lowercase=False, encoding='utf8mb4')

    # t0 = time()
    tfidf = tfidf_vectorizer.fit_transform(docs)

    # Fit the NMF model
    print("Fitting the NMF model with tf-idf features,"
          "n_samples=%d and n_features=%d..."
          % (len(docs), len(docs)))
    # t0 = time()
    nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)
    # exit()
    # print("done in %0.3fs." % (time() - t0))
    end = time.time()

    print("\nTopics in NMF model:")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, n_top_words)

    print("tiempo NMF: ", end - start)

def LDA_sklearn(datos, n_topics, n_top_words, iteraciones):
    # Use tf (raw term count) features for LDA.
    # el numero de veces que cada termino ocurre en cada documento y sumarlos;
    print("Extracting tf features for LDA...")
    start = time.time()
    docs = []
    for t in datos:
        docs.append(', '.join(str(x) for x in t))

    # max_df ignora terminos que tengan arriba de #de documentos si entero, si flotante porcentaje de documentos
    # min_df menos de # de documentos
    # crea una matriz de #DocumentosXDiferentesPalabras con el term frequency en cada celda
    tf_vectorizer = CountVectorizer(min_df=2, lowercase=False, encoding='utf8mb4')  # max_features=n_features, max_df=0.95,

    # fit_transform(raw_documents[, y])    Learn the vocabulary dictionary and return term - document matrix.
    tf = tf_vectorizer.fit_transform(docs)

    iter = iteraciones
    print("Fitting LDA models with tf features, n_samples=%d and n_features=%d iter=%d..." % (tf.shape[0], tf.shape[1], iter))

    # learning method: if the data size is large, the online update will be much faster than the batch update
    # learning offser : A (positive) parameter that downweights early iterations in online learning
    # random state: Pseudo-random number generator seed control.
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=iter, learning_method='online', learning_offset=10.0, random_state=0)

    # Learn model for the data X with variational Bayes method.
    lda.fit(tf)
    end = time.time()

    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)

    print("tiempo LDA sklearn: ", end - start)

def LDA_gensim(datos, n_topics, passes):
    # turn our tokenized documents into a id <-> term dictionary
    start = time.time()
    print("Fitting LDA gensim ")
    dictionary = corpora.Dictionary(datos)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in datos]

    # generate LDA model
    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=n_topics, id2word=dictionary, passes=passes)

    end = time.time()
    # Prints the topics.
    for top in ldamodel.print_topics():
        print(top)
    print

    print("tiempo LDA_gensim: ", end - start)

def contarPalabras(datos):
    # palabras = []
    contar = {}
    start = time.time()
    for row in datos:
        for palabra in row:
            if palabra in contar:
                contar[palabra]+=1
            else:
                contar[palabra]=1
    # ordenar por valor, descendente
    sorted_x = sorted(contar.items(), key=operator.itemgetter(1), reverse=True)
    end = time.time()

    print('Palabras principales por conteo de palabras: ')
    for i in range(0,10):
        print(i, sorted_x[i])

    print("tiempo Contar: ", end - start)

# Sinnia y TASS
# 1) obtener LDA sobre entrenamiento
# 2) Utilizando las combinaciones lineales, clasificar con el valor maximo tweets de validacion
# 3) obtener precision y recall sobre test

# TASS vs Resultados de TASS, para indicar porque LDA

def demo():
    datos = ObtenerTweets.obtenerTweetsArchivo()
    datos = LimpiarTweets.limpiarTexto(datos)

    #NMF_sklearn(datos, 3, 5)

    LDA_gensim(datos, 3, 20)
    LDA_sklearn(datos, 3, 5, 20)


if __name__ == '__main__':
    demo()
