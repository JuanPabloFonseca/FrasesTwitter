import gensim
import numpy as np
model = gensim.models.Word2Vec.load_word2vec_format("SBW-vectors-300-min5.bin", fvocab=None, binary=True, encoding='utf8', unicode_errors='ignore')

def calcW2V(corpus):
    vec_final = []
    for tuit in corpus:
        words = tuit.split(",")
        vec_tuit = []
        for word in words:
            try:
                if word.strip(" ") != "":
                    vec_tuit.append(model[word.strip(" ")])
            except KeyError:
                continue
        a = np.array(vec_tuit)
        vec_final.append(np.mean(a, axis=0))  # average all vectors into one
    return vec_final
        # result have a matrix size |tweets| x 200