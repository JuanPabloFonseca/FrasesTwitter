import LDA_cluster
import ObtenerTweets
import LimpiarTweets

import numpy
import pandas as pd
import statsmodels
import pylab


def principal():
    #s= ["política,economía","otros,entretenimiento","política","música,entretenimiento","música","economía,política"]
    #s=pd.Series(s)
    #print(s.str.get_dummies(sep=','))

    topicos=obtenerTopicosTweets() #obtiene tópicos
    topicos=pd.Series(topicos)
    tablaTopicos=topicos.str.get_dummies(sep=',')#genera la matriz binaria
    print(tablaTopicos)
    #print(tablaTopicos.std())
    #print(tablaTopicos["econom?a"][3])



def obtenerTopicosTweets():
    data = []
    with open('tass/tass_2015/tweetsTopic.txt') as f:
        for line in f:
            data.append(line.split('\\',1)[0])

    return data



if __name__ == '__main__':
    principal()