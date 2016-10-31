#!/usr/bin/python3.5
import LDA_cluster
import ObtenerTweets
import LimpiarTweets

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import normalized_mutual_info_score

import pandas as pd

import numpy as np

def baseline():
    datosTrain = ObtenerTweets.obtenerTweetsArchivo('train')
    datosTr = LimpiarTweets.limpiarTexto(datosTrain)
    num_topicos = 3

    # numpy.random.RandomState(15485863)
    # numpy.random.seed(15485863)

    #print("sklearn lda")
    #ldask=LDA_cluster.LDA_sklearn(datosTr,num_topicos,5,1)

    #print(str(LDA_cluster.clasTweet(['corona','modelo','basave','gobierno'],ldask,"sklearn"))) prueba de que sí funciona esto

    comportamiento = []
    pruebas = 10
    for i in range(0,pruebas-1):
        print("Iteracion {0}".format(i))
        ldam=LDA_cluster.LDA_gensim(datosTr,num_topicos,1)
        # contarPalabras(datosTr)

        # intentar con cross validation

        datosTest = ObtenerTweets.obtenerTweetsArchivo('test')
        datosTe = LimpiarTweets.limpiarTexto(datosTest)

        ### TRAIN
        # clasificacion_original = LDA_cluster.clasifOriginal("train")
        # clasificacion_modelo = LDA_cluster.clasifica(datosTr, ldam)

        # resultados('TRAIN', clasificacion_original, clasificacion_modelo, datosTrain, num_topicos)

        ### TEST
        clasificacion_original = LDA_cluster.clasifOriginal("test")
        clasificacion_modeloG = LDA_cluster.clasifica(datosTe, ldam,"gensim")
        ####clasificacion_modeloS = LDA_cluster.clasifica(datosTe, ldask, "sklearn")

        pr, rc, fb, su = resultados('TEST gensim ' + str(i),clasificacion_original, clasificacion_modeloG, datosTest, num_topicos)
        ####resultados('TEST sklearn',clasificacion_original, clasificacion_modeloS, datosTest, num_topicos)

        print("Precision {0}, Recall {1}, F1 {2}".format(pr,rc,fb))
        comportamiento.append([pr, rc, fb])

        print("Purity LDA: {0}".format(purity(clasificacion_original,clasificacion_modeloG)))
        print("NMI: {0}".format(normalized_mutual_info_score(clasificacion_original,clasificacion_modeloG)))

    r = np.array(comportamiento)
    prom = r.mean(axis=0)

    ds = r.std(axis=0)

    print("Al ejecutar {0} pruebas, Precision {1}, Recall {2}, F1 {3} (Promedio)".format(pruebas, prom[0], prom[1], prom[2]))
    print("Al ejecutar {0} pruebas, Precision {1}, Recall {2}, F1 {3} (Desviación Estándar)".format(pruebas, ds[0], ds[1], ds[2]))
    LDA_cluster.showPlots()

    # mostrarTweets(num_topicos, datos_originales, clasificacion_original)
    # matriz binaria de clusters
    # entrenar modelo por topico
    # resultado es matriz binaria con el resultado de cada modelo

def resultados(titulo, clasificacion_original, clasificacion_modelo, datos_originales, num_topicos):
    #print("Datos: {0}".format(len(clasificacion_original)))
    #for j in range(0, num_topicos):
    #    print("Topico {0} es {1}".format(j, len([k for k in clasificacion_modelo if k == j])))

    similaridad = LDA_cluster.indiceJaccard(clasificacion_original, clasificacion_modelo)
    for e in similaridad[0]:
        print("El cluster original {0}, se parece a {1}, en un {2}%".format(e['original'], e['similar'], e['vector_simil']))

    LDA_cluster.mostrarMatrixConfusion(titulo, clasificacion_original, similaridad[1], labels=['0', '1', '2'])

    # imprimir una muestra representativa de los clusters modelo

    # agregar precision, recall y F1
    # micro: Calculate metrics globally by counting the total true positives, false negatives and false positives.


    ori =  pd.DataFrame(data=pd.get_dummies(clasificacion_original), columns=set(clasificacion_original))
    sim = pd.DataFrame(data=pd.get_dummies(similaridad[1]), columns=set(clasificacion_original))  # pd.get_dummies(similaridad[1])

    pr, rc, fb, su = precision_recall_fscore_support(ori, sim, average='macro')

    return (pr, rc, fb, su)

    # la curva ROC si se requiere deberia ser por topico
    #fpr, tpr, thresholds = roc_curve(clasificacion_original, similaridad[1], pos_label=2)


def mostrarTweets(num_topicos, datos_originales, clasificacion_original):
    print("Muestra de tweets")
    for j in range(0, num_topicos):
        limite = 0
        print("Topico %d" % j)
        for i in range(0, len(datos_originales)):
            if (clasificacion_original[i] == j):
                limite += 1
                print(datos_originales[i])
            if limite > 10:
                break

def purity(c_original,c_modelo):
    purity = 0
    for i in range(len(set(c_modelo))):
        cnt=[0]*len(set(c_original))
        for j in range(len(c_modelo)):
            if c_modelo[j] == i:
                cnt[c_original[j]]=cnt[c_original[j]]+1
        purity = purity + max(cnt)
    purity = purity/len(c_modelo)
    return purity


if __name__ == '__main__':
    baseline()
