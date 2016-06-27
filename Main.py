#!/usr/bin/python3.5
import LDA_cluster
import ObtenerTweets
import LimpiarTweets

def demo():
    datosTrain = ObtenerTweets.obtenerTweetsArchivo('train')
    datosTr = LimpiarTweets.limpiarTexto(datosTrain)
    num_topicos = 3
    ldam=LDA_cluster.LDA_gensim(datosTr,num_topicos,1)
    # contarPalabras(datosTr)

    datosTest = ObtenerTweets.obtenerTweetsArchivo('test')
    datosTe = LimpiarTweets.limpiarTexto(datosTest)

    ### TRAIN
    clasificacion_original = LDA_cluster.clasifOriginal("train")
    clasificacion_modelo = LDA_cluster.clasifica(datosTr, ldam)

    resultados('TRAIN', clasificacion_original, clasificacion_modelo, datosTrain, num_topicos)

    ### TEST
    clasificacion_original = LDA_cluster.clasifOriginal("test")
    clasificacion_modelo = LDA_cluster.clasifica(datosTe, ldam)

    print("TEST")
    resultados(clasificacion_original, clasificacion_modelo, datosTest, num_topicos)


def resultados(titulo, clasificacion_original, clasificacion_modelo, datos_originales, num_topicos):

    print("Datos: {0}".format(len(clasificacion_original)))

    for j in range(0, num_topicos):
        print("Topico {0} es {1}".format(j, len([k for k in clasificacion_modelo if k == j])))

    similaridad = LDA_cluster.indiceJaccard(clasificacion_original, clasificacion_modelo)

    # formateo de impresion Jaccard
    for e in similaridad[0]:
        if e['original'] == 0:
            original = 'corona'
        else:
            if e['original'] == 1:
                original = 'modelo'
            else:
                original = 'pan'
        print("El cluster original {0}, se parece a {1}, en un {2}%".format(original, e['similar'],
                                                                            e['vector_simil']))

    LDA_cluster.mostrarMatrixConfusion(titulo, clasificacion_original, similaridad[1], labels=['0', '1', '2'])

    # imprimir una muestra representativa de los clusters modelo
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


if __name__ == '__main__':
    demo()