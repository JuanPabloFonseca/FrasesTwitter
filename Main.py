#!/usr/bin/python3.5
import LDA_cluster
import ObtenerTweets
import LimpiarTweets

def demo():
    datosTr = ObtenerTweets.obtenerTweetsArchivo('train')
    datosTr = LimpiarTweets.limpiarTexto(datosTr)
    num_topicos = 3
    ldam=LDA_cluster.LDA_gensim(datosTr,num_topicos,1)
    # contarPalabras(datosTr)

    datosTest = ObtenerTweets.obtenerTweetsArchivo('test')
    datosTe = LimpiarTweets.limpiarTexto(datosTest)

    clasificacion_modelo = LDA_cluster.clasifica(datosTe,ldam)

    print("Datos TEST: {0}".format(len(datosTest)))

    for j in range(0, num_topicos):
        print("Cardinalidad del topico {0} es {1}".format(j, len([k for k in clasificacion_modelo if k == j])))

    similaridad = LDA_cluster.indiceJaccard(LDA_cluster.clasifOriginal("test"),clasificacion_modelo)

    # formateo de impresion Jaccard
    for e in similaridad:
        print("El cluster original {0}, se parece a {1}, en un {2}%".format(e['original'], e['similar'],
                                                                            round(max(e['vector_simil']) * 100, 2)))

    # imprimir una muestra representativa de los clusters modelo
    print("Muestra de tweets")
    for j in range(0,num_topicos):
        limite = 0
        print("Topico %d" % j)
        for i in range(0,len(datosTest)):
            if(clasificacion_modelo[i] == j):
                limite += 1
                print(datosTest[i])
            if limite > 10:
                break


    '''i=0
    for c in ct:
        print(datosTest[i],end=' ')
        print("pertenece al tópico ", c)
        i=i+1'''

    #cargarTweetsEnDB()



if __name__ == '__main__':
    demo()