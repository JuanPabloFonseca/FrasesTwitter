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

    LDA_cluster.indiceJaccard(LDA_cluster.clasifOriginal("test"),clasificacion_modelo)

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