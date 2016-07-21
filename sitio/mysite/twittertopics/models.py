from django.db import models
from .sinniapipeline.snow_pipeline import procesarArchivo

class Topics:
    def __init__(self):
        self.tweets = ['hola esto es un tweet', 'esto es otro tweet', 'esto es un tercer tweet']

    def obtenerTopicos(self, archivo):

        time_window_mins = 14400.0
        n_documentos_maximos = 5
        factor_frecuencia = 0.001
        num_ngrams_in_tweet = 3
        minimo_usuarios = 3
        minimo_hashtags = 3
        ngrama_minimo = 2
        ngrama_maximo = 4

        ngramas  = procesarArchivo(archivo=archivo, time_window_mins=time_window_mins, n_documentos_maximos=n_documentos_maximos,
                        factor_frecuencia=factor_frecuencia, num_ngrams_in_tweet=num_ngrams_in_tweet, minimo_usuarios=minimo_usuarios,
                        minimo_hashtags=minimo_hashtags, ngrama_minimo=ngrama_minimo, ngrama_maximo=ngrama_maximo)


        return ngramas

# Create your models here.




