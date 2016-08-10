from django.db import models
from .sinniapipeline.snow_pipeline import pipeline

class Topics:
    def __init__(self):
        #self.tweets = ['hola esto es un tweet', 'esto es otro tweet', 'esto es un tercer tweet']
        self.time_window_mins = 14400.0
        self.n_documentos_maximos = 5
        self.factor_frecuencia = 0.001
        self.num_ngrams_in_tweet = 3
        self.minimo_usuarios = 3
        self.minimo_hashtags = 3
        self.ngrama_minimo = 2
        self.ngrama_maximo = 3
        self.pipe = pipeline()

    def obtenerModelo(self, archivo):
        return self.pipe.obtenerModelo(archivo=archivo,
                                   time_window_mins=self.time_window_mins,
                                   n_documentos_maximos=self.n_documentos_maximos,
                                   factor_frecuencia=self.factor_frecuencia,
                                   num_ngrams_in_tweet=self.num_ngrams_in_tweet,
                                   minimo_usuarios=self.minimo_usuarios,
                                   minimo_hashtags=self.minimo_hashtags,
                                   ngrama_minimo=self.ngrama_minimo,
                                   ngrama_maximo=self.ngrama_maximo)

    def obtenerTopicos(self, LinkageMatrix, threshold, Xclean, inv_map, map_index_after_cleaning, tweets_cluster):
        return self.pipe.obtenerTopicos(LinkageMatrix, threshold,Xclean,inv_map,map_index_after_cleaning,tweets_cluster)

    def obtenerClusterDeTweet(self, tweet, centroides, cnt, inv_map):
        return self.pipe.clusterDelTweet(tweet,centroides,cnt,inv_map)



# Create your models here.