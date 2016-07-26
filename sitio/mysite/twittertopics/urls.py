from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),

    url(r'^tweet$', views.identificarTweet, name='identificarTweet'),

    # url(r'archivo$', views.leerArchivo, name='archivo')
]