from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    # url(r'archivo$', views.leerArchivo, name='archivo')
]