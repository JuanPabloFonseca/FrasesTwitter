from django.shortcuts import render
from django.core.cache import cache
from .models import Topics



def index(request):
    if request.method == "POST":
        L = cache.get('Linkage')
        tweets = cache.get('tweets')
        Xclean = cache.get('Xclean')
        inv_map = cache.get('inv')
        map = cache.get('map')

        if L == None:
            print(L)

            if 'archivo_tweets' in request.FILES:
                my_uploaded_file = request.FILES['archivo_tweets'].read()
                content = my_uploaded_file.decode('utf-8')
                lineas = [s.strip() for s in content.splitlines()]

                topics = Topics()
                L, tweets, Xclean, inv_map, map = topics.obtenerModelo(lineas)

                cache.set('Linkage', L)
                cache.set('tweets', tweets)
                cache.set('Xclean', Xclean)
                cache.set('inv', inv_map)
                cache.set('map', map)

        threshold = request.POST['threshold']
        threshold = float(threshold)

        resultado = topics.obtenerTopicos(L,threshold,Xclean,inv_map,map,tweets)

        context = {'topics': resultado}
        return render(request, 'twittertopics/index.html', context)
    else:
        context = {}
        return render(request, 'twittertopics/index.html', context)

# def leerArchivo(request):

# Create your views here.
