from django.shortcuts import render
from django.core.cache import cache
from .models import Topics

import base64
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from django.views.decorators.csrf import ensure_csrf_cookie

@ensure_csrf_cookie
def index(request):
    if request.method == "POST":
        L = cache.get('Linkage')
        tweets = cache.get('tweets')
        Xclean = cache.get('Xclean')
        inv_map = cache.get('inv')
        map = cache.get('map')
        indL = cache.get('indL')
        cuenta = cache.get('cuenta')
        centroides_primera = cache.get('centroides_primera')
        cTop = Topics()

        if L is None or 'archivo_tweets' in request.FILES:
            print("Hay archivo")
            my_uploaded_file = request.FILES['archivo_tweets'].read()
            content = my_uploaded_file.decode('utf-8')
            lineas = [s.strip() for s in content.splitlines()]

            L, tweets, centroides_primera, Xclean, inv_map, map, indL, cuenta = cTop.obtenerModelo(lineas)

            # dh = sch.dendrogram(L)

            cache.set('Linkage', L)
            cache.set('tweets', tweets)
            cache.set('Xclean', Xclean)
            cache.set('inv', inv_map)
            cache.set('map', map)
            cache.set('indL', indL)
            cache.set('cuenta', cuenta)
            cache.set('centroides_primera', centroides_primera)
        else:
            print(inv_map)

        if 'threshold' in request.POST:
            threshold = request.POST['threshold']
            threshold = float(threshold)
        else:
            threshold = 0.5

        topics, centroides = cTop.obtenerTopicos(L,threshold,centroides_primera, Xclean,inv_map,map,tweets, indL, cuenta)
        # cuenta=[resultado[cl][0] for cl in range(len(resultado))]
        cache.set('centroides',centroides)
        # cache.set('cuenta',cuenta)

        plt.clf() # prueba clean previous plot

        plt.figure(1, figsize=(6, 5))

        show_leaf_counts = True
        ddata = augmented_dendrogram(L,
                       color_threshold=threshold,
                       # p=6,
                       # truncate_mode='lastp',
                       show_leaf_counts=show_leaf_counts)

        # plt.title("show_leaf_counts = %s" % show_leaf_counts)

        plt.savefig('plot.png')

        with open("plot.png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())

        context = {'topics': topics, 'plot':encoded_string}
        cache.set('topics', topics)
        return render(request, 'twittertopics/index.html', context)
    else:
        context = {}
        return render(request, 'twittertopics/index.html', context)

def identificarTweet(request):
    if request.method == "POST":
        centroide = cache.get('centroides')
        inv_map = cache.get('inv')
        topics = cache.get('topics')
        tweet = request.POST['tweet']

        cuenta = 0

        cTop = Topics()
        num_cluster = cTop.obtenerClusterDeTweet(tweet, centroide, cuenta, inv_map)

        context = {'cluster': num_cluster, 'topics': topics}
        return render(request, 'twittertopics/index.html', context)


def augmented_dendrogram(*args, **kwargs):
        ddata = dendrogram(*args, **kwargs)
        ## para mostrar etiquetas de distancias
        #if not kwargs.get('no_plot', False):
        #    for i, d in zip(ddata['icoord'], ddata['dcoord']):
        #        x = 0.5 * sum(i[1:3])
        #        y = d[1]
        #        plt.plot(x, y, 'ro')
        #        # plt.figure(x,y,'ro',1, figsize=(6, 5))
        #        plt.annotate("%.3g" % y, (x, y), xytext=(0, -8),
        #                     textcoords='offset points',
        #                     va='top', ha='center')
        return ddata
# Create your views here.
