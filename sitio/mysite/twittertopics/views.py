from django.shortcuts import render

from .models import Topics



def index(request):
    if request.method == "POST":
        my_uploaded_file = request.FILES['archivo_tweets'].read()

        content = my_uploaded_file.decode('utf-8')

        lineas = [s.strip() for s in content.splitlines()]

        #for i in range(100):
        #    print(lineas[i])


        topics = Topics()
        resultado = topics.obtenerTopicos(lineas)


        context = {'topics': resultado}
        return render(request, 'twittertopics/index.html', context)
    else:
        context = {}
        return render(request, 'twittertopics/index.html', context)

# def leerArchivo(request):


# Create your views here.
