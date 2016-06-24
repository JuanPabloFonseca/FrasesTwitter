# README #

En esta carpeta se encuentran las herramientas desarrolladas y acopladas por Hugo Huipet sobre LDA
La carpeta de TEXT SCRAPPER & LDA contiene 3 archivos:

* run que se corre junto con el archivo stoplist y obtiene el '.text' de un ARCHIVOJSON para obtener todos los tweets, despues les hace un tratamiento basico para obtener el archivo final limpio de caracteres especiales, mayúsculas, urls, acentos etc -> fileName

>>#### ./run ARCHIVOJSON #

* LDA programa ejecutable que fue obtenido de http://gibbslda.sourceforge.net/ y se ejecuta con

>>#### ./lda -est -ntopics # -niters # -savestep # -twords # -dfile [fileName] #

* stoplist la lista de stopwords usada, requiere que sea un sólo string con las palabras concatenadas con | y rodeadas de los caracteres \bpalabra\b

En la carpeta TweetsClassifier se encuentra un proyecto de eclipse que se ejecuta de la siguiente manera:
desde 

#### ./bin/java itam.twitter.base.Classifier [ARCHIVODEREGLAS] [ARCHIVOACLASIFICAR] #

>### El ARCHIVOACLASIFICAR # 
>
>>Es la lista de tweets a clasificar en cada linea un tweet y el tweet ya debe de estar limpio al menos de caracteres especiales y acentos (para clasificar mejor de preferencia tokens separados por espacios)
>
>### El ARCHIVODEREGLAS #
>
>>Es el conjunto de topicos que genera el LDA y debe tener un formato similar a (ADVERTENCIA: SIN ESPACIOS ENTRE LOS RENGLONES ES CULPA DEL BITBUCKET T_T):


Linea de TOPIC1:

palabra1 valor1

palabra2 valor2

.

.

.

Linea de TOPICn:

palabrax valorx

palabray valory

.

.

.


