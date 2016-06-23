import stopwords


#regresa una lista no tan mala de stopwords en español
def creaStopWords():
    es_stop = stopwords.get_stopwords('es')
    he = ['he', 'has', 'hemos', 'habéis', 'han', 'haya', 'hayas', 'hayamos', 'hayáis', 'hayan', 'habré',
          'habrás', 'habrá', 'habremos', 'habréis', 'habrán', 'habría', 'habrías', 'habríamos', 'habríais',
          'habrían', 'había', 'habías', 'habíamos', 'habíais', 'habían', 'hube', 'hubiste', 'hubo', 'hubimos',
          'hubisteis', 'hubieron', 'hubiera', 'hubieras', 'hubiéramos', 'hubierais', 'hubieran', 'hubiese',
          'hubieses', 'hubiésemos', 'hubieseis', 'hubiesen', 'habiendo', 'habido', 'habida', 'habidos', 'habidas']
    ser = ['son', 'sea', 'seas', 'seamos', 'seáis', 'sean', 'seré', 'serás', 'será',
           'seremos', 'seréis', 'serán', 'sería', 'serías', 'seríamos', 'seríais', 'serían', 'éramos', 'erais',
           'fuiste', 'fuisteis', 'fuera', 'fueras', 'fuéramos', 'fuerais', 'fueran', 'fuese', 'fueses',
           'fuésemos', 'fueseis', 'fuesen', 'sido']
    es_stop = es_stop + he
    es_stop = es_stop + ser
    es_stop.append('a')
    es_stop.append('al')
    es_stop.append('algun')
    es_stop.append('cabe')
    es_stop.append('de')
    es_stop.append('del')
    es_stop.append('este')
    es_stop.append('esto')
    es_stop.append('estar')
    es_stop.append('estaría')
    es_stop.append('estarían')
    es_stop.append('estaria')
    es_stop.append('estarian')
    es_stop.append('estarias')
    es_stop.append('estariamos')
    es_stop.append('estarías')
    es_stop.append('estaríamos')
    es_stop.append('estábamos')
    es_stop.append('estabamos')
    es_stop.append('estaban')
    es_stop.append('estos')
    es_stop.append('estas')
    es_stop.append('está')
    es_stop.append('ella')
    es_stop.append('hacia')
    es_stop.append('hasta')
    es_stop.append('le')
    es_stop.append('les')
    es_stop.append('mas')
    es_stop.append('más')
    es_stop.append('mi')
    es_stop.append('mis')
    es_stop.append('no')
    es_stop.append('so')
    es_stop.append('un')
    es_stop.append('tambien')
    es_stop.append('tmb')
    es_stop.append('o')
    es_stop.append('y')
    es_stop.append('se')
    es_stop.append('c')
    es_stop.append('segun')
    es_stop.append('según')
    es_stop.append('q')
    es_stop.append('que')
    es_stop.append('qué')
    es_stop.append('qe')
    es_stop.append('te')
    es_stop.append('ti')
    es_stop.append('tú')
    es_stop.append('tu')
    es_stop.append('ya')

    es_stop.remove('trabajo')
    es_stop.remove('trabajar')
    es_stop.remove('trabajas')
    es_stop.remove('trabaja')
    es_stop.remove('trabajamos')
    es_stop.remove('trabajais')
    es_stop.remove('trabajan')
    es_stop.remove('conseguir')
    es_stop.remove('consigo')
    es_stop.remove('consigue')
    es_stop.remove('consigues')
    es_stop.remove('conseguimos')
    es_stop.remove('consiguen')
    es_stop.remove('tiempo')
    return es_stop