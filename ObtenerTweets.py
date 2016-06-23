#!/usr/bin/python3.5

import pymysql
import pandas as pd
from sqlalchemy import create_engine
import json

def obtenerTweetsArchivo():
    data = []
    with open('/home/eduardomartinez/Documents/Sinnia/json/todos.json') as f:
        for line in f:
            data.append(json.loads(line))

    datos = []
    for row in data:
        datos.append(row['text'])
    return datos

def cargarTweetsEnDB():


    data = []
    with open('/home/eduardomartinez/Documents/Sinnia/json/corona.json') as f:
        for line in f:
            data.append(json.loads(line))
    datos = []
    for tweet in data:
        text = tweet['text']  # encode unicode_escape
        user_id = int(tweet['user']['id'])
        id = int(tweet['id'])
        datos.append({'status_text': text})
    engine = create_engine('mysql://root:root@localhost:3306/sinnia') # ?charset=utf8mb4 para caracteres de 4 bytes
    pd.DataFrame(datos).to_sql('corona_csv', engine, if_exists='append') #replace genera la tabla, append utiliza la misma tabla


def obtenerDatosBD():
    db = pymysql.connect(host="localhost", user="root", passwd="root", db="sinnia", charset='utf8')
    # name of the data base # you must create a Cursor object. It will let you execute all the queries you need
    cur = db.cursor()
    # Use all the SQL you like
    cur.execute("SELECT status_text FROM corona_csv")
    datos = []
    for row in cur.fetchall():
        datos.append(row[0])
    db.close()
    return datos