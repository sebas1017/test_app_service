#!/usr/bin/env python
# coding: utf-8

# In[9]:


#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import json
import matplotlib.pyplot as plt





def format_result_price(price_format):
    return  "${:,.2f}".format(price_format)  
def location_lat(location_point):
    location_point = location_point.replace("POINT ","").replace(' ',',').replace('(','').replace(')','').split(",")
    location_point = [i for i in location_point]
    return location_point[1]

def location_lon(location_point):
    location_point = location_point.replace("POINT ","").replace(' ',',').replace('(','').replace(')','').split(",")
    location_point = [i for i in location_point]
    return location_point[0]

def round_price(value):
    return round(abs(value), 2)
    #formateo visualmente correcto para moneda 
    #return "${:,.2f}".format(value)

def check_digit(value):
    try:
        if  value.strip() == "Sin especificar"  or value.isnumeric() is False:
            result = 0
        elif value.isnumeric():
            result = value
            media_parqueaderos.append(int(result))
            
        return result
    except Exception as e:
        print(f"Error en funcion check_digit -> {e}")
    return value

def check_banos(bano):
    try:
        if  bano.strip() == "Sin especificar"  or bano.isnumeric() is False:
            result = 0
        elif bano.isnumeric():
            result = bano
            
        return result
    except Exception as e:
        print(f"Error en funcion check_banos -> {e}")
    return bano
        
def antiguedad_inf(antiguedad):
    if 'sin especificar' not in antiguedad:
        antiguedad = antiguedad.replace('años','').split("a")
        if len(antiguedad) == 2:
            antiguedad = [i.replace(' ','') for i in antiguedad]
            return antiguedad[0]
        return antiguedad[0].replace('más de','').replace(' ','')
def antiguedad_sup(antiguedad):
    if 'sin especificar' not in antiguedad:
        antiguedad = antiguedad.replace('años','').split("a")
        if len(antiguedad) == 2:
            antiguedad = [i.replace(' ','') for i in antiguedad]
            return antiguedad[1]
        return antiguedad[0].replace('más de','').replace(' ','')
def price_administration(administracion):
    administracion = json.loads(administracion.replace("'",'"'))
    return float(administracion["precio"])

def include_administration(administracion):
    administracion = json.loads(administracion.replace("'",'"'))
    return administracion["incluida"]


# ## 1.0 Importar Data Finca raiz


file_name = "./DATASET_FINCA_RAIZ_CALI.csv"
df = pd.read_csv(file_name)
media_parqueaderos = []


# ## Limpieza de datos



df["precio_administracion"] = df["administracion"].apply(price_administration)
df["incluida_administracion"] = df["administracion"].apply(include_administration)
df["lat"] = df["location_point"].apply(location_lat)
df["lon"] = df["location_point"].apply(location_lon)
#limpiar columna de area replace, y convertir tipo de dato a flotante

df["descripcionGeneral"] = df["descripcionGeneral"].astype(str)
df["habitaciones"] = (df["habitaciones"].map(check_digit) )
df["habitaciones"] = df["habitaciones"].astype(int)
df["areaConstruida"] = df["areaConstruida"].str.replace('m2', '').astype(float)
df["estado"] = df["estado"].astype(str)
df["banos"] = (df["banos"].map(check_banos) )
df["banos"] = df["banos"].astype(int)
df["areaPrivada"] = df["areaPrivada"].str.replace('m2', '').astype(float)
df["precioM2"] = (df["precioM2"].map(round_price))
df["precioM2"] = df["precioM2"].astype(float)
df["parqueaderos"] = (df["parqueaderos"].map(check_digit))
df["parqueaderos"] = df["parqueaderos"].astype(int)
df["estrato"] = df["estrato"].str.replace('Estrato', '').replace(' ','').astype(float)
df["fuente_data"] = "Finca Raiz"
#eliminar columnas sin registros
nan_value = float("NaN")
df.replace("", nan_value, inplace=True)
df.dropna(how='all', axis=1, inplace=True)
df.drop('Unnamed: 0', inplace=True, axis=1)

df["precio_propiedad"] = df["precioM2"] * df["areaPrivada"]
df["precio_propiedad"] = df["precio_propiedad"].astype(float)
df["ciudad"] = "CALI"
df.drop('administracion', inplace=True, axis=1)
df.drop('location_point', inplace=True, axis=1)
df.drop('incluida_administracion', inplace=True, axis=1)
df.drop('antiguedad', inplace=True, axis=1)
df.drop('piso', inplace=True, axis=1)
media_parqueaderos = sum(media_parqueaderos) / len(media_parqueaderos)

df['parqueaderos'] = df['parqueaderos'].replace([0],int(media_parqueaderos ))
df["precio_propiedad"] = df["areaPrivada"] * df["precioM2"]
df["precio_propiedad"] =  df["precio_propiedad"].astype(float).round(2)
df["precio_propiedad"] = (df["precio_propiedad"].map(round_price))






def banos(json_data):
    data = json.loads(json_data)
    return data["banos"]

def tipo_negocio(json_data):
    data = json.loads(json_data)
    return data["tiponegocio"]

def parqueadero(json_data):
    data = json.loads(json_data)
    return data["garajes"]

def antiguedad(json_data):
    data = json.loads(json_data)
    return data["tiempodeconstruido"]

def habitaciones(json_data):
    data = json.loads(json_data)
    return data["habitaciones"]


def areaprivada(json_data):
    data = json.loads(json_data)
    return data["areaprivada"]

def areaconstruida(json_data):
    data = json.loads(json_data)
    return data["areaconstruida"]
def descripcion(json_data):
    data = json.loads(json_data)
    return data["descripcion"]

def valoradministracion(json_data):
    data = json.loads(json_data)
    return data["valoradministracion"]

def estrato(json_data):
    data = json.loads(json_data)
    return data["estrato"]

def estado(json_data):
    data = json.loads(json_data)
    return data["estado"]


def antiguedad_inf(antiguedad):
    if antiguedad is not None :
        if 'sin especificar' not in antiguedad:
            antiguedad = antiguedad.replace('años','').split("a")
            if len(antiguedad) == 2:
                antiguedad = [i.replace(' ','') for i in antiguedad]
                return antiguedad[0]
            return antiguedad[0].replace('más de','').replace(' ','')
    return 'sin especificar'

def antiguedad_sup(antiguedad):
    if antiguedad is not None:
        if 'sin especificar' not in antiguedad:
            antiguedad = antiguedad.replace('años','').split("a")
            if len(antiguedad) == 2:
                antiguedad = [i.replace(' ','') for i in antiguedad]
                return antiguedad[1]
            return antiguedad[0].replace('más de','').replace(' ','')
    return 'sin especificar'


def precioM2(json_data):
    data = json.loads(json_data)
    if data["valorventa"] is not None and (data["areaconstruida"] is not None ):
        if int(data["areaconstruida"])!=0:
            return data["valorventa"]/ data["areaconstruida"]
    return None

def lon(json_data):
    data = json.loads(json_data)
    if data["longitud"] is not None:
        return data["longitud"]
    return 0
def lat(json_data):
    data = json.loads(json_data)
    if data["latitud"] is not None:
        return data["latitud"]
    return 0

def func_format_dataframe(df, fuente_data, ciudad):
    print("Fuente data ", fuente_data)
    if ciudad == "CALI":
        df["tipo_negocio"] = "Venta"
    elif ciudad == "BOGOTA":
        df["tipo_negocio"] = df["data"].apply(tipo_negocio)
    df["banos"] = df["data"].apply(banos)
    df["parqueaderos"] = df["data"].apply(parqueadero)
    df["antiguedad"] = df["data"].apply(antiguedad)
    df["habitaciones"] = df["data"].apply(habitaciones)
    df["areaPrivada"] = df["data"].apply(areaprivada)
    df["areaConstruida"] = df["data"].apply(areaconstruida)
    df["descripcionGeneral"] = df["data"].apply(descripcion)
    df["ciudad"] = "BOGOTA"
    df["precio_administracion"] = df["data"].apply(valoradministracion)
    df["estrato"] = df["data"].apply(estrato)
    df["estado"] = df["data"].apply(estado)
    df["precioM2"] = df["data"].apply(precioM2)
    df["lat"] = df["data"].apply(lat)
    df["lon"] = df["data"].apply(lon)
    df["antiguedad_inf"] = df["antiguedad"].apply(antiguedad_inf)
    df["antiguedad_sup"] = df["antiguedad"].apply(antiguedad_sup)
    df["fuente_data"] = fuente_data
    df["ciudad"] = ciudad
    df = df[df["url"].str.contains("apartamento")]
    # Eliminar columnas innecesarias
    df.drop('fecha_inicial', inplace=True, axis=1)
    df.drop('fecha_final', inplace=True, axis=1)
    df.drop('data', inplace=True, axis=1)
    df.drop('fuente', inplace=True, axis=1)
    df.drop('url', inplace=True, axis=1)
    df.drop('activo', inplace=True, axis=1)
    df.drop('latitud', inplace=True, axis=1)
    df.drop('longitud', inplace=True, axis=1)
    df.drop('valorarriendo', inplace=True, axis=1)
    # Rename columns
    df.rename(columns = {'valorventa':'precio_propiedad'}, inplace = True)
    return df



pd.set_option('display.float_format', '{:.6f}'.format)


# ## Importar data metro cuadrado (M2) bogota y cali



file_name_m2_cali = "./datafinal_scraper_cali.csv"
df_m2_cali = pd.read_csv(file_name_m2_cali)

file_name_bogota = "./scrapper_jsons_data_bogota.csv"
df_bogota = pd.read_csv(file_name_bogota)



df_m2_cali = func_format_dataframe(df_m2_cali, "M2", "CALI")


df_bogota = func_format_dataframe(df_bogota, "M2", "BOGOTA")






# # Limpieza datos

# # Concat dataframes



df2 = pd.concat([df,df_bogota,df_m2_cali], ignore_index=True)


# # Eliminar columnas




def precioPropiedad(precioM2,areaConstruida):
        return precioM2 * areaConstruida

df2.drop('precio_administracion', inplace=True, axis=1)

df2.drop('tipoApartamento', inplace=True, axis=1)
df2 = df2.drop(df2[df2.lat == '0.0'].index)
df2 = df2.drop(df2[df2.lon == '0.0'].index)
df2.dropna(inplace=True)
df2['precio_propiedad'] = df2.apply(lambda x: precioPropiedad(x.precioM2, x.areaConstruida), axis=1)
df2.dtypes

df2 = df2.drop(df2[df2.areaPrivada ==  0 ].index)
df2 = df2.drop(df2[df2.banos  ==  0].index)
df2 = df2.drop(df2[df2.lon  ==  0].index)
df2 = df2.drop(df2[df2.lat ==  0].index)
df2 = df2.drop(df2[df2.estrato  ==  0].index)
df2.drop_duplicates(subset=["lon","lat"], keep="first", inplace=True)


df2["habitaciones"] = df2["habitaciones"].astype(int)
df2["areaConstruida"] = df2["areaConstruida"].astype(int)
df2["parqueaderos"] = df2["parqueaderos"].astype(int)
df2["estrato"] = df2["estrato"].astype(int)
df2["banos"] = df2["banos"].astype(int)
df2["areaPrivada"] = df2["areaPrivada"].astype(int)
df2["precioM2"] = df2["precioM2"].astype(int)


df_cali = df2[df2["ciudad"]=="CALI"]
df_bogota = df2[df2["ciudad"]=="BOGOTA"]
df_cali = df_cali[df_cali["tipo_negocio"]=="Venta"]
df_bogota = df_bogota[df_bogota["tipo_negocio"]=="Venta"]








# In[ ]:





# In[11]:


#sns.heatmap(df_cali.corr(), annot=True, cmap='coolwarm')


# In[12]:


agrupado = df_cali.groupby('habitaciones')

for nombre, grupo in agrupado:
    plt.scatter(grupo['habitaciones'], grupo['precioM2'], label=nombre)

plt.xlabel('Número de habitaciones')
plt.ylabel('Precio del metro cuadrado')
plt.legend()
plt.show()



# In[13]:


# agrupar datos por número de baños

# calcular media del precio del metro cuadrado para cada grupo
media_precio_m2 = agrupado['precioM2'].mean()

# graficar resultados
plt.plot(media_precio_m2.index, media_precio_m2.values)
plt.xlabel('Número de baños')
plt.ylabel('Precio promedio del metro cuadrado')
plt.show()


# In[14]:


from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline

def predecir_precio_propiedad(df):
    # Seleccionar las columnas que se utilizarán como características para el modelo
    X = df[['habitaciones', 'areaConstruida', 'banos', 'areaPrivada', 'parqueaderos', 'estrato', 'precioM2']]

    # Seleccionar la columna objetivo, que es el precio de la propiedad
    y = df['precio_propiedad']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.6, random_state=42)

    # Crear varios modelos
    modelos = [
      
        ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42)),
       
       
        
       
    ]
    import numpy as np
    # Entrenar y evaluar cada modelo
    for nombre, modelo in modelos:
        modelo.fit(X_entrenamiento, y_entrenamiento)
        y_pred = modelo.predict(X_prueba)
        score = r2_score(y_prueba, y_pred)
        input_data = np.array([[2, 60, 1, 52, 1, 3, 2600000]])
        
        price_prediction = modelo.predict(input_data)[0]
        print(f"Puntaje de {nombre}: {score:.4f}")
        print(f"precio de prediccion de {nombre}: {price_prediction:.4f}")
        
    models = [
    Pipeline([
        ('scaler', StandardScaler()),
        ('elasticnet', ElasticNet())
    ])
]

    params = [
        {'elasticnet__alpha': [0.1, 0.5, 1.0], 'elasticnet__l1_ratio': [0.1, 0.5, 0.9]}
    ]

    for i, model in enumerate(models):
       
        clf = GridSearchCV(model, params[i], cv=5)
        clf.fit(X_entrenamiento, y_entrenamiento)
        
        score = clf.score(X_prueba, y_prueba)
        print(20*"----")
        print(f"Puntaje de {model.named_steps}: {score:.4f}")
        print(f"mejores parametros {clf.best_params_}")
        print(f"mejor puntaje {clf.best_score_}")
    
        print(20*"----")
        
    
    
valor_final = predecir_precio_propiedad(df_cali)
print("VALOR PREDICION")
print(valor_final)


# In[16]:


from sklearn.neighbors import NearestNeighbors
import numpy as np
def precio_en_base_a_comparables(input_data):
    df = df_cali
    # Seleccionar las características de los comparables
    X = df[['habitaciones', 'areaConstruida', 'banos', 'areaPrivada', 'parqueaderos', 'estrato', 'precioM2']]
    # Seleccionar el precio de los comparables
    y = df['precio_propiedad']
    # Escalar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    input_data_scaled = scaler.transform(input_data.reshape(1,-1))
    # Entrenar modelo
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('elasticnet', ElasticNet(alpha=0.1, l1_ratio=0.5))
    ])
    model.fit(X_scaled, y)
    # Evaluar modelo
    prediction = model.predict(input_data_scaled)[0]
    # Calcular score del modelo
    score = r2_score(y, model.predict(X_scaled))
    return prediction, score, model

# Ejemplo de entrada
input_data = np.array([3, 120, 2, 100, 1, 4, 0]).reshape(1,-1)

# Obtener los precios de los comparables más cercanos y la predicción del precio de la propiedad
prediction, score, model = precio_en_base_a_comparables(input_data)

df_cali.info()
# Imprimir los precios de los comparables y la predicción
#print("Precios de los comparables más cercanos: ", prices)
print("Predicción del precio de la propiedad: ", prediction)
print("score del modelo: ", score)


