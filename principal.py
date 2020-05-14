# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:28:55 2020

@author: User
"""

# -- ------------------------------------------------------------------------------------ -- #
# -- proyecto: Microestructura y Sistemas de Trading - Laboratorio 4 - Proyecto Final
# -- archivo: principal.py
# -- mantiene: Fernanda Pinedo, Oscar Flores, Francisco Rodriguez
# -- repositorio: https://github.com/OscarFlores-IFi/proyecto_equipo5
# -- ------------------------------------------------------------------------------------ -- #
from time import time

import funciones as fn
import visualizaciones as vn
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

from procesos import Genetico

gen = Genetico.genetico

# Leer el archivo: Indicador económico USA
datos = fn.f_leer_archivo(param_archivo='archivos/FedInterestRateDecision-UnitedStates.xlsx', sheet_name= 0)
vn.v_indicador_orig(datos)
vn.v_rmrdsv_original(datos)

# Prueba de Estacionariedad (Dickey-Fuller)
estacionariedad = fn.f_stationarity(datos)
estacionariedad_dif = fn.f_dif_stationarity(datos)
vn.v_indicador_dif(datos)
vn.v_rmrdsv_dif(datos)

# Autocorrelación y autocorrelación parcial
fac_lm = fn.f_autocorr_lm(datos)
facp = fn.f_autocorr_par(datos)
vn.v_fac_estac(datos)
vn.v_facp_estac(datos)

# Estacionalidad de Indicador
vn.v_seasonality(datos)

# Estimación del modelo ARIMA
vn.v_model_arima_orig(datos)

# Heterocedasticidad del Indicador
heterocedasticidad_bp = fn.f_heter_bp(datos)
heterocedasticidad_w = fn.f_heter_w(datos)
heterocedasticidad_arch = fn.f_het_arch(datos)

# Prueba de Normalidad (Dos)
normalidad_shw = fn.f_norm_shw(datos)
normalidad_dagp = fn.f_norm_dagp(datos)
vn.v_norm_hist(datos)
sesgo = fn.f_skewness(datos)
vn.v_norm_qq(datos)

normalidad_residuales = fn.f_norm_resid(datos)
sesgo_residuales = fn.f_skewness_resid(datos)
vn.v_norm_resids(datos)

# Detección de atípicos
vn.v_det_at(datos)
vn.v_det_at_dif(datos)


#%%
################################################################################
################################################################################

# Descargar datos para cada TimeStamp de datos:
time_delta = pd.to_timedelta('00:31:00')
granularity = 'M1'
instrument = "EUR_USD"
oatk='107596e9d65c' + '1bbc9175953d917140' + '12-f975c6201dddad03ac1592232c0ea0ea'
try:
    # Si tenemos los datos ya descargados, los importamos.
    datos_instrumento = pickle.load(open('precios.sav','rb'))
except:
    # Si no tenemos los datos descargados, los descargamos y los guardamos.
    print('no se pudo cargar archivos de pickle, se descargarán todos nuevamente.')
    datos_instrumento = {i : fn.f_precios_masivos(i, i + time_delta, granularity, instrument, oatk,  p5_ginc=4900)
                         for i in datos.datetime}
    pickle.dump(datos_instrumento, open('precios.sav','wb'))

# Claisificar las ocurrencias según parámetros; Actual, Consensys, Previous
clasificacion = fn.f_clasificacion_ocurrencia(datos)
#print(clasificacion)

# DataFrame de escenarios.
df_escenarios = fn.f_df_escenarios(datos_instrumento, clasificacion)

df_escenarios[df_escenarios.escenario == 'A']
#print(df_escenarios)


#Supongamos la siguiente estrategia dependiendo de cada escenario.
df_decisiones = pd.DataFrame(data = [['compra',10,25, 10000],['compra', 10, 25, 10000],['compra',10,25, 10000],['venta', 10, 30, 100000]],
                            index = ['A', 'B', 'C', 'D'],
                            columns = ['operacion', 'StopLoss', 'TakeProfit', 'Volume'])


# BackTesting
# timestamp escenario operacion volumen resultado pips capital capital_acm
df_backtest = fn.f_df_backtest(datos_instrumento, clasificacion, df_decisiones)
#print(df_backtest)


################################################################################
################################################################################

# Optimización de ratio de Sharpe usando algorítmo genético creado manualmente
data = [datos_instrumento, clasificacion]

filename = 'genetico.sav'

# optimización considerando todo el timepo (no separado Train de Test).
# EN CASO DE QUE NO EXISTA GENETICO.SAV, EJECUTAR LA SIGUIENTE LINEA:
[padres,hist_mean,hist_std,hist_sharpe,hist_padres] = gen(data, filename =filename)
[padres,hist_mean,hist_std,hist_sharpe,hist_padres] = pickle.load(open(filename,'rb'))
plt.plot(hist_sharpe[:,-8:]) # Grafica los mejores 8 padres después de entrenarlos

################################################################################
################################################################################
# Separar datos para train/Test
training_ratio = 0.9 # Entrenamos el 70 % de los datos.
train_data_timestamps = datos[round(len(datos_instrumento)*(1-training_ratio)):].datetime
test_data_timestamps = datos[0:round(len(datos_instrumento)*(1-training_ratio))].datetime


training_data = { i: datos_instrumento[i] for i in train_data_timestamps }
testing_data = { i: datos_instrumento[i] for i in test_data_timestamps }


training_clasification = fn.f_clasificacion_ocurrencia(datos[round(len(datos_instrumento)*(1-training_ratio)):])
testing_clasification = fn.f_clasificacion_ocurrencia(datos[:round(len(datos_instrumento)*(1-training_ratio))])


# Optimización de ratio de Sharpe usando algorítmo genético creado manualmente
train_data = [training_data, training_clasification]


filename = 'genetico2.sav'


# Optimización de periodo de training.
# EN CASO DE QUE NO EXISTA GENETICO.SAV, EJECUTAR LA SIGUIENTE LINEA:
[padres,hist_mean,hist_std,hist_sharpe,hist_padres] = gen(train_data, filename = filename)
[padres,hist_mean,hist_std,hist_sharpe,hist_padres] = pickle.load(open(filename,'rb'))
plt.plot(hist_sharpe[:,-8:]) # Grafica los mejores 8 padres después del entrenamiento.


seleccionado = padres[-1]
seleccionado
df_decisiones = pd.DataFrame(data = [seleccionado],
                            index = ['A'],
                            columns = ['operacion', 'StopLoss', 'TakeProfit', 'Volume'])
df_decisiones['operacion'][df_decisiones['operacion']==1] = 'compra'
df_decisiones['operacion'][df_decisiones['operacion']==0] = 'venta'
print(df_decisiones)
df_prueba = fn.f_df_backtest(testing_data, testing_clasification, df_decisiones)
print(df_prueba)

df_entrenamiento = fn.f_df_backtest(training_data, training_clasification, df_decisiones)
df_entrenamiento

# Gráfica de convergencia 
vn.v_hist_sharpe(hist_sharpe)
vn.capital_backtest(df_backtest)
vn.capital_prueba(df_prueba)
vn.capital_entrenamiento(df_entrenamiento)

#%%
# Métricas de atribución al desempeño
mad = fn.f_stat_mad(df_backtest, df_entrenamiento, df_prueba)





