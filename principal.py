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

from genetico import Genetico

gen = Genetico.genetico

# Leer el archivo: Indicador económico USA
datos = fn.f_leer_archivo(param_archivo='archivos/FedInterestRateDecision-UnitedStates.xlsx', sheet_name= 0)
"""vn.v_indicador_orig(datos)
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
vn.v_det_at_dif(datos)"""

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
################################################################################

# Optimización de ratio de Sharpe usando algorítmo genético creado manualmente
data = [datos_instrumento, clasificacion]

genetic_filename = 'genetico.sav'
[punt,padres,hist_mean,hist_std,hist_sharpe,hist_padres] = gen(data, genetic_filename)
plt.plot(hist_mean)
