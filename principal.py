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

import funciones as fn
import visualizaciones as vn
import pandas as pd
import pickle

# Leer el archivo: Indicador económico USA
datos = fn.f_leer_archivo(param_archivo='archivos/FedInterestRateDecision-UnitedStates.xlsx', sheet_name= 0)
vn.v_indicador_orig(datos)

# Autocorrelación del Indicador
fac_lb = fn.f_autocorr(datos)
vn.v_fac(datos)
fac_lm = fn.f_autocorr_lm(datos)

# Autocorrelación parcial del Indicador
facp = fn.f_autocorr_par(datos)
vn.v_facp(datos)

# Heterocedasticidad del Indicador
heterocedasticidad_bp = fn.f_heter_bp(datos)
heterocedasticidad_w = fn.f_heter_w(datos)
heterocedasticidad_arch = fn.f_het_arch(datos)

# Prueba de Normalidad (Dos)
normalidad_shw = fn.f_norm_shw(datos)
normalidad_dagp = fn.f_norm_dagp(datos)
vn.v_norm_hist(datos)
vn.v_norm_qq(datos)

# Prueba de Estacionariedad (Dickey-Fuller)
estacionariedad = fn.f_stationarity(datos)
estacionariedad_dif = fn.f_dif_stationary(datos)
vn.v_fac_estac(datos)
vn.v_facp_estac(datos)

# Prueba de Estacionalidad
vn.v_preseasonality(datos)
vn.v_seasonality(datos)

# Detección de atípicos
#atipicos = fn.f_det_atip(datos)
vn.v_det_atip(datos)




########################################################################################################################
########################################################################################################################
# Descargar datos para cada TimeStamp de datos:
time_delta = pd.to_timedelta('00:30:00')
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
print(clasificacion)

# DataFrame de escenarios.
df_escenarios = fn.f_df_escenarios(datos_instrumento, clasificacion)
print(df_escenarios)
