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

import matplotlib.pyplot as plt
# Leer el archivo: Indicador económico USA
datos = fn.f_leer_archivo(param_archivo='archivos/FedInterestRateDecision-UnitedStates.xlsx', sheet_name= 0)

"""#Graficar indicador
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
vn.v_det_atip(datos)"""




########################################################################################################################
########################################################################################################################
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
print(clasificacion)

# DataFrame de escenarios.
df_escenarios = fn.f_df_escenarios(datos_instrumento, clasificacion)
print(df_escenarios)

df_escenarios[df_escenarios.escenario == 'A']


#Supongamos la siguiente estrategia dependiendo de cada escenario.
df_decisiones = pd.DataFrame(data = [['compra', 20, 40, 1000],['venta', 40, 80, 2000],['compra', 20, 40, 1000],['venta', 40, 80, 2000]],
                            index = ['A', 'B','C','D'],
                            columns = ['operacion', 'StopLoss', 'TakeProfit', 'Volume'])

df_decisiones

# BackTestingn
# timestamp escenario operacion volumen resultado pips capital capital_acm

timestamp = []
operaciones = []
volumen = []
resultado = []
pips = []

pips_transaccion = 10000
import funciones as fn
for (dato, clas) in zip(datos_instrumento.items(), clasificacion):
    posicion = df_decisiones.loc[clas].operacion
    StopLoss = df_decisiones.loc[clas].StopLoss
    TakeProfit = df_decisiones.loc[clas].TakeProfit
    pos_volume = df_decisiones.loc[clas].Volume

    timestamp_cierre_operacion, gain_loss, pips_gl = fn.f_Gain_Loss(dato[1], TakeProfit, StopLoss, pips_transaccion, posicion)

    timestamp.append(dato[0])
    operaciones.append(posicion)
    volumen.append(pos_volume)
    resultado.append(gain_loss)
    pips.append(pips_gl)
    #plt.plot(dato[1].iloc[:,1:])
    #plt.show()
df_backtest = pd.DataFrame({'escenario': clasificacion,
                            'operacion': operaciones,
                            'volumen': volumen,
                            'resultado': resultado,
                            'pips': pips}, index = timestamp)

df_backtest
print(df_backtest)
