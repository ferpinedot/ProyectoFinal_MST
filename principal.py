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

# BackTesting
# timestamp escenario operacion volumen resultado pips capital capital_acm
for (dato, clas) in zip(datos_instrumento.items(), clasificacion):
    TakeProfit = 20
    StopLoss =10
    pips_transaccion = 10000
    timestamp_cierre_operacion, gain_loss, pips_gl = Gain_Loss(dato[1], TakeProfit, StopLoss, pips_transaccion)
    print(gain_loss, pips_gl)


def Gain_Loss(dato, TakeProfit, StopLoss, pips_transaccion):
    TP = dato.TimeStamp[(dato.High > dato.Open[0]+TakeProfit/pips_transaccion)==True]
    SL = dato.TimeStamp[(dato.Low < dato.Open[0]-StopLoss/pips_transaccion)==True]
    try:
        if TP.iloc[0]: # Se cumple TakeProfit
            try:
                if TP.iloc[0] < SL.iloc[0]: # Si se cumple StopLoss lo compara para ver cual se cumple primero
                    return (TP.iloc[0], 'Gain', TakeProfit)
                return (SL.iloc[0], 'Loss', StopLoss)
            except:
                return(TP.iloc[0], 'Gain', TakeProfit)
    except:
        try:
            if SL.iloc[0]:
                return(SL.iloc[0], 'Loss', StopLoss)
        except:
            dif = dato.Close.iloc[-1] - dato.Open.iloc[0]
            if dif > 0:
                return(dato.TimeStamp.iloc[-1], 'Gain', dif*pips_transaccion)
            return(dato.TimeStamp.iloc[-1], 'Loss', -dif*pips_transaccion)

    #print((dato[1].iloc[:,1:] > dato[1].Open[0]+TakeProfit/pips_transaccion).sum(axis=1).sum() > 0)


dato[1].TimeStamp.iloc[-1]
dato[1].Close.iloc[-1] - dato[1].Open.iloc[0]
(dato[1].TimeStamp[-1], 'Loss', -dif*pips_transaccion)
SL.iloc[0]
TP.iloc[0]
TP.iloc[0] < SL.iloc[0]




plt.plot(dato[1].iloc[:,1:])
