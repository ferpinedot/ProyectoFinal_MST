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
# from pyeasyga import pyeasyga
import numpy as np


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

# Identificación del modelo ARIMA
vn.v_model_arima_orig(datos)

# Estacionalidad de Indicador 
vn.v_seasonality(datos)

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

# Detección de atípicos
vn.v_det_at(datos)
vn.v_det_at_dif(datos)

#%%
########################################################################################################################
########################################################################################################################

from pyeasyga import pyeasyga

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

df_escenarios[df_escenarios.escenario == 'A']


#Supongamos la siguiente estrategia dependiendo de cada escenario.
df_decisiones = pd.DataFrame(data = [['compra',10,25, 10000],['compra', 10, 25, 10000],['compra',10,25, 10000],['venta', 10, 30, 100000]],
                            index = ['A', 'B', 'C', 'D'],
                            columns = ['operacion', 'StopLoss', 'TakeProfit', 'Volume'])


# BackTesting
# timestamp escenario operacion volumen resultado pips capital capital_acm
df_backtest = fn.f_df_backtest(datos_instrumento, clasificacion, df_decisiones)

# Optimización de ratio de Sharpe usando algorítmo genético de librería pyeasyga.
data = [datos_instrumento, clasificacion]
ga = pyeasyga.GeneticAlgorithm(data, population_size=10,
                               generations=20,
                               crossover_probability=0.8,
                               mutation_probability=0.05,
                               elitism=True,
                               maximise_fitness=True)

def create_individual(data):
    #uniques = pd.unique(data[1])
    uniques = ['A', 'B']
    individuo = [[np.random.randint(0, 2), np.random.randint(1, 1000),
                  np.random.randint(1, 1000), np.random.randint(1, 1000)]
                  for _ in uniques]
    return individuo
ga.create_individual = create_individual

def fitness(individual, data):
    decisiones = pd.DataFrame(data= individual,
                              index = pd.unique(data[1]),
                              columns = ['operacion', 'StopLoss', 'TakeProfit', 'Volume'])
    decisiones['operacion'][decisiones['operacion'] == 0] = 'venta'
    decisiones['operacion'][decisiones['operacion'] == 1] = 'compra'
    print(decisiones)

    datos_instrumento = data[0]
    clasificacion = data[1]

    df_backtest = fn.f_df_backtest(datos_instrumento, clasificacion, decisiones)
    mean = df_backtest['capital acumulado'].mean()
    std = df_backtest['capital acumulado'].std()
    print((mean - 0.003)/std)
    return (mean - 0.003)/std # 0.003 is the risk free rate for every 1.5 months.
ga.fitness_function = fitness

ga.run()

print(ga.best_individual())