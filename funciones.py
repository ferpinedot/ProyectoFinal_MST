# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:28:14 2020

@author: User
"""

# -- ------------------------------------------------------------------------------------ -- #
# -- proyecto: Microestructura y Sistemas de Trading - Laboratorio 4 - Proyecto Final
# -- archivo: funciones.py - donde se llevarán a cabo las funciones a utilizar
# -- mantiene: Fernanda Pinedo, Oscar Flores, Francisco Rodriguez
# -- repositorio: https://github.com/OscarFlores-IFi/proyecto_equipo5
# -- ------------------------------------------------------------------------------------ -- #


import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datos as dt
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import statsmodels.stats.diagnostic as smd
from statsmodels.formula.api import ols
import scipy.stats
from scipy.stats import shapiro
from scipy.stats import normaltest


#%% Parte 2: Datos Históricos
#%% Aspecto Matemático

# Lectura del archivo de datos en excel o csv

def f_leer_archivo(param_archivo, sheet_name = 0):
    """
    Parameters
    ----------
    param_archivo : str : nombre de archivo a leer
    Returns
    -------
    df_data : pd.DataFrame : con informacion contenida en archivo leido
    
    Debugging
    ---------
    param_archivo = 'FedInterestRateDecision-UnitedStates.xlsx'
    """

    #df_data = pd.read_csv(param_archivo)
    df_data = pd.read_excel(param_archivo, sheet_name = 0)
    df_data.columns = [i.lower() for i in list(df_data.columns)]
    return df_data


# Autocorrelación
def f_autocorr(datos):
    """
    Parameters
    ----------
    series : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Seleccionar la serie como el valor actual del indicador
    serie = datos['actual']
    autocorr = sm.stats.diagnostic.acorr_ljungbox(serie, lags=None, boxpierce=False)
    lbva = pd.DataFrame(autocorr[0])
    lbva.columns = ['Valor']
    pva = pd.DataFrame(autocorr[1])
    pva.columns = ['Valor']
    pva = pva.round(5)

    return {'Valor Test Estadístico' : lbva.copy(), 'P-value' : pva.copy()}
    

# Autocorrelación parcial
def f_autocorr_par(datos):
    """
    Parameters
    ----------
    datos : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    """
    serie = datos['actual']
    autocorr_par = sm.tsa.stattools.pacf(serie, method = 'yw', alpha = 0.01)
    facp = pd.DataFrame(autocorr_par[0])
    facp.columns = ['Valor']
    facp = facp.round(2)
    int_conf = pd.DataFrame(autocorr_par[1])
    int_conf.columns = ['Límite inferior', 'Límite superior']
    return {'Autocorrelaciones Parciales': facp.copy(), 'Intervalos de confianza': int_conf.copy()}


# Prueba de Normalidad
def f_norm_shw(datos):
    """
    Parameters
    ----------
    datos : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    serie = datos['actual']
    norm_shap_w = shapiro(serie)
    stat_shw = norm_shap_w[0]
    pvalue_shw = norm_shap_w[1]
    alpha = 0.05
    normal_shw = 'Si' if pvalue_shw > alpha else 'No'
    return {'Statistic Value': stat_shw, 'P-value': pvalue_shw, '¿Normal?': normal_shw}
    

def f_norm_dagp(datos):
    """
    Parameters
    ----------
    datos : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    serie = datos['actual']    
    norm_dagp = normaltest(serie)
    stat_dagp = norm_dagp[0]
    pvalue_dagp = norm_dagp[1]
    alpha = 0.05
    normal_dagp = 'Si' if pvalue_dagp > alpha else 'No'
    return {'Statistic Value': stat_dagp, 'P-value': pvalue_dagp, '¿Normal?': normal_dagp}


# Estacionariedad
def f_stationarity(datos):
    """
    Parameters
    ----------
    datos : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    serie = datos['actual']
    # resultado = adfuller(serie)
    # return {'Dicky Fuller Test Statistic': resultado[0], 'P-Value': resultado[1], 'Valores críticos': resultado[4]}
    
    stc = sm.tsa.stattools.adfuller(serie, maxlag = 2, regression = "ct", autolag = 'AIC', store = False, regresults = False)
    adf = stc[0]
    pv = stc[1]
    ul = stc[2]
    nob = stc[3]
    cval = stc[4]
    icmax = stc[5] 
    alpha = 0.05
    stty = 'No' if pv > alpha else 'Si'
    return {'Dicky Fuller Test Statistic': adf, 'P-Value': pv, 'Número de rezagos': ul, 'Número de observaciones': nob, 'Valores críticos': cval, 'Criterior de información maximizada': icmax, '¿Estacionaria?': stty}
   

