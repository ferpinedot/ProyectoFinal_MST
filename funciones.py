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
import datos as dt
import entradas as et
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
#from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import het_breuschpagan 
import statsmodels.stats.diagnostic as smd
#from statsmodels.formula.api import OLS
import scipy.stats
from scipy.stats import shapiro
from scipy.stats import normaltest
from statsmodels.sandbox.stats.diagnostic import acorr_lm
import statsmodels.sandbox.stats.diagnostic as ssd
from statsmodels.sandbox.stats.diagnostic import het_arch
from scipy import stats

from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#%% Parte 2: Datos Históricos
#%% Aspecto Matemático/Estadístico

# Lectura del archivo de datos en excel o csv
def f_leer_archivo(param_archivo, sheet_name = 0):
    """
    Función para leer el archivo de excel
    
    Parameters
    ----------
    param_archivo : str : nombre de archivo a leer
    
    Returns
    -------
    df_data : pd.DataFrame : con información contenida en archivo leido
    
    Debugging
    ---------
    param_archivo = 'FedInterestRateDecision-UnitedStates.xlsx'
    """

    #df_data = pd.read_csv(param_archivo)
    df_data = pd.read_excel(param_archivo, sheet_name = 0)
    # Volver nombre de columnas a minúsculas
    df_data.columns = [i.lower() for i in list(df_data.columns)]
    return df_data


# Estacionariedad
def f_stationarity(datos):
    """
    Prueba Dickey-Fuller de estacionariedad
    
    Parameters
    ----------
    datos : pd.DataFrame : con información contenida en archivo leido

    Returns
    -------
    dict : Valores del test de estacionariedad, p-value, número de rezagos, número de observaciones, valores críticos, criterio de información maxmizada, y si es estacionaria o no

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
    stty = 'No' if pv > et.alpha else 'Si'
    return {'Dickey Fuller Test Statistic': adf, 'P-Value': pv, 'Número de rezagos': ul, 'Número de observaciones': nob, 'Valores críticos': cval, 'Criterio de información maximizada': icmax, '¿Estacionaria?': stty}


# Diferenciación a la serie de tiempo
def f_dif_stationary(datos):
    """
    En caso de que la serie resulte No Estacionaria se le puede aplicar una 
    diferenciación para volverla estacionaria
    
    Parameters
    ----------
    datos : pd.DataFrame : con información contenida en archivo leido

    Returns
    -------
    dict : Valores del test de estacionariedad, p-value, número de rezagos, número de observaciones, valores críticos, criterio de información maxmizada, y si es estacionaria o no

    """
    datos = datos.set_index('datetime')
    datos_dif = datos - datos.shift()
    datos_dif.dropna(inplace= True)
    stc_dif = sm.tsa.stattools.adfuller(datos_dif['actual'], maxlag = 2, regression = "ct", autolag = 'AIC', store = False, regresults = False)
    adf = stc_dif[0]
    pv = stc_dif[1]
    ul = stc_dif[2]
    nob = stc_dif[3]
    cval = stc_dif[4]
    icmax = stc_dif[5] 
    stty = 'No' if pv > et.alpha else 'Si'
    return {'Dickey Fuller Test Statistic': adf, 'P-Value': pv, 'Número de rezagos': ul, 'Número de observaciones': nob, 'Valores críticos': cval, 'Criterio de información maximizada': icmax, '¿Estacionaria?': stty}
   

# #  Transformación logarítmica y diferenciación
# def f_diff_stationarity(datos):
#     """
#     En el caso de que la serie de tiempo no sea Estacionaria y requiera una 
#     transformación logarítmica y una diferenciación.
    
#     Parameters
#     ----------
#     datos : pd.DataFrame : con información contenida en archivo leido

#     Returns
#     -------
#     None.

#     """
#     datos = datos.set_index('datetime')
#     # Transformación logarítmica al df
#     dats_log = np.log(datos)
#     dats_log_dif = dats_log - dats_log.shift()
#     #plt.plot(dats_log_dif)

#     dats_log_dif.dropna(inplace= True)
#     stc_diff = sm.tsa.stattools.adfuller(dats_log_dif['actual'], maxlag = 2, regression = "ct", autolag = 'AIC', store = False, regresults = False)
#     adf = stc_diff[0]
#     pv = stc_diff[1]
#     ul = stc_diff[2]
#     nob = stc_diff[3]
#     cval = stc_diff[4]
#     icmax = stc_diff[5] 
#     stty = 'No' if pv > et.alpha else 'Si'
#     return {'Dicky Fuller Test Statistic': adf, 'P-Value': pv, 'Número de rezagos': ul, 'Número de observaciones': nob, 'Valores críticos': cval, 'Criterior de información maximizada': icmax, '¿Estacionaria?': stty}


# Autocorrelación Ljunj-Box de datos post-diferencia
def f_autocorr_lb(datos):
    """
    Test de autocorrelación de Ljung-Box
    
    Parameters
    ----------
    datos : pd.DataFrame : con información contenida en archivo leido

    Returns
    -------
    dict : Valor del test estadístico de Ljung-Box y P-value del mismo
    
    """
    datos = datos.set_index('datetime')
    datos_dif = datos - datos.shift()
    datos_dif.dropna(inplace= True)
    serie = datos_dif['actual']
    autocorr = sm.stats.diagnostic.acorr_ljungbox(serie, lags=None, boxpierce=False)
    lbva = pd.DataFrame(autocorr[0])
    lbva.columns = ['Valor']
    pva = pd.DataFrame(autocorr[1])
    pva.columns = ['Valor']
    #pva = pva.round
    return {'Valor Test Estadístico' : lbva, 'P-value' : pva.copy()}


# Autocorrelación Multiplicadores de Lagrange de datos post-diferencia 
def f_autocorr_lm(datos):
    """
    Test de autocorrelación con los Multiplicadores de Lagrange
    
    Parameters
    ----------
    datos : pd.DataFrame : con información contenida en archivo leido

    Returns
    -------
    dict : Valor test de Multiplicadores de Lagrange, P-value del mismo

    """
    datos = datos.set_index('datetime')
    datos_dif = datos - datos.shift()
    datos_dif.dropna(inplace= True)
    serie = datos_dif['actual']
    acf_lm = ssd.acorr_lm(serie, autolag = 'aic', store = False)
    lm = acf_lm[0]
    pva_lm = acf_lm[1]
    #pva_lm = pva_lm.round(5)
    fval = acf_lm[2]
    pva_f = acf_lm[3]
    #pva_f = pva_f.round(5)
    autocorr_lm = 'Si' if pva_lm <= et.alpha else 'No'
    return {'Lagrange Multiplier Value': lm, 'LM P-Value': pva_lm, 'F-Statistic Value': fval, 'F-Statistic P-Value': pva_f, '¿Autocorrelación?': autocorr_lm}


# Autocorrelación Parcial con datos del Indicador post-diferencia
def f_autocorr_par(datos):
    """
    Test de autocorrelación parcial
    
    Parameters
    ----------
    datos : pd.DataFrame : con información contenida en archivo leido

    Returns
    -------
    dict : Autocorrelaciones parciales e intervalos de confianza
    """
    datos = datos.set_index('datetime')
    datos_dif = datos - datos.shift()
    datos_dif.dropna(inplace= True)
    serie = datos_dif['actual']
    autocorr_par = sm.tsa.stattools.pacf(serie, method = 'yw', alpha = et.alpha)
    facp = pd.DataFrame(autocorr_par[0])
    facp.columns = ['Valor']
    facp = facp.round(2)
    int_conf = pd.DataFrame(autocorr_par[1])
    int_conf.columns = ['Límite inferior', 'Límite superior']
    # autocorr_p = pd.merge(facp, int_conf, left_index = True, right_index = True)
    return {'Autocorrelaciones Parciales': facp.copy(), 'Intervalos de confianza': int_conf.copy()}
    

# Heterocedasticidad de Indicador post-dif
def f_heter_bp(datos):
    """
    Prueba Breusch-Pagan de heterocedasticidad 
    
    Parameters
    ----------
    datos : pd.DataFrame : con información contenida en archivo leido

    Returns
    -------
    dict : Valores de heterocedasticidad y p-value

    """
    datos = datos.set_index('datetime')
    datos_dif = datos - datos.shift()
    datos_dif.dropna(inplace= True)
    datos_dif = datos_dif.reset_index() 
    serie = datos_dif['actual']
    indxx = datos_dif.index
    het_model = sm.OLS(serie, sm.add_constant(indxx)).fit()
    # heter = het_model.params    
    resids = het_model.resid
    het = smd.het_breuschpagan(resids, het_model.model.exog)
    lm_stat = het[0]
    pvalue_lm = het[1]
    f_stat = het[2]
    pvalue_f = het[3]
    heteroscedastico = 'Si' if pvalue_f < et.alpha else 'No'
    return {'Lagrange Multiplier Value': lm_stat, 'LM P-value': pvalue_lm, 'Statistic Value': f_stat, 'F-Statistic P-Value': pvalue_f, '¿Heteroscedástico?': heteroscedastico}


def f_heter_w(datos):
    """
    Prueba White de heterocedasticidad
    
    Parameters
    ----------
    datos : pd.DataFrame : con información contenida en archivo leido

    Returns
    -------
    dict : Valores de heterocedasticidad y p-value

    """
    datos = datos.set_index('datetime')
    datos_dif = datos - datos.shift()
    datos_dif.dropna(inplace= True)
    datos_dif = datos_dif.reset_index() 
    serie = datos_dif['actual']
    indxx = datos_dif.index
    het_model_w = sm.OLS(serie, sm.add_constant(indxx)).fit()
    resids = het_model_w.resid
    white_het = smd.het_white(resids, het_model_w.model.exog)
    lm_stat_w = white_het[0]
    pvalue_lm_w = white_het[1]
    f_stat_w = white_het[2]
    pvalue_f_w = white_het[3]
    heteroscedastico_w = 'Si' if pvalue_f_w < et.alpha else 'No'
    return {'Lagrange Multiplier Value': lm_stat_w, 'LM P-value': pvalue_lm_w, 'Statistic Value': f_stat_w, 'F-Statistic P-Value': pvalue_f_w, '¿Heteroscedástico?': heteroscedastico_w}


def f_het_arch(datos):
    """
    Prueba ARCH de heterocedasticidad, apropiada para los datos de una serie de tiempo
    
    Parameters
    ----------
    datos : pd.DataFrame : con información contenida en archivo leido

    Returns
    -------
    dict : Valores de heterocedasticidad y p-value

    """
    datos = datos.set_index('datetime')
    datos_dif = datos - datos.shift()
    datos_dif.dropna(inplace= True)
    datos_dif = datos_dif.reset_index() 
    serie = datos_dif['actual']
    het_ach = ssd.het_arch(serie, nlags = 20, store = False)
    lm_stat_arch = het_ach[0]
    pvalue_lm_arch = het_ach[1]
    f_stat_arch = het_ach[2]
    pvalue_f_arch = het_ach[3]
    heteroscedastico_arch = 'Si' if pvalue_f_arch < et.alpha else 'No'
    return {'Lagrange Multiplier Value': lm_stat_arch, 'LM P-value': pvalue_lm_arch, 'Statistic Value': f_stat_arch, 'F-Statistic P-Value': pvalue_f_arch, '¿Heteroscedástico?': heteroscedastico_arch}
    

# Prueba de Normalidad
def f_norm_shw(datos):
    """
    Prueba Shapiro-Wilk de normalidad 
    
    Parameters
    ----------
    datos : pd.DataFrame : con información contenida en archivo leido

    Returns
    -------
    dict : Valores de la prueba de normalidad y p-value

    """
    datos = datos.set_index('datetime')
    datos_dif = datos - datos.shift()
    datos_dif.dropna(inplace= True)
    datos_dif = datos_dif.reset_index() 
    serie = datos_dif['actual']
    norm_shap_w = shapiro(serie)
    stat_shw = norm_shap_w[0]
    pvalue_shw = norm_shap_w[1]
    normal_shw = 'Si' if pvalue_shw > et.alpha else 'No'
    return {'Statistic Value': stat_shw, 'P-value': pvalue_shw, '¿Normal?': normal_shw}
    

def f_norm_dagp(datos):
    """
    Prueba D'Agostino de normalidad
    
    Parameters
    ----------
    datos : pd.DataFrame : con información contenida en archivo leido

    Returns
    -------
    dict : Valores de prueba de normalidad y p-value

    """
    datos = datos.set_index('datetime')
    datos_dif = datos - datos.shift()
    datos_dif.dropna(inplace= True)
    datos_dif = datos_dif.reset_index() 
    serie = datos_dif['actual']  
    norm_dagp = normaltest(serie)
    stat_dagp = norm_dagp[0]
    pvalue_dagp = norm_dagp[1]
    normal_dagp = 'Si' if pvalue_dagp > et.alpha else 'No'
    return {'Statistic Value': stat_dagp, 'P-value': pvalue_dagp, '¿Normal?': normal_dagp}


# Sesgo
def f_skewness(datos):
    """
    Función para saber el sesgo de la distribución de datos en caso de no tener una distribución normal
    
    Parameters
    ----------
    datos : pd.DataFrame : con información contenida en archivo leido

    Returns
    -------
    dict : Nivel de sesgo, tipo de asimetría si hay

    """
    datos = datos.set_index('datetime')
    datos_dif = datos - datos.shift()
    datos_dif.dropna(inplace= True)
    datos_dif = datos_dif.reset_index() 
    serie = datos_dif['actual']
    skewness = stats.skew(serie)
    asimetria = 'Si' if skewness < -1 or skewness > 1 else 'No'
    tipo_simetria = 'Positiva' if skewness > 1 else 'Negativa' if skewness < -1 else 'Simétrico'
    return {'Nivel de sesgo': skewness, 'Asimétrico?': asimetria, 'Tipo de asimetría si hay': tipo_simetria}
    

