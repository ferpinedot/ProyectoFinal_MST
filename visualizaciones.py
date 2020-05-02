# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:29:33 2020

@author: User
"""

# -- ------------------------------------------------------------------------------------ -- #
# -- proyecto: Microestructura y Sistemas de Trading - Laboratorio 4 - Proyecto Final
# -- archivo: visualizaciones.py 
# -- mantiene: Fernanda Pinedo, Oscar Flores, Francisco Rodriguez
# -- repositorio: https://github.com/OscarFlores-IFi/proyecto_equipo5
# -- ------------------------------------------------------------------------------------ -- #

import funciones as fn
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL



# Datos a usar para gráficas
datos = fn.f_leer_archivo(param_archivo='archivos/FedInterestRateDecision-UnitedStates.xlsx', sheet_name= 0)

# Visualización del indicador origial con datos 'actual'
def v_indicador_orig(datos):
    """
    Parameters
    ----------
    datos : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    datos.plot(x = 'datetime', y = 'actual', kind = 'line', color = 'blue')
    plt.xticks(rotation=45)
    plt.title('Valor del Indicador')
    plt.xlabel('tiempo')
    plt.ylabel('Valor Actual')
    plt.show()


# Visualización de autocorrelación para estacionalidad post-dif
def v_fac_estac(datos):
    """
    Parameters
    ----------
    datos : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    datos = datos.set_index('datetime')
    datos_dif = datos - datos.shift()
    datos_dif.dropna(inplace= True)
    serie = datos_dif['actual']
    serie = serie.reset_index(drop = True)
    plot_acf(serie)
    plt.show()
    

# Visualización de autocorrelación parcial para estacionalidad post-dif
def v_facp_estac(datos):
    """
    Parameters
    ----------
    datos : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    datos = datos.set_index('datetime')
    datos_dif = datos - datos.shift()
    datos_dif.dropna(inplace= True)
    serie = datos_dif['actual']
    serie = serie.reset_index(drop = True)
    plot_pacf(serie)
    plt.show()
    
    
# QQ plot de indicador para distribución normal
def v_norm_qq(datos):
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
    qqplot(serie, line = 's')
    plt.title('QQ plot de Indicador')
    plt.xlabel('Cuantiles teóricos')
    plt.ylabel('Cuantiles de muestra')
    plt.show()
    
    
# Histograma de datos para distribución normal
def v_norm_hist(datos):
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
    plt.hist(serie)
    plt.title('Histograma de valores del Indicador')
    plt.xlabel('Valor actual del Indicador')
    plt.ylabel('Número de observaciones')
    plt.show()    
    
# Visualización de autocorrelación
def v_fac(datos):
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
    plot_acf(serie)
    plt.show()
    

# Visualización de autocorrelación parcial
def v_facp(datos):
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
    plot_pacf(serie)
    plt.show()
    

 # Estacionalidad   
def v_preseasonality(datos):
    """
    Parameters
    ----------
    datos : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """       
    serie = datos.set_index('datetime')
    serie = serie['actual']
    cycle, trend = sm.tsa.filters.hpfilter(serie, 50)
    fig, ax = plt.subplots(3,1)
    ax[0].plot(serie)
    ax[0].set_title('Interest Rate')
    ax[1].plot(trend)
    ax[1].set_title('Trend')
    ax[2].plot(cycle)
    ax[2].set_title('Cycle')
    plt.show()
    
    
def v_seasonality(datos):
    """
    Parameters
    ----------
    datos : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    serie = datos.set_index('datetime')
    serie = serie['actual']
    serie = serie.resample('M').mean().ffill()    
    result = STL(serie).fit()
    charts = result.plot()
    plt.show()

# Detección de atípicos
def v_det_atip(datos):
    """
    Parameters
    ----------
    datos : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    datos.plot(x = 'datetime', y = 'actual', kind = 'scatter', color = 'blue')
    plt.xticks(rotation=45)
    plt.title('Valor del Indicador')
    plt.xlabel('tiempo')
    plt.ylabel('Valor Actual')
    plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    