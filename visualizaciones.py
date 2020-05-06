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
#import seaborn as sb
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# Datos a usar para gráficas
datos = fn.f_leer_archivo(param_archivo='archivos/FedInterestRateDecision-UnitedStates.xlsx', sheet_name= 0)

# Visualización del indicador origial con datos 'actual'
def v_indicador_orig(datos):
    """
    Visualización de los datos actuales del indicador
    
    Parameters
    ----------
    datos : pd.DataFrame : con información contenida en archivo leido

    Returns
    -------
    Gráfica de línea de los valores (%) de las tasas de interés del indicador

    """
    datos.plot(x = 'datetime', y = 'actual', kind = 'line', color = 'blue')
    plt.xticks(rotation=45)
    plt.title('Valor del Indicador')
    plt.xlabel('tiempo')
    plt.ylabel('Valor Actual')
    plt.show()


#Visualización del indicador original con media móvil y desviación estándar móvil
def v_rmrdsv(datos):
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
    serie = datos['actual']
    rolling_mean = serie.rolling(window = 2).mean()
    rolling_std = serie.rolling(window = 2).std()
    plt.plot(serie, color = 'blue', label = 'Valor actual Indicador')
    plt.plot(rolling_mean, color = 'red', label = 'Media Móvil')
    plt.plot(rolling_std, color = 'black', label = 'Desviación Estándar Móvil')
    plt.legend(loc = 'best')
    plt.title('Indicador con media y desviación estándar móviles')
    plt.show()


# Visualización de serie del Indicador ya diferenciado
def v_indicador_dif(datos):
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
    plt.plot(serie, color = 'blue')
    plt.title('Indicador con una diferencia')
    plt.show()
   
    
# Visualización de Indicador ya diferenciado con media y varianza móviles
def v_rmrdsv_dif(datos):
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
    rolling_mean = serie.rolling(window = 2).mean()
    rolling_std = serie.rolling(window = 2).std()
    plt.plot(serie, color = 'blue', label = 'Valor actual Indicador')
    plt.plot(rolling_mean, color = 'red', label = 'Media Móvil')
    plt.plot(rolling_std, color = 'black', label = 'Desviación Estándar Móvil')
    plt.legend(loc = 'best')
    plt.title('Indicador con una diferencia con media y desviación estándar móviles')
    plt.show()
    
    
# Visualización de autocorrelación para estacionariedad del Indicador post-dif
def v_fac_estac(datos):
    """
    Visualización de la prueba de autocorrelación para la estacionariedad después de sacarle una diferencia a los datos
    
    Parameters
    ----------
    datos : pd.DataFrame : con información contenida en archivo leido

    Returns
    -------
    Gráfica de autocorrelación para datos con una diferenciación

    """
    datos = datos.set_index('datetime')
    datos_dif = datos - datos.shift()
    datos_dif.dropna(inplace= True)
    serie = datos_dif['actual']
    serie = serie.reset_index(drop = True)
    plot_acf(serie)
    plt.show()
    

# Visualización de autocorrelación parcial para estacionariedad de Indicador post-dif
def v_facp_estac(datos):
    """
    Visualización de la prueba de autocorrelación parcial para la estacionariedad después de sacarle una diferencia a los datos
    
    Parameters
    ----------
    datos : pd.DataFrame : con información contenida en archivo leido

    Returns
    -------
    Gráfica de autocorrelación parcial para datos con una diferenciación

    """
    datos = datos.set_index('datetime')
    datos_dif = datos - datos.shift()
    datos_dif.dropna(inplace= True)
    serie = datos_dif['actual']
    serie = serie.reset_index(drop = True)
    plot_pacf(serie)
    plt.show()


# Estacionalidad   
def v_preseasonality(datos):
    """
    Visualización de la pre-prueba de estacionalidad por medio de gráficas de los datos
    
    Parameters
    ----------
    datos : pd.DataFrame : con información contenida en archivo leido

    Returns
    -------
    Tres gráficas en una imagen que reflectan parte de la prueba de estacionalidad de los datos

    """       
    datos = datos.set_index('datetime')
    datos_dif = datos - datos.shift()
    datos_dif.dropna(inplace= True)
    serie = datos_dif['actual']
    cycle, trend = sm.tsa.filters.hpfilter(serie, 50)
    fig, ax = plt.subplots(3,1)
    ax[0].plot(serie)
    ax[0].set_title('Interest Rate actual post-dif')
    ax[1].plot(trend)
    ax[1].set_title('Trend')
    ax[2].plot(cycle)
    ax[2].set_title('Cycle')
    plt.show()
    
    
def v_seasonality(datos):
    """
    Visualización de la prueba de estacionalidad por medio de gráficas de los datos
    
    Parameters
    ----------
    datos : pd.DataFrame : con información contenida en archivo leido

    Returns
    -------
    Cuatro gráficas en una imagen que reflejan la prueba de estacionalidad de los datos 

    """
    datos = datos.set_index('datetime')
    datos_dif = datos - datos.shift()
    datos_dif.dropna(inplace= True)
    serie = datos_dif['actual']
    serie = serie.resample('M').mean().ffill()    
    result = STL(serie).fit()
    charts = result.plot()
    plt.show()


# Histograma de datos para distribución normal
def v_norm_hist(datos):
    """
    Visualización de la prueba de normalidad para saber si es distribución gaussiana o no
    
    Parameters
    ----------
    datos : pd.DataFrame : con información contenida en archivo leido

    Returns
    -------
    Gráfica tipo histograma de los datos 

    """
    datos = datos.set_index('datetime')
    datos_dif = datos - datos.shift()
    datos_dif.dropna(inplace= True)
    serie = datos_dif['actual']
    plt.hist(serie)
    plt.title('Histograma de valores del Indicador con diferencia')
    plt.xlabel('Valor actual del Indicador')
    plt.ylabel('Número de observaciones')
    plt.show()  
    

# QQ plot de indicador para distribución normal
def v_norm_qq(datos):
    """
    Visualización de la prueba de normalidad por medio de gráficas
    
    Parameters
    ----------
    datos : pd.DataFrame : con información contenida en archivo leido

    Returns
    -------
    Gráfica cuantil-cuantil de los datos

    """
    datos = datos.set_index('datetime')
    datos_dif = datos - datos.shift()
    datos_dif.dropna(inplace= True)
    serie = datos_dif['actual']
    qqplot(serie, line = 's')
    plt.title('QQ plot de Indicador con diferencia')
    plt.xlabel('Cuantiles teóricos')
    plt.ylabel('Cuantiles de muestra')
    plt.show()
    
    
# Detección de atípicos boxplot
def v_det_at(datos):
    """
    Visualización de los datos atípicos por medio de un boxplot 
    
    Parameters
    ----------
    datos : pd.DataFrame : con información contenida en archivo leido

    Returns
    -------
    Gráfica tipo bigote caja (boxplot) para identificar los datos atípicos de los datos actuales del indicador

    """
    datos = datos.set_index('datetime')
    datos_dif = datos - datos.shift()
    datos_dif.dropna(inplace= True)
    serie = datos_dif['actual']
    dat_at = sb.boxplot(data = serie, orient = 'v')
    dat_at.set_title('Datos atípicos del Indicador')
    dat_at.set_ylabel('Valor actual')
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    