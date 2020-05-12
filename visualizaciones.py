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
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
import seaborn as sb
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from scipy import stats
from scipy.stats import normaltest


#%%
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
def v_rmrdsv_original(datos):
    """
    Parameters
    ----------
    datos : pd.DataFrame : con información contenida en archivo leido

    Returns
    -------
    Gráfica de lpinea de los valores del indicador original junto con su media 
    y desviación estándar móviles

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
    datos : pd.DataFrame : con información contenida en archivo leido

    Returns
    -------
    Gráfica del indicador ya estacionario, con una diferencia

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
    datos : pd.DataFrame : con información contenida en archivo leido

    Returns
    -------
    Gráfica del indicador ya estacionario, con una diferencia y con media y
    desviación estándar móviles

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


# Visualizaicón de un modelo ARIMA(3,1,3) 
def v_model_arima_orig(datos):
    """
    Parameters
    ----------
    datos : pd.DataFrame : con información contenida en archivo leido

    Returns
    -------
    Gráfica del indicador ya estacionario y el modelo ARIMA(3,1,3) ajustado al
    comportamiento de ese indicador
    
    """    
    datos = fn.f_leer_archivo(param_archivo='archivos/FedInterestRateDecision-UnitedStates.xlsx', sheet_name= 0)
    datos = datos.set_index('datetime')
    datos_dif = datos - datos.shift()
    datos_dif.dropna(inplace= True)
    datos_dif_ac = datos_dif['actual']
    model = ARIMA(datos_dif_ac, order = (3,1,3))
    results = model.fit(disp=-1)    
    predictions_ARIMA_dif = pd.Series(results.fittedvalues, copy=True)
    predictions_ARIMA_dif_cumsum = predictions_ARIMA_dif.cumsum()
    plt.plot(datos_dif_ac)
    plt.plot(predictions_ARIMA_dif_cumsum)
    plt.title('ARIMA(3,1,3)')
    plt.show()
    
    
# Visualización de estacionalidad
def v_seasonality(datos):
    """
    Visualización de la prueba de estacionalidad por medio de gráficas de los 
    datos ya estacionarios
    
    Parameters
    ----------
    datos : pd.DataFrame : con información contenida en archivo leido

    Returns
    -------
    Cuatro gráficas en una imagen que reflejan la prueba de estacionalidad de 
    los datos 

    """
    datos = fn.f_leer_archivo(param_archivo='archivos/FedInterestRateDecision-UnitedStates.xlsx', sheet_name= 0)
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
    Visualización de la prueba de normalidad para saber si es distribución 
    gaussiana o no
    
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
     
    
# Visualización de normalidad con residuales   
def v_norm_resids(datos):
    """
    Visualización de la prueba de normalidad con residuales del modelo ARIMA
    
    Parameters
    ----------
    datos : pd.DataFrame : con información contenida en archivo leido

    Returns
    -------
    Gráfica tipo histograma de los residuales del modelo ARIMA

    """
    datos = fn.f_leer_archivo(param_archivo='archivos/FedInterestRateDecision-UnitedStates.xlsx', sheet_name= 0)
    datos = datos.set_index('datetime')
    datos_dif = datos - datos.shift()
    datos_dif.dropna(inplace= True)
    datos_dif_ac = datos_dif['actual']
    model = sm.tsa.ARIMA(datos_dif_ac, order = (3,1,3)).fit(disp=False)
    resid = model.resid
    fig = plt.figure(figsize = (12,8))
    ax0 = fig.add_subplot(111)
    sb.distplot(resid, fit = stats.norm, ax = ax0)
    (mu, sigma) = stats.norm.fit(resid)
    plt.legend(['Normal dist. ($\mu=$ {:.2f} y $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de residuales')
    fig = plt.figure(figsize=(12,8))
            

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
    serie = datos['actual']
    dat_at = sb.boxplot(data = serie, orient = 'v')
    dat_at.set_title('Datos atípicos del Indicador')
    dat_at.set_ylabel('Valor actual')
    plt.show()


# Visualización de atípicos de Indicador diferenciado por boxplot
def v_det_at_dif(datos):
    """
    Visualización de los datos atípicos del indicador estacionario por medio de un boxplot 
    
    Parameters
    ----------
    datos : pd.DataFrame : con información contenida en archivo leido

    Returns
    -------
    Gráfica tipo bigote caja (boxplot) para identificar los datos atípicos de 
    los datos actuales del indicador ya estacionario

    """
    datos = datos.set_index('datetime')
    datos_dif = datos - datos.shift()
    datos_dif.dropna(inplace= True)
    serie = datos_dif['actual']
    dat_at = sb.boxplot(data = serie, orient = 'v')
    dat_at.set_title('Datos atípicos del Indicador')
    dat_at.set_ylabel('Valor actual')
    plt.show() 
    
