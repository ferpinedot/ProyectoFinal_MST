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
    
    
# QQ plot de indicador para heteroscedasticidad
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    