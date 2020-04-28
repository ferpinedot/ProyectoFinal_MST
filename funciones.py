# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:28:14 2020

@author: User
"""

# -- ------------------------------------------------------------------------------------ -- #
# -- proyecto: Microestructura y Sistemas de Trading - Laboratorio 4 - Proyecto Final
# -- archivo: funciones.py - donde se llevarán a cabo las funciones a utilizar
# -- mantiene: Fernanda Pinedo, Oscar Flores, Francisco Rodriguez
# -- repositorio: https://github.com/ferpinedot/proyecto_equipo5
# -- ------------------------------------------------------------------------------------ -- #


import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datos as dt
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf


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


    


