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



def f_precios_masivos(p0_fini, p1_ffin, p2_gran, p3_inst, p4_oatk, p5_ginc):
    """
    Parameters
    ----------
    p0_fini
    p1_ffin
    p2_gran
    p3_inst
    p4_oatk
    p5_ginc

    Returns
    -------
    dc_precios

    Debugging
    ---------

    """

    def f_datetime_range_fx(p0_start, p1_end, p2_inc, p3_delta):
        """

        Parameters
        ----------
        p0_start
        p1_end
        p2_inc
        p3_delta

        Returns
        -------
        ls_resultado

        Debugging
        ---------
        """

        ls_result = []
        nxt = p0_start

        while nxt <= p1_end:
            ls_result.append(nxt)
            if p3_delta == 'minutes':
                nxt += timedelta(minutes=p2_inc)
            elif p3_delta == 'hours':
                nxt += timedelta(hours=p2_inc)
            elif p3_delta == 'days':
                nxt += timedelta(days=p2_inc)

        return ls_result

    # inicializar api de OANDA

    api = API(access_token=p4_oatk)

    gn = {'S30': 30, 'S10': 10, 'S5': 5, 'M1': 60, 'M5': 60 * 5, 'M15': 60 * 15,
          'M30': 60 * 30, 'H1': 60 * 60, 'H4': 60 * 60 * 4, 'H8': 60 * 60 * 8,
          'D': 60 * 60 * 24, 'W': 60 * 60 * 24 * 7, 'M': 60 * 60 * 24 * 7 * 4}

    # -- para el caso donde con 1 peticion se cubran las 2 fechas
    if int((p1_ffin - p0_fini).total_seconds() / gn[p2_gran]) < 4999:

        # Fecha inicial y fecha final
        f1 = p0_fini.strftime('%Y-%m-%dT%H:%M:%S')
        f2 = p1_ffin.strftime('%Y-%m-%dT%H:%M:%S')
        print(f1,f2)
        # Parametros pra la peticion de precios
#        params = {"granularity": p2_gran, "price": "M", "dailyAlignment": 16, "from": f1,
#                  "to": f2}
        params = {"granularity": p2_gran, "price": "M", "from": f1, "to": f2}

        # Ejecutar la peticion de precios
        a1_req1 = instruments.InstrumentsCandles(instrument=p3_inst, params=params)
        a1_hist = api.request(a1_req1)
        print(a1_hist)
        # Para debuging
        # print(f1 + ' y ' + f2)
        lista = list()

        # Acomodar las llaves
        for i in range(len(a1_hist['candles']) - 1):
            lista.append({'TimeStamp': a1_hist['candles'][i]['time'],
                          'Open': a1_hist['candles'][i]['mid']['o'],
                          'High': a1_hist['candles'][i]['mid']['h'],
                          'Low': a1_hist['candles'][i]['mid']['l'],
                          'Close': a1_hist['candles'][i]['mid']['c']})

        # Acomodar en un data frame
        r_df_final = pd.DataFrame(lista)
        r_df_final = r_df_final[['TimeStamp', 'Open', 'High', 'Low', 'Close']]
        r_df_final['TimeStamp'] = pd.to_datetime(r_df_final['TimeStamp'])
        r_df_final['Open'] = pd.to_numeric(r_df_final['Open'], errors='coerce')
        r_df_final['High'] = pd.to_numeric(r_df_final['High'], errors='coerce')
        r_df_final['Low'] = pd.to_numeric(r_df_final['Low'], errors='coerce')
        r_df_final['Close'] = pd.to_numeric(r_df_final['Close'], errors='coerce')

        return r_df_final

    # -- para el caso donde se construyen fechas secuenciales
    else:

        # hacer series de fechas e iteraciones para pedir todos los precios
        fechas = f_datetime_range_fx(p0_start=p0_fini, p1_end=p1_ffin, p2_inc=p5_ginc,
                                     p3_delta='minutes')

        # Lista para ir guardando los data frames
        lista_df = list()

        for n_fecha in range(0, len(fechas) - 1):

            # Fecha inicial y fecha final
            f1 = fechas[n_fecha].strftime('%Y-%m-%dT%H:%M:%S')
            f2 = fechas[n_fecha + 1].strftime('%Y-%m-%dT%H:%M:%S')

            # Parametros pra la peticion de precios
            params = {"granularity": p2_gran, "price": "M", "dailyAlignment": 16, "from": f1,
                      "to": f2}

            # Ejecutar la peticion de precios
            a1_req1 = instruments.InstrumentsCandles(instrument=p3_inst, params=params)
            a1_hist = api.request(a1_req1)

            # Para debuging
            print(f1 + ' y ' + f2)
            lista = list()

            # Acomodar las llaves
            for i in range(len(a1_hist['candles']) - 1):
                lista.append({'TimeStamp': a1_hist['candles'][i]['time'],
                              'Open': a1_hist['candles'][i]['mid']['o'],
                              'High': a1_hist['candles'][i]['mid']['h'],
                              'Low': a1_hist['candles'][i]['mid']['l'],
                              'Close': a1_hist['candles'][i]['mid']['c']})

            # Acomodar en un data frame
            pd_hist = pd.DataFrame(lista)
            pd_hist = pd_hist[['TimeStamp', 'Open', 'High', 'Low', 'Close']]
            pd_hist['TimeStamp'] = pd.to_datetime(pd_hist['TimeStamp'])

            # Ir guardando resultados en una lista
            lista_df.append(pd_hist)

        # Concatenar todas las listas
        r_df_final = pd.concat([lista_df[i] for i in range(0, len(lista_df))])

        # resetear index en dataframe resultante porque guarda los indices del dataframe pasado
        r_df_final = r_df_final.reset_index(drop=True)
        r_df_final['Open'] = pd.to_numeric(r_df_final['Open'], errors='coerce')
        r_df_final['High'] = pd.to_numeric(r_df_final['High'], errors='coerce')
        r_df_final['Low'] = pd.to_numeric(r_df_final['Low'], errors='coerce')
        r_df_final['Close'] = pd.to_numeric(r_df_final['Close'], errors='coerce')

        return r_df_final




#%% Parte 2: Datos Históricos
#%% Aspecto Matemático

# Lectura del archivo de datos en excel o csv
def f_leer_archivo(param_archivo, sheet_name = 0, index_col = 0):
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
    df_data = pd.read_excel(param_archivo, sheet_name)
    df_data.columns = [i.lower() for i in list(df_data.columns)]
    print(df_data)
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


# Clasificación de la ocurrencia para: previous, consensus, actual.
def f_clasificacion_ocurrencia(datos):
    """
    Parameters
    ----------
    datos : pd.DataFrame : columnas de preicio actual, consensus y previous del indicador económico seleccionado.

    Returns
    -------
    None.

    Debug
    -----
    datos = f_leer_archivo(param_archivo='archivos/FedInterestRateDecision-UnitedStates.xlsx',sheet_name = 0)
    """
    ac = datos.actual >= datos.consensus
    #ap = datos.actual > datos.previous # no utilizado. en ningun momento se compara si el actual es mayor o menor al previous
    cp = datos.consensus >= datos.previous

    def clasificacion(ac, cp):
        if ac:
            if cp:
                return 'A'
            else:
                return 'B'
        elif cp:
            return 'C'
        else:
            return 'D'

    print([clasificacion(i,j) for (i,j) in zip(ac,cp)])
