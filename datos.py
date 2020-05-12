# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:25:08 2020

@author: User
"""

# -- ------------------------------------------------------------------------------------ -- #
# -- proyecto: Microestructura y Sistemas de Trading - Laboratorio 4 - Proyecto Final
# -- archivo: datos.py - para jalar los datos
# -- mantiene: Fernanda Pinedo, Oscar Flores, Francisco Rodriguez
# -- repositorio: https://github.com/ferpinedot/proyecto_equipo5
# -- ------------------------------------------------------------------------------------ -- #

import funciones as fn

# Leer el archivo: Indicador económico USA
datos = fn.f_leer_archivo(param_archivo = 'archivos/FedInterestRateDecision-UnitedStates.xlsx', sheet_name= 0)
