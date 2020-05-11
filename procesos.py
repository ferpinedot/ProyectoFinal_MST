class Genetico:
    # -- ------------------------------------------------------------------------------------ -- #
    # -- proyecto: Microestructura y Sistemas de Trading - Laboratorio 4 - Proyecto Final
    # -- archivo: procesos.py 
    # -- mantiene: Fernanda Pinedo, Oscar Flores, Francisco Rodriguez
    # -- repositorio: https://github.com/OscarFlores-IFi/proyecto_equipo5
    # -- ------------------------------------------------------------------------------------ -- #


    def create_first_generation(data, n):
        import pandas as pd
        import numpy as np
        # data debería ser una lista, en la posición [0] contiene la descarga de datos de los instrumentos
        # en la posición [1] debería tener la clasificación de cada uno de los instrumentos.
        uniques = pd.unique(data[1])
        individuos = np.concatenate((np.random.randint(0,2,size=(n,1)),
                                np.random.randint(1,42,size=(n,1)),
                                np.random.randint(1,42,size=(n,1)),
                                np.random.randint(1,2380,size=(n,1))),axis=1)

        if len(uniques) > 1:
            for _ in range(len(uniques) - 1):
                individuos = np.concatenate((individuos,
                            np.concatenate((np.random.randint(0,2,size=(n,1)),
                                                    np.random.randint(1,42,size=(n,1)),
                                                    np.random.randint(1,42,size=(n,1)),
                                                    np.random.randint(1,2380,size=(n,1))),axis=1)),axis = 1)
        return individuos



    def fitness(individual, data, print_backtest = False):
        import pandas as pd
        import numpy as np

        import funciones as fn

        # individual será uno de los vectores de toma de df_decisiones
        # data es una lista con 2 elementos en el interior;
        # en la posición [0] contiene la descarga de datos de los instrumentos
        # en la posición [1] debería tener la clasificación de cada uno de los instrumentos
        uniques = pd.unique(data[1])
        indiv = [individual[i*4:i*4+4] for i in range(len(individual)//4)]
        decisiones = pd.DataFrame(data= indiv,
                                  index = pd.unique(data[1]),
                                  columns = ['operacion', 'StopLoss', 'TakeProfit', 'Volume'])
        decisiones['operacion'][decisiones['operacion'] == 0] = 'venta'
        decisiones['operacion'][decisiones['operacion'] == 1] = 'compra'
        #print(decisiones)

        datos_instrumento = data[0]
        clasificacion = data[1]

        df_backtest = fn.f_df_backtest(datos_instrumento, clasificacion, decisiones)
        if print_backtest:
            print(df_backtest)
        ###############################################
        # calcular rendimientos de capital acumulado

        rend = np.log(df_backtest['capital acumulado'][:-1].values/df_backtest['capital acumulado'][1:].values)
        ###############################################
        mean = rend.mean()
        std = rend.std()
        sharpe = (mean - 0.002)/std # 0.025 / 8.

        return (mean,std,sharpe) # 0.003 is the risk free rate for every 1.5 months.



    def genetico(data,iteraciones = 30, n_vec = 2**6, filename = ''):
        from time import time

        import pandas as pd
        import numpy as np
        import pickle

        import funciones as fn

        t1 = time()
        #func = función a optimizar, esta deberá dar los resultados del vector de decisiones en todas las empresas probadas.
        #C0 = Condicion inicial de los padres. (len(C0)<nvec)
        #csv = datos sobre los cuales se simulará
        #cetes = datos diarios de tasa de interés para el dinero no invertido en activos.
        #ndias = vector, dias en los que se hacen clusters de los datos para reconocimiento de patrones.
        #model_close = modelo de reconocimiento de patrones con la información de clusters entrenados.
        #l_vec = longitud de vector de toma de decisiones, en potencias de 2
        #n_vec = cantidad de vectores de toma de decisiones, en potencias de 2.
        #iteraciones = número de ciclos completos que dará el algorítmo genético.
        #C = multiplicador de castigo por desviación estándar
        #rf = tasa libre de riesgo para optimización con respecto al ratio de Sharpe.
        #nombre = texto, nombre que tendrá el archivo en dónde se guardarán los resultados del AG.
        # número de vectores de toma de decisiones (hijos), tiene que ser potencia de 2.

        decisiones = Genetico.create_first_generation(data, n_vec)

        l_vec = decisiones.shape[1] # longitud de cada uno de los vectores de decisiones.

        hist_mean = np.ones((iteraciones,n_vec//4*5)) # historial de media
        hist_std = np.zeros((iteraciones,n_vec//4*5)) # historial de desviación estandar
        hist_sharpe = np.zeros((iteraciones,n_vec//4*5)) # historial de calificaciones
        hist_padres = []

        punt = np.ones(n_vec//4*5)*-50 # puntuaciones de hijos y padres, se sobre-escribe en cada ciclo
        padres = np.zeros((n_vec//4,l_vec)) # padres, se sobre-escribe en cada ciclo


        for cic in range(iteraciones): # Generaciones del algorítmo.

            for i in np.arange(n_vec): ## se simulan todos vectores de decisión para escoger el que de la suma mayor
                [mu, stdev, sharpe] = Genetico.fitness(decisiones[i],data)
                #print(mu, stdev, sharpe)

                hist_mean[cic, i] = mu
                hist_std[cic, i] = stdev
                hist_sharpe[cic, i] = sharpe

            # Se toma la calificación de cada vector de toma de decisiones.
            punt[:n_vec] = hist_sharpe[cic,:n_vec] # Para maximizar con respecto a Sharpe
            #punt[:n_vec] = hist_mean[cic, :n_vec] # Para maximizar con respecto a rendimientos

            # Se escogen los padres.
            selectos = np.concatenate((decisiones,padres)) # agregamos los 'padres' de las nuevas generaciones a la lista.
            indx = np.argsort(punt)[-int(n_vec//4):] # Indice donde se encuentran los mejores padres
            padres = selectos[indx] # se escojen los padres
            punt[n_vec:] = punt[indx] # se guarda la puntuación de los padres.
            hist_mean[cic,n_vec:] = hist_mean[cic,:][indx]
            hist_std[cic,n_vec:] = hist_mean[cic,:][indx]
            hist_sharpe[cic,n_vec:] = hist_mean[cic,:][indx]
            #print(padres)
            #print(punt)
            hist_padres.append(padres)

            # Selección
            decisiones = np.array([[np.random.choice(padres.T[i]) for i in range(l_vec)] for i in range(n_vec)])

            # Mutación
            prob_mutacion = 0.2
            mutados = np.random.choice([0,1], p=[1-prob_mutacion, prob_mutacion], size=(n_vec, l_vec))
            for i in range(len(mutados)):
                for j in range(len(mutados[0])):
                    if mutados[i,j]:
                        if j % 4 == 0:
                            decisiones[i,j] = np.random.randint(0,2) # compra o venta: 0 o 1
                        elif j % 4 == 1:
                            decisiones[i,j] = np.random.randint(1,42) # intervalo de StopLoss entre 0 y 42
                        elif j % 4 == 2:
                            decisiones[i,j] = np.random.randint(1,42) # intervalo de TakeProfit entre 0 y 21
                        else:
                            decisiones[i,j] = np.random.randint(1,2380) # Volumen de operaciones. En el peor de los casos se pierde 1000 usd. con apalancamiento 100x
            # Para imprimir el proceso del algoritmo genérico en relación al total por simular y el tiempo de cada iteracion.
            print(punt)
        print(padres[-1])
        Genetico.fitness(padres[-1],data,True)

        print('tiempo de ejecución en seg.:')
        print(time()-t1)

        if filename:
            pickle.dump([punt,padres,hist_mean,hist_std,hist_sharpe,hist_padres],open( filename,'wb')) # guarda las variables más importantes al finalizar.

        return([punt,padres,hist_mean,hist_std,hist_sharpe,hist_padres])
