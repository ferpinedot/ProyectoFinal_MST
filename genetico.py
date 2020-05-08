class Genetico:

    import pandas as pd
    import numpy as np
    from time import time
    import pickle
    import funciones as fn



    def fitness(individual, data):
        # individual será uno de los vectores de toma de df_decisiones
        # data es una lista con 2 elementos en el interior;
        # en la posición [0] contiene la descarga de datos de los instrumentos
        # en la posición [1] debería tener la clasificación de cada uno de los instrumentos
        uniques = pd.unique(data[1])
        indiv = [i[i*4:i*4+4] for i in range(uniques//4)]
        decisiones = pd.DataFrame(data= indiv,
                                  index = pd.unique(data[1]),
                                  columns = ['operacion', 'StopLoss', 'TakeProfit', 'Volume'])
        decisiones['operacion'][decisiones['operacion'] == 0] = 'venta'
        decisiones['operacion'][decisiones['operacion'] == 1] = 'compra'
        print(decisiones)

        datos_instrumento = data[0]
        clasificacion = data[1]

        df_backtest = fn.f_df_backtest(datos_instrumento, clasificacion, decisiones)
        mean = df_backtest['capital acumulado'].mean()
        std = df_backtest['capital acumulado'].std()
        print((mean - 0.003)/std)
        return (mean - 0.003)/std # 0.003 is the risk free rate for every 1.5 months.

    def create_first_generation(data, n):
        # data debería ser una lista, en la posición [0] contiene la descarga de datos de los instrumentos
        # en la posición [1] debería tener la clasificación de cada uno de los instrumentos.
        uniques = pd.unique(data[1])
        individuos = np.concatenate((np.random.randint(0,2,size=(n,1)),
                                np.random.randint(0,1000,size=(n,3))),axis=1)

        if uniques > 1:
            for _ in range(uniques - 1):
                individuos = np.concatenate((individuos,
                            np.concatenate((np.random.randint(0,2,size=(n,1)),
                            np.random.randint(0,1000,size=(n,3))),axis=1)),
                            axis = 1)
        return individuos

    def genetico(data,save = ''):


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

        t1 = time()

        decisiones = create_first_generation(data)

        decisiones = np.random.randint(-1,2,(n_vec,l_vec)) # Inicial.
        decisiones[-len(C0):] = C0

        hist_mean = np.zeros((iteraciones,n_vec//4*5)) # historial de media
        hist_std = np.zeros((iteraciones,n_vec//4*5)) # historial de desviación estandar
        hist_cal = np.zeros((iteraciones,n_vec//4*5)) # historial de calificaciones
        hist_padres = []

        punt = np.zeros(n_vec//4*5) # puntuaciones de hijos, se sobre-escribe en cada ciclo
        padres = np.zeros((n_vec//4,l_vec)) # padres, se sobre-escribe en cada ciclo

        #Para castigar y premiar baja desviación de rendimientos.
        pct_mean = np.zeros(punt.shape)
        pct_std = np.zeros(punt.shape)

        pct_mean = np.zeros(punt.shape)
        pct_std = np.zeros(punt.shape)


        for cic in range(iteraciones):
            t2 = time()
            for i in np.arange(n_vec): ## se simulan todos vectores de decisión para escoger el que de la suma mayor

                #######################################################################
                Sim = func(csv,ndias,model_close,decisiones[i],cetes) #########################
                pct = Sim[:,1:]/Sim[:,:-1]-1 ##########################################
                pct = pct.mean(axis=0) ##############################################
                pct_mean[i] = pct.mean() ########################################## todas las empresas
                pct_std[i] = pct.std() ############################################
                #######################################################################

            # Se da una calificación a cada vector de toma de decisiones.
            punt[pct_mean<pct_std] = (pct_mean[pct_std>pct_mean]-rf/252)/pct_std[pct_std>pct_mean]
#            punt = pct_mean-C*pct_std # Se le da una calificación (Vector de calificaciones)

            # Se escogen los padres.
            decisiones = np.concatenate((decisiones,padres)) # agregamos los 'padres' de las nuevas generaciones a la lista.
            indx = np.argsort(punt)[-int(n_vec//4):] # Indice donde se encuentran los mejores padres
            padres = decisiones[indx] # se escojen los padres
            pct_mean[-int(n_vec//4):] = pct_mean[indx] # se guarda la media que obtuvieron los padres
            pct_std[-int(n_vec//4):] = pct_std[indx] # se guarda la desviación que obtuvieron los padres

            hist_mean[cic,:] = pct_mean #se almacena el promedio de los padres para observar avance generacional
            hist_std[cic,:] = pct_std
            hist_cal[cic,:] = punt

            # Se mutan los vectores de toma de decisiones
            decisiones = np.array([[np.random.choice(padres.T[i]) for i in range(l_vec)] for i in range(n_vec)])
            for k in range(n_vec): ## mutamos la cuarta parte de los dígitos de los n_vec vectores que tenemos.
                for i in range(int(l_vec//4)):
                    decisiones[k][np.random.randint(0,l_vec)] = np.random.randint(0,3)-1

            # Para imprimir el proceso del algoritmo genérico en relación al total por simular y el tiempo de cada iteracion.
            print((np.ceil((1+cic)/iteraciones*1000)/10,time()-t2))

            # Cada 10 iteraciones se guardan los resultados de las simulaciones en un respaldo.
            resp = 5 #respaldo a cada resp
            if cic % resp == 0:
                hist_padres.append(padres)
                pickle.dump([punt,padres,hist_mean,hist_std,hist_cal,hist_padres],open('tmp.sav','wb'))


        print('tiempo de ejecución en seg.:')
        print(time()-t1)

        pickle.dump([punt,padres,hist_mean,hist_std,hist_cal,hist_padres],open(nombre + '.sav','wb')) # guarda las variables más importantes al finalizar.
