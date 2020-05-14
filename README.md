# Proyecto Final
Equipo 5: 
- Oscar Flores
- Fernanda Pinedo 
- Francisco Rodríguez 

## Se analiza el comportamiento del EUR_USD durante el comunicado del indicador FED INTEREST RATE DECISION de la economía USA

El proyecto a continuación consiste en encontrar la relación entre el indicador económico escogido, y las reacciones del precio de un activo financiero. Para este caso en particular, se escogió el indicador *Fed Interest Rate Decision*, indicador de las tasas de interés de la Reserva Federal de los Estados Unidos; este para poder observar las reacciones y el comportamiento del EUR/USD durante cada comunicado del mismo indicador.

Está compuesto por cuatro etapas, elaboradas por tres integrantes del equipo, en donde cada integrante tiene un perfil diferente, cada uno se encarga de un aspecto diferente del proyecto:
- *Financiero*: El integrante encargado del aspecto financiero eligió el indicador económico que se utilizó para el proyecto, así como fue su trabajo realizar las validaciones visuales para encontrar un patrón de que sucede con el tipo de cambio EUR/USD cuando se publica el indicador para así diseñar la estrategia de administración del capital y realizar aportes a la regla de trading.
- *Matemático/Estadístico*: El miembro encargado del aspecto matemático/estadístico se encarga de realizar la caracterización econométrica de la serie de tiempo del indicador escogido por el financiero, realizando así por medio de la metodología Box-Jenkins, pruebas de estacionariedad, autocorrelación, autocorrelación parcial, estacionalidad, heterocedasticidad, normalidad y detección de atípicos.
- *Programador*: El encargado del aspecto computacional es el que realiza la clasificación de escenarios de ocurrencia, clasificando en cuatro las reacciones que tiene el activo financiero EUR_USD cada que se hace un comunicado por parte del indicador.

Siendo las anteriores parte de la primera y segunda etapa, la tercera etapa del proyecto se trata de la programación de las reglas de trading de acuerdo a cada uno de los cuatro escenarios. Haciendo uso de al menos seis archivos .py siendo estos parte de la estructura del proyecto:
1. Entradas.py 
2. Datos.py
3. Procesos.py
4. Funciones.py
5. Principal.py
6. Visualizaciones.py

Así como el notebook de Jupyter .ipynb donde está explicado el proyecto de una manera más detallada, el .gitignore, una carpeta de imágenes usadas por el financiero para explicar visualmente la estrategia explicada a detalle en el notebook, y la carpeta de archivos en donde se encuentra el excel con los datos del indicador escogido.

Por último, la cuarta etapa es el backtest y la optimización de la estrategia, en donde el backtesting se usa para simular la estrategia de trading propuesta anteriormente con los datos pasados, para luego, utilizando métodos de optimización, se busque mejorar el rendimiento de las métricas de atribución al desempeño escogidas por los miembros del equipo, modificando de esta manera el stop loss, take profit y volumen.



Cualquier duda o comentario, no duden en contactarnos:

| Perfil | Nombre | GitHub Username | Correo ITESO |
|:---: | :-: | :-:| :-:|
| Financiero | Francisco Ricardo Rodríguez Ramírez | @FranciscoRodRam | if705783@iteso.mx |
| Matemático/Estadístico | María Fernanda Pinedo Talango | @ferpinedot | if705971@iteso.mx |
| Programador | Oscar Eduardo Flores Hernández | @OscarFlores-IFi | if715029@iteso.mx |

Todos actualmente estudiantes de la carrera de Ingeniería Financiera en el Instituto Tecnológico de Estudios Superiores de Occidente (ITESO)


Mayo/2020


