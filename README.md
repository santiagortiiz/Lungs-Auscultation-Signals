# Lungs-Auscultation-Signals

Este proyecto contiene el análisis de una base de datos de Kaggle sobre auscultación pulmonar (https://www.kaggle.com/vbookshelf/respiratory-sound-database). 

Los audios disponibles se limpian con filtros digitales lineales, y no lineales (como el filtro de Wavelet), 
se calculan algunos índices para su caracterización y se genera una tabla de Excel con la información recopilada, 
que es analizada estadísticamente, encontrando como resultado que los ciclos respiratorios sanos difieren 
de los enfermos en el promedio de la potencia espectral de su señal


Se anexan 7 archivos .py en los cuales se desarrolló todo el proyecto, **se recomienda leerlos en el siguiente orden**
para comprender a cabalidad la metodología empleada en este estudio:

**-Filter_design.py:** Diseño de filtros digitales.

**-Filter_routine.py:** Rutina que permite crear un filtro, aplicarlo, y visualizar su comportamiento en un diagrama de Bode.

**-Wavelet.py:** Contiene la clase procesador que ejecuta la rutina de descomposición de una señal, filtrado wavelet, y reconstrucción por transformada de Haar.

**-Processor.py:** Contiene la rutina que se encarga de limpiar una señal de auscultación aplicando secuencialmente filtros digitales y el filtro wavelet, 
además de permitir graficar el análisis frecuencial y exportar la señal a formatos .mat o .wav en cualquier etapa del proceso.

**-Features.py:** Contiene el método features que permite calcular los índices de interés en este estudio y los retorna en forma de diccionario.

**-Auscultation_signals.py:** Rutina que llama el método process del archivo Processor.py, y el método features del archivo Features para generar una señal
de auscultación filtrada, y extraer los índices de interés.

**-Database_Processing.py:** Script que aplica la rutina descrita en Auscultation_signals.py sobre cada uno de los elementos de la base de datos, 
y genera una tabla de Excel que recopila toda la información disponible en la base de datos trabajada, en conjunto con la información extraída del 
procesamiento de las señales.


Finalmente se anexa el archivo de Excel que generó Database_Procesing.py, y el archivo Statistics, 
el cual es un Jupyter Notebook que contiene un análisis detallado de los datos recopilados y con base en el cual se realizó este artículo.
