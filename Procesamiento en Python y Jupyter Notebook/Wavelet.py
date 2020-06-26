# -*- coding: utf-8 -*-
"""
Created on Sun May 17 16:26:18 2020

@author: USER
"""
import matplotlib.pyplot as plt
import numpy as np
import math

#%%
class procesador(object):
    def __init__(self):                                                                     # Se inicializan los atributos que tendra el modelo 
        self.senal = np.asarray([]) 
        self.x_min = 0
        self.x_max = 0
        
        self.nivel_max = 0                                                                 # FILTRAR Y PRESENTAR la señal cargada
        self.senalFiltrada = np.asarray([])
   
    def wavelet(self, senal, x_min, x_max, graficar, nivel, umbral, ponderar, dureza, ver):
        self.senal = senal
        self.ponderar = ponderar
        self.ver = ver
        
        if x_max > x_min:
            self.x_min = int(x_min)
            self.x_max = int(x_max)
        else:
            self.x_min = 0
            self.x_max = int(len(senal))
        
        if (graficar == None) or (graficar == 0):
            self.graficar = 0
        else:
            self.graficar = graficar
            
        
        self.descomponerSenal(nivel, umbral, ponderar, dureza)
        #print("ya Guarde la señal")
        return (self.senalFiltrada - self.senal)
#%%
    def descomponerSenal(self, nivel, umbral, ponderar, dureza):
        wavelet = [-1/np.sqrt(2) , 1/np.sqrt(2)]                                           # Filtro pasa altas
        escala = [1/np.sqrt(2) , 1/np.sqrt(2)]                                           # Filtro pasa bajas
        
                                                                                       # Se analiza la dimension de la señal (1 o 2), para
        senal = self.senal                                                            # determinar si debe tratarse un canal particular
        longitudOriginal = len(self.senal)
                                                                                            # Se determina la longitud original de la señal que se
                                                                                           # requiere para la reconstrucción luego del filtrado
        
                                                                                            # La descomposicion se guardara de la siguiente forma:
        aproximaciones = [senal]                                                           # [senal, aprox 1, aprox 2, ..., aprox n]                                                  
        detalles = []                                                                      # [detalle 1, detalle 2, ..., detalle n]
        posicion = 0
        
        self.nivel_max = np.floor(math.log(longitudOriginal/2,2)-1)                      # Se determina el nivel máximo de descomposición, en
        if nivel > self.nivel_max:                                                          # caso de que el usuario supere dicho valor, se limita
            nivel = self.nivel_max
        
        for i in range(int(nivel)):
            
            senalDescomponer = aproximaciones[posicion];                                    # Se recorre el vector aproximaciones, comenzando
                                                                                            # por la señal original, y guardando aproximacion
            if len(senalDescomponer) % 2 != 0:                                              # y detalle descompuesto de cada nivel en las 
                senalDescomponer = np.append(senalDescomponer,0);                           # LISTAS (aproximaciones y detalles)
        
            aproximacion = np.convolve(senalDescomponer,escala,'full');
            aproximacion = aproximacion[1::2];
        
            detalle = np.convolve(senalDescomponer,wavelet,'full');                                  
            detalle = detalle[1::2]; 
            
            aproximaciones.append(aproximacion);
            detalles.append(detalle);
            
            posicion = posicion + 1;
        
        aproximacion_n = aproximaciones[-1].copy();                                         # Se almacena la última aproximacion descompuesta, y
        detalles = detalles[::-1];                                                          # Los detalles se reordenan descendentemente:
                                                                                            # [detalle n, ..., detalle 2, detalle 1]
       
        senalDescompuesta = aproximacion_n;                                                 # La señal descompuesta constará de la unión de la
        for detalle in detalles:                                                            # aprox del ultimo nivel (aprox n) y los detalles
            senalDescompuesta = np.append(senalDescompuesta,detalle);                       # de todos los niveles en orden descendente:
                                                                                            # (aprox n, detalle n, ..., detalle 2, detalle 1)                   
        
        self.filtrar(umbral, ponderar, dureza, aproximacion_n, detalles, longitudOriginal, senalDescompuesta);
       
    #%%                                                                    2) RUTINA DE FILTRADO
        
    def filtrar(self, opcionUmbral, opcionPonderar, opcionDureza, aproximacion_n, detalles, longitudOriginal, senalDescompuesta):
        
        numMuestras = 0;                                                                        
        for detalle in detalles:                                                            # Para calcular el Umbral que determina lo que es
            numMuestras = numMuestras + len(detalle);                                       # ruido, se suman laslongitud de los detalles de 
        numMuestras = numMuestras + len(aproximacion_n);                                    # cada nivel y la aproximacion del ultimo nivel
     
                                         #%% a) Análisis de la opción de umbral seleccionada      
        if opcionUmbral == 'Universal':       
            umbral = np.sqrt(2*(np.log(numMuestras)));                                      # Se aplica una fórmula para el umbral según la
                                                                                            # selección del usuario
        elif opcionUmbral == 'Minimax':       
            umbral = 0.3936 + 0.1829*(np.log(numMuestras)/np.log(2));
            
        '''
        elif opcionUmbral == 'Sure':
            sx2 = sort(abs(x)).^2;
            risks = (n-(2*(1:n))+(cumsum(sx2)+(n-1:-1:0).*sx2))/n;
            best = min(risks);
            thr = sqrt(sx2(best));
            print('Escogio Sure \nUmbral = ', str(umbral));
        '''   
        
                                    #%% b) Análisis de la opción de ponderación seleccionada     
        banderaMultiNivel = 0;
        if opcionPonderar == 'Común':                                                       # El umbral calculado en a) no debe aplicarse,                           
            umbral = umbral;                                                                # directamente, debe ponderarse en función de  
            umbrales = [];                                                                  # diferentes criterios estadísticos, y puede 
                                                                                            # determinarse con: ninguno, el último, o todos
        if opcionPonderar == 'Primer nivel':                                                # los detalles descompuestos de la señal
            umbral = umbral*(np.median(np.absolute(detalles[-1])))/0.6745;
            umbrales = [];                                                                  # Para la opción primer nivel, el umbral se determina
                                                                                            # con el detalle del ultimo nivel descompuesto
        if opcionPonderar == 'Multi nivel':      
            banderaMultiNivel = 1;                                                          # Se activa una bandera requerida para aplicar el filtro
            umbrales = [];
            for detalle in detalles:                                                        # Para la opción multi nivel, se calcula un umbral
                sigma = (np.median(np.absolute(detalle)))/0.6745;                           # para cada detalle, y se van guardando en la                        
                umbrales.append(umbral*sigma);                                              # LISTA umbrales
                                                                                     
            
                               #%% c) Análisis de la opción de opción de dureza seleccionada         
    
        umbralDetalle = 0;                                                                  # Posicion empleada para recorrer 
                                                                                            # vector de umbrales en caso de existir 
                                                                                            
        if opcionDureza == 'Duro':                                                          # MECANISMO DE FILTRADO:
            if banderaMultiNivel == 1:                                                      
                for detalle in detalles:                                                    # Para filtrar una señal, debe aplicarse una ecuación
                    detalle[np.absolute(detalle) < umbrales[umbralDetalle]] = 0;            # a los detalles descompuestos, tal que cada valor
                    umbralDetalle = umbralDetalle + 1;                                      # se elimine si: 
            else:                                                                           # Su valor absoluto esta por debajo del umbral determinado.
                for detalle in detalles:                                                    
                    detalle[np.absolute(detalle) < umbral] = 0;
                                                                                            # Sí el valor absoluto del detalle está por encima del 
                                                                                            # umbral determinado, hay 2 opciones:                                                                             
            self.reconstruir(aproximacion_n, detalles, longitudOriginal, senalDescompuesta, umbral, umbrales);                        
                                                                                            # Opcion DURO: El detalle se deja intacto/igual
        if opcionDureza == 'Suave':
            detallesFiltrados = [];                                                         # Opcion SUAVE: Al valor absoluto se le resta el umbral
                                                                                            # y se multiplica por el signo del detalle
            if banderaMultiNivel == 0:                                                      # --------------------------------------------------------    
                for detalle in detalles:
                    signo = detalle.copy();                                                 # Si la opción escogida es Suave, se realiza el mismo
                    signo[signo < 0] = -1;                                                  # proceso para aplicar la formula antes descrita,
                    signo[signo >= 0] = 1;                                                  # la diferencia está en la banderaMultinivel, ya que
                                                                                            # es 1 si se escogió esta opción, y se debe emplear 
                    detalle[np.absolute(detalle) < umbral] = 0;                             # la LISTA DE UMBRALES generada.
                                                                                            # En caso de que la bandera sea 0, se aplica el mismo
                    detalleTemporal = [];                                                   # umbral a todos los detalles
                    
                    for detalle_k in range(len(detalle)):
                        if np.absolute(detalle[detalle_k]) >= umbral:
                            detalle[detalle_k] = signo[detalle_k]*(np.absolute(detalle[detalle_k])-umbral);
                        detalleTemporal.append(detalle[detalle_k]);
                    detallesFiltrados.append(detalleTemporal);
                
                
            if banderaMultiNivel == 1:                                                          
                 
                for detalle in detalles:
                    signo = detalle.copy();
                    signo[signo < 0] = -1;
                    signo[signo >= 0] = 1;
                    
                    detalle[np.absolute(detalle) < umbral] = 0;
                     
                    detalleTemporal = [];
                    
                    for detalle_k in range(len(detalle)):
                        if np.absolute(detalle[detalle_k]) >= umbral:
                            detalle[detalle_k] = signo[detalle_k]*(np.absolute(detalle[detalle_k])-umbrales[umbralDetalle]);
                        detalleTemporal.append(detalle[detalle_k]);
                    detallesFiltrados.append(detalleTemporal);
                    umbralDetalle = umbralDetalle + 1;                                          
            
            self.reconstruir(aproximacion_n,detalles,longitudOriginal,senalDescompuesta,umbral,umbrales); 
    
    #%%                                                              3) RUTINA DE RECONSTRUCCION
    def reconstruir(self,aproximacion_n,detalles,longitudOriginal,senalDescompuesta,umbral,umbrales):
                                                                                            # Recibe la aproximacion del ultimo nivel, y los 
        detallesFiltrados = detalles;                                                       # detalles de cada nivel en orden descendente:
        wavelet_inv = [1/np.sqrt(2) , -1/np.sqrt(2)];                                       # [aprox n], [detalle n, ..., detalle 2, detalle 1]]
        escala_inv = [1/np.sqrt(2) , 1/np.sqrt(2)];
        
        niveles = [];
        
        for i in range(len(detalles)):                                      
            if i == 0:
                numPuntos = len(aproximacion_n);                                            # El mecanismo de reconstrucción es inverso al 
                                                                                            # de descomposición:
                Aproximacion = np.zeros((2*numPuntos));                                     # 1) Generando un vector de zeros de tamaño doble
                Aproximacion[0::2] = aproximacion_n;                                        # 2) Cada 2 muestras a partir de la 2, se 
                Aproximacion[1::2] = 0;                                                     # introduce la ultima aproximacion
                
                Aproximacion = np.convolve(Aproximacion,escala_inv,'full');                 # El mismo proceso a la aprox y al detalle
                                                                                            # 3) Luego se hace la convolucion con la señal inversa
                numPuntos = len(detalles[i])
                Detalle = np.zeros((2*numPuntos));
                Detalle[0::2] = detalles[i];
                Detalle[1::2] = 0;
                
                Detalle = np.convolve(Detalle,wavelet_inv,'full');                                        
                
                nivel = Aproximacion + Detalle;                                             # NOTA:
                niveles.append(nivel);                                                      # El primer "if" reconstruye sobre la aproximacion y los
                                                                                            # detalles del último nivel (con la bandera "i = 0"). 
            else:                                                                           # Una vez reconstruida, se procede en el "else" a reconstruir
                                                                                            # sobre con los residuos sucesivos.
                if len(niveles[i-1]) > len(detalles[i]):
                    nivelTemporal = niveles[i-1];                                           
                    nivelTemporal = nivelTemporal[0:len(detalles[i])];
                    niveles[i-1] = nivelTemporal;                                           # Cada residuo se almacena en la LISTA niveles
                    
                numPuntos = len(niveles[i-1]);
                Aproximacion = np.zeros((2*numPuntos));                                        
                Aproximacion[0::2] = niveles[i-1];                                            
                Aproximacion[1::2] = 0;                                                         
                
                Aproximacion = np.convolve(Aproximacion,escala_inv,'full');                 
                                                                                             
                numPuntos = len(detalles[i])
                Detalle = np.zeros((2*numPuntos));
                Detalle[0::2] = detalles[i];
                Detalle[1::2] = 0;
                
                Detalle = np.convolve(Detalle,wavelet_inv,'full');                                        
                
                nivel = Aproximacion + Detalle; 
                niveles.append(nivel);
            
            
        senalFiltrada = niveles[-1].copy();                                                 # El ultimo residuo/término representa el primer nivel, 
        senalFiltrada = senalFiltrada[0:longitudOriginal];                                  # o nivel inicial de la señal original, pero filtrado
        #print(type(senalFiltrada));
        self.senalFiltrada = senalFiltrada;
        #print("ya filtre la señal")
        if (self.graficar == 1):
            self.graficarSenales(self.senal[self.x_min:self.x_max],
                                           senalFiltrada[self.x_min:self.x_max],
                                           senalDescompuesta,
                                           detallesFiltrados,
                                           umbral,
                                           umbrales,
                                           self.nivel_max); 
        
    def graficarSenales(self,senalOriginal,senalFiltrada,senalDescompuesta,detallesFiltrados,umbral,umbrales,nivel_max):
        ponderar = self.ponderar;
        ver = self.ver;
        
        if ver == 'Filtro Wavelet':                                                         # Esta función recibe del modelo  
            ver = 'Ver filtrada';                                                           # varios aspectos resultantes de las etapas 
                                                                                            # de descomposición, filtrado y reconstrucción 
        if ver == 'Ver filtrada':            
            plt.figure()
            plt.plot(senalFiltrada)
            
        elif ver == 'Ver original':
            plt.figure()
            plt.plot(senalOriginal)
            
        elif ver == 'Comparar':
            plt.figure()
            plt.plot(senalOriginal)
            plt.plot(senalFiltrada)
            plt.xlabel('Muestras')
            plt.ylabel('Amplitud')
            plt.title('Signal After Wavelet')
        
        elif ver == 'Aprox y detalles':                                                     # Sí el usuario desea ver la señal descompuesta
                                                                                             # analiza la opción de ponderación escogida
                                                                                            # para determinar si deben mostrarse uno
                                                                                              # o varios umbrales (uno por cada detalle
                                                                                            # en la opción Multi nivel)
            plt.figure()
            plt.plot(senalDescompuesta);
            
            if ponderar == 'Multi nivel':
                for i in (umbrales):
                    plt.plot(i*np.ones(len(senalDescompuesta)));
                    plt.plot(-i*np.ones(len(senalDescompuesta)));
            
            else:
                plt.plot(umbral*np.ones(len(senalDescompuesta)));
                plt.plot(-umbral*np.ones(len(senalDescompuesta)));
    
        elif ver == 'Detalles filtrados':
            plt.figure()
            
            for detalleFiltrado in detallesFiltrados:
                plt.plot(detalleFiltrado);
            
            if ponderar == 'Multi nivel':
                for i in range(len(umbrales)):
                    plt.plot(i*np.ones(len(senalDescompuesta)));
                    plt.plot(-i*np.ones(len(senalDescompuesta)));
            
            else:
                plt.plot(umbral*np.ones(len(senalDescompuesta)));
                plt.plot(-umbral*np.ones(len(senalDescompuesta)));
            
        
        
        
        