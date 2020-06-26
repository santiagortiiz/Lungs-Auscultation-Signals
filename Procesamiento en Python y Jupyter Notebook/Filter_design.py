# -*- coding: utf-8 -*-
"""
Created on Sun May 17 11:15:17 2020

@author: USER
"""

import numpy as np;
import scipy.signal as signal

#%%     Sinc Function generation

def fkernel(m, f, w):
    m = np.arange(-m/2, (m/2)+1) # longitud 
    b = np.zeros((m.shape[0])) # vector de zeros de longitud m
    b[m==0] = 2*np.pi*f # El vector en m = 0 se reemplaza por la funcion seno
    b[m!=0] = np.sin(2*np.pi*f*m[m!=0]) / m[m!=0] # y en el resto de valores se rellena con la fn sinc
    b = b * w # El vector b se enventana
    b = b / np.sum(b) # y luego se normaliza con el promedio de sus datos
    return b

#%%     Control of FIR filter

def fspecinv(b):
    b = -b
    b[int((b.shape[0]-1)/2)] = b[int((b.shape[0]-1)/2)]+1
    return b

def firws(m, f , w , t = None):
    """
    Designs windowed sinc type I linear phase FIR filter.
    Parameters:        
        m: filter order.
        f: cutoff frequency/ies (-6 dB;pi rad / sample).
        w: vector of length m + 1 defining window. 
        t: 'high' for highpass, 'stop' for bandstop filter. {default low-/bandpass}
    Returns:
        b: numpy.ndarray
            filter coefficients 
    """
    f = np.squeeze(f) # Toma las frecuencias de corte en forma de arreglo y las divide a la mitad
    f = f / 2; 
    w = np.squeeze(w) # Toma la funcion ventana definida y la transforma en un arreglo
    
    if (f.ndim == 0): #low pass     # Sí el usuario solo ingresó un valor de f, la dimension
        b = fkernel(m, f, w)        # de f es 0 y por defecto es pasa bajas
    else:                           # De lo contrario, es pasa banda. Despues de determinar que filtro
        b = fkernel(m, f[0], w) #band # se desea, se general la funcion kernel
    
    if (f.ndim == 0) and (t == 'high'): # Sin embargo si la dimension es 0, pero el usuario ingreso
        b = fspecinv(b)                 # pasa altas, se invierte el espectro de la funcion kernel
                                        # generada
    
    elif (f.size == 2):            # Sí el usuario ingreso 2 frecuencias, es porque desea pasa banda o
        b = b + fspecinv(fkernel(m, f[1], w)) # rechazabanda, por lo que inicialmente se crea el
                                   # rechazabandas como la suma del P. bajas ó el P. Banda
                                   # (ya creado y asignado a 'b'), con un nuevo kernel con el espectro
                                   # invertido que dependerá de la frecuencia alta ingresada en f
        
        if t == None or (t != 'stop'): #  En caso de que no se haya escogido rechazabanda, se 
            b = fspecinv(b) #bandpass  # invierte el rechazabanda recien generado
    return b

#%%     Filter Design
    
def filter_design(srate, locutoff = 0, hicutoff = 0, revfilt = 0):
    #Constants
    TRANSWIDTHRATIO = 0.25;
    fNyquist = srate/2;  
    
    #The prototipical filter is the low-pass, we design a low pass and transform it
    if hicutoff == 0: #Convert highpass to inverted lowpass
        hicutoff = locutoff
        locutoff = 0
        revfilt = 1 #invert the logic for low-pass to high-pass and for
                    #band-pass to notch
    if locutoff > 0 and hicutoff > 0:
        edgeArray = np.array([locutoff , hicutoff])
    else:
        edgeArray = np.array([hicutoff]);
    
    #Not negative frequencies and not frequencies above Nyquist
    if np.any(edgeArray<0) or np.any(edgeArray >= fNyquist):
        print('Cutoff frequency out of range')
        return False  
    
    # Max stop-band width
    maxBWArray = edgeArray.copy() # Band-/highpass
    if revfilt == 0: # Band-/lowpass
        maxBWArray[-1] = fNyquist - edgeArray[-1];
    elif len(edgeArray) == 2: # Bandstop
        maxBWArray = np.diff(edgeArray) / 2;
    maxDf = np.min(maxBWArray);
    
    # Default filter order heuristic
    if revfilt == 1: # Highpass and bandstop
        df = np.min([np.max([maxDf * TRANSWIDTHRATIO, 2]) , maxDf]);
    else: # Lowpass and bandpass
        df = np.min([np.max([edgeArray[0] * TRANSWIDTHRATIO, 2]) , maxDf]);
    
    print(df)
    
    filtorder = 3.3 / (df / srate); # Hamming window
    filtorder = np.ceil(filtorder / 2) * 2; # Filter order must be even.
    
    # Passband edge to cutoff (transition band center; -6 dB)
    dfArray = [[df, [-df, df]] , [-df, [df, -df]]];
    cutoffArray = edgeArray + np.array(dfArray[revfilt][len(edgeArray) - 1]) / 2;
    print('pop_eegfiltnew() - cutoff frequency(ies) (-6 dB): '+str(cutoffArray)+' Hz\n');
    # Window
    winArray = signal.hamming(int(filtorder) + 1);
    # Filter coefficients
    if revfilt == 1:
        filterTypeArray = ['high', 'stop'];
        b = firws(filtorder, cutoffArray / fNyquist, winArray, filterTypeArray[len(edgeArray) - 1]);
    else:
        b = firws(filtorder, cutoffArray / fNyquist, winArray);

    return filtorder, b; 

#%%     Filtre Graph
    
def mfreqz(b,a,nyq_rate = 1):
    
    """
    Plot the impulse response of the filter in the frequency domain

    Parameters:
        
        b: numerator values of the transfer function (coefficients of the filter)
        a: denominator values of the transfer function (coefficients of the filter)
        
        order: order of the filter 
                
        nyq_rate = nyquist frequency
    """
    
    w,h = signal.freqz(b,a);
    
    normalizedFrequency = (w/max(w))*nyq_rate 
    magnitude = abs(h)
    magnitude_dB = 20 * np.log10 (magnitude)
    h_phase = np.unwrap(np.arctan2(np.imag(h),np.real(h)))

    return normalizedFrequency, magnitude, magnitude_dB, h_phase

