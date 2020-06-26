# -*- coding: utf-8 -*-
"""
Created on Sun May 31 14:21:31 2020

@author: SANTIAGO
"""

import numpy as np
import scipy.signal as signal

#%%

def features(chunk):
    
    var = np.var(chunk)
    ran = abs(max(chunk)-min(chunk))
    coarse_sum = 0
    fine_sum = []
    spectrum_av = 0
    
    # Coarse Sum
    cont = 0
    if len(chunk) >= 4096:
        for i in range(int(len(chunk)/4096)):
            if (i+1)*4096 < len(chunk): # end limit 
                coarse_sum += sum(chunk[i*4096:(i+1)*4096])
            else: 
                coarse_sum += sum(chunk[i*4096:])
            cont += 1
        
        coarse_sum /= cont
    else:
        coarse_sum = sum(chunk)
    
    # Fine Sum
    cont = 0
    for i in range(int(len(chunk)/800)):
        temporal_sum = 0
        if (i+1)*800 < len(chunk): # end limit
            for i in range(8):
                temporal_sum += sum(chunk[i*100:(i+1)*100])
        else:
            longitud = len(chunk) - 800
            for i in range(8):
                temporal_sum += sum(chunk[longitud + i*100 : longitud + (i+1)*100])
            
        cont += 1
        temporal_sum /= 8
        fine_sum.append(temporal_sum)
        
    fine_sum = max(fine_sum)
    
    # Spectrum mean
    sr = 4000
    f, Pxx = signal.welch(chunk, sr)
    spectrum_av = np.mean(Pxx)
    
    dictionary = {'Var':var,'Range':ran,'Coarse_Av':coarse_sum,
                  'Fine_Av':fine_sum,'Spectrum Av':spectrum_av}
    return dictionary

        



