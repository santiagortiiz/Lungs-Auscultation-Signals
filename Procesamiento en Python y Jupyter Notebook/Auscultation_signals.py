# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:50:15 2020

@author: USER
"""
from Features import features
from Processor import process
from Wavelet import procesador
procesador = procesador()

import librosa

#%% Signal load
auscultation_signal, sr = librosa.load('audio_and_txt_files/104_1b1_Ll_sc_Litt3200.wav',4000) 
#sio.savemat('auscultation_signal.mat', {'original': auscultation_signal}) 

#%%
time, filtered_signal = process(auscultation_signal, sr, 
                          start=0, 
                          end=0,
                          original_spectrum=1,
                          original_time_freq=1,
                          bandpass_filter=1, # BandPass[100,1000] Interest Frequency
                          bandpass_graph=1, 
                          bandpass_spectrum=1,
                          bandpass_time_freq=1,
                          export_bandpass=0, # .mat (1), .wav (2)
                          wavelet=1,
                          x_min = 0,
                          x_max = 0,
                          level = 4,
                          thershold = 'Minimax', # Universal, Minimax
                          ponder = 'Multi nivel', # Comun, Primer nivel, Multi nivel
                          hardness = 'Suave', # Duro, Suave
                          see = 'Comparar', # Ver filtrada, Ver original, Comparar, Aprox y detalles, Multi nivel, Detalles filtrados, Multi nivel
                          wavelet_graph=1,
                          wavelet_spectrum=1,
                          wavelet_time_freq=1,
                          export_wavelet=0, # .mat (1), .wav (2)
                          compare=1)

#%%

dictionary = features(filtered_signal)