# -*- coding: utf-8 -*-
"""
Created on Sat May 30 15:51:19 2020

@author: SANTIAGO
"""
from Filter_routine import generate_filter, apply_filter
from Wavelet import procesador
procesador = procesador()

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio 
import scipy.signal as signal

#%%
def process(auscultation_signal, sr, 
            start=0, # time
            end=0, # time
            original_spectrum=0,
            original_time_freq=0,
            bandpass_filter=1,
            bandpass_graph=0,
            bandpass_spectrum=0,
            bandpass_time_freq=0,
            export_bandpass=0, # .mat (1), .wav (2)
            wavelet=1,
            x_min = 0,
            x_max = 0,
            level = 4,
            thershold = 'Minimax', # Universal, Minimax
            ponder = 'Multi nivel', # Comun, Primer nivel, Multi nivel
            hardness = 'Suave', # Duro, Suave
            see = 'Comparar', # Ver filtrada, Ver original, Comparar, Aprox y detalles, Multi nivel, Detalles filtrados, Multi nivel
            wavelet_graph=0,
            wavelet_spectrum=0,
            wavelet_time_freq=0,
            export_wavelet=0, # .mat (1), .wav (2)
            compare=0):
    
    if end == 0: end = len(auscultation_signal)
    
    if (x_max == 0): x_max = len(auscultation_signal)
    elif (x_max < x_min): 
        x_min = 0
        x_max = len(auscultation_signal)
    elif (x_max > len(auscultation_signal)): x_max = len(auscultation_signal)
    
    time = np.arange(0, len(auscultation_signal)/sr, 1/sr)

# Original Signal
    
    # Show spectrum of the signal after BandPass
    if original_spectrum == 1:
        plt.figure()
        f, Pxx = signal.welch(auscultation_signal[(time >= start) & (time <= end)], sr, 'hanning', sr*2, sr)
        plt.title('Original Signal Spectrum')
        plt.plot(f, Pxx)
        plt.xlabel('Frequency')
        plt.ylabel('Pxx')
    
    # Show time freq analysis after BandPass filter
    if original_time_freq == 1:
        n_mels = 128
        hop_length = 256 # 2^8 = 256
        n_fft = 1024 # 2^10 = 1024
        plt.figure()
        
        S = librosa.feature.melspectrogram(auscultation_signal[(time >= start) & (time <= end)], sr=sr, n_fft=n_fft, 
                                           hop_length = hop_length, n_mels = n_mels)
        
        S_DB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_DB, sr=sr, hop_length = hop_length, 
                                 x_axis='time', y_axis='mel')
        
        plt.colorbar(format='%+2.0f dB') 
        plt.title('After BandPass Signal Time-Frequency')

# BandPass
        
    # Filter the signal with a bandpass with cutoff[100:1000]
    # (Fs, LowCutOff, HighcutOff, RevFilt, Signal, Graph response)
    if bandpass_filter == 1:
        if bandpass_graph != 1: bandpass_graph = 0
        filter_generated = generate_filter(sr, 100, 1000, 0, auscultation_signal, graph = bandpass_graph)
        bandpass_signal = apply_filter(filter_generated, auscultation_signal)
        bandpass_signal = np.asfortranarray(bandpass_signal)
        
    # Show spectrum of the signal after BandPass
    if bandpass_spectrum == 1:
        plt.figure()
        f, Pxx = signal.welch(bandpass_signal[(time >= start) & (time <= end)], sr, 'hanning', sr*2, sr)
        plt.title('After BandPass Signal Spectrum')
        plt.plot(f, Pxx)
        plt.xlabel('Frequency')
        plt.ylabel('Pxx')
    
    # Show time freq analysis after BandPass filter
    if bandpass_time_freq == 1:
        n_mels = 128
        hop_length = 256 # 2^8 = 256
        n_fft = 1024 # 2^10 = 1024
        plt.figure()
        
        S = librosa.feature.melspectrogram(bandpass_signal[(time >= start) & (time <= end)], sr=sr, n_fft=n_fft, 
                                           hop_length = hop_length, n_mels = n_mels)
        
        S_DB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_DB, sr=sr, hop_length = hop_length, 
                                 x_axis='time', y_axis='mel')
        
        plt.colorbar(format='%+2.0f dB') 
        plt.title('After BandPass Signal Time-Frequency')
     
    # Export Wavelet to: .mat (1) or .wav (2)
    if export_bandpass == 1:
        sio.savemat('filtered_by_bandpass.mat', {'filtered': bandpass_signal[(time >= start) & (time <= end)]}) 
    
    elif export_bandpass == 2:
        sio.savemat('filtered_by_bandpass.mat', {'filtered': bandpass_signal})     
    
# Wavelet 
    
    # Apply Wavelet Filter
    if wavelet == 1:
        x_min = x_min
        x_max = x_max
        graph = wavelet_graph
        level = level
        thershold = thershold # Universal, Minimax
        ponder = ponder # Comun, Primer nivel, Multi nivel
        hardness = hardness # Duro, Suave
        see = see # Ver filtrada, Ver original, Comparar, Aprox y detalles, Multi nivel, Detalles filtrados, Multi nivel
        signal_after_wavelet = procesador.wavelet(bandpass_signal,
                                                  x_min, x_max, 
                                                  graph,
                                                  level,
                                                  thershold,
                                                  ponder,
                                                  hardness,
                                                  see)
# 2nd BandPass
    filtered_signal = apply_filter(filter_generated, signal_after_wavelet)
    filtered_signal = np.asfortranarray(filtered_signal)
    
    # Show spectrum of the signal after wavelet filter
    if wavelet_spectrum == 1:
        plt.figure()
        f, Pxx = signal.welch(filtered_signal[(time >= start) & (time <= end)], sr, 'hanning', sr*2, sr)
        plt.title('After Wavelet Spectrum')
        plt.plot(f, Pxx)
        plt.xlabel('Frequency')
        plt.ylabel('Pxx')
    
    # Show time freq analysis after wavelet filter
    if wavelet_time_freq == 1:
        n_mels = 128
        hop_length = 256 # 2^9
        n_fft = 1024 # 2^11
        plt.figure()
        
        S = librosa.feature.melspectrogram(filtered_signal[(time >= start) & (time <= end)], sr=sr, n_fft=n_fft,
                                           hop_length = hop_length, n_mels = n_mels)
        
        S_DB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_DB, sr=sr, hop_length = hop_length,
                                 x_axis='time', y_axis='mel')
        
        plt.colorbar(format='%+2.0f dB') 
        plt.title('After Wavelet Time-Frequency')
    
    # Export Wavelet to: .mat (1) or .wav (2)
    if export_wavelet == 1:
        sio.savemat('filtered_by_wavelet.mat', {'filtered': filtered_signal[(time >= start) & (time <= end)]}) 
    
    elif export_wavelet == 2:
        librosa.output.write_wav('nivel_'+str(level)+'_'+str(thershold)+'_'+str(ponder)+'_'+str(hardness)+'.wav',
                                 10*filtered_signal[(time >= start) & (time <= end)], sr)
    
    # Compare Signals
    if compare == 1:
        plt.figure()
        plt.subplot(3,1,1)
        plt.title('Original')
        librosa.display.waveplot(auscultation_signal[(time >= start) & (time <= end)], sr = sr) 
        plt.grid()
        plt.subplot(3,1,2)
        plt.title('BandPass')
        librosa.display.waveplot(bandpass_signal[(time >= start) & (time <= end)], sr = sr) 
        plt.grid()
        plt.subplot(3,1,3)
        plt.title('BandPass + Wavelet + BandPass')
        librosa.display.waveplot(filtered_signal[(time >= start) & (time <= end)], sr = sr)      
        plt.grid()
        
        plt.tight_layout()
        
    return time, filtered_signal[(time >= start) & (time <= end)]
    
        
