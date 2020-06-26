# -*- coding: utf-8 -*-
"""
Created on Sun May 17 11:35:18 2020

@author: USER
"""

from Filter_design import filter_design, mfreqz
import scipy.signal as signal 
import matplotlib.pyplot as plt

#%%
def generate_filter(fs, lowF, highF, revfilt, auscultation_signal, graph):
    #Filter design
    order, filter_generated = filter_design(fs, locutoff = lowF, hicutoff = highF, revfilt = revfilt)
    
    # Filter behavior
    if (graph == 1): filter_behavior_graph(order, filter_generated, fs)
    
    return filter_generated

def apply_filter(filter_generated, input_signal):
    filteredSignal = signal.filtfilt(filter_generated, 1, input_signal) #(b,a,signal)
    return filteredSignal

def filter_behavior_graph(order, filter_generated,fs):
    normalizedFrequency, magnitude, magnitude_dB, h_phase = mfreqz(filter_generated, 1, fs/2)
    
    plt.figure()
    plt.plot(normalizedFrequency,magnitude)
    plt.title('Filter Generated Order '  + str(order) + ' - Magnitude')                                    
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.tight_layout()
    
    plt.figure()
    plt.plot(normalizedFrequency,magnitude_dB)
    plt.title('Filter Generated Order '  + str(order) + ' - Magnitude [dB]')                                    
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Amplitude [dB]')
    plt.grid()
    plt.tight_layout()
    
    plt.figure()
    plt.plot(normalizedFrequency,h_phase)
    plt.title('Filter Generated Order '  + str(order) + ' - Phase [pi-rad/sample]')                               
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Phase [pi-rad/sample]')
    plt.grid()
    plt.tight_layout()