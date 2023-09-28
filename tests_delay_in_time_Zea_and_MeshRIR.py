# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 23:24:02 2023

@author: Ramón Pozuelo
"""


import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal

from load_mic_MeshRIR import load_mic_MeshRIR
from module_functions_DataBases_RIR import load_DB_ZEA




def fft_and_shift(t,f):
    fft_f = np.fft.fftshift( np.fft.fft(f) )
    frq_f = np.fft.fftshift( np.fft.fftfreq( t.size, t[1]-t[0] ) )
    
    return frq_f, fft_f

def detect_phase_shift(t, x, y):
    '''detect phase shift between two signals from cross correlation maximum'''
    N = len(t)
    L = t[-1] - t[0]
    
    window = 1 #signal.windows.hann(N)   # Hanning window 
    x_windowed = x*window
    y_windowed = y*window
    
    # cc = signal.correlate(x, y, mode="same")
    cc = signal.correlate(x_windowed, y_windowed, mode="same")

    i_max = np.argmax(cc)
    phi_shift = np.linspace(-0.5*L, 0.5*L , N)
    delta_phi = phi_shift[i_max]

    return delta_phi

def two_point_corr(t, x, y):
    fx = np.fft.fft(x)
    fy = np.fft.fft(y)
    
    n = fx.size
    sp = fx*np.conj(fy)/n
    two_pcorr = np.fft.ifft(sp)
    return two_pcorr.real
    
    
if __name__ == "__main__":

    path_MeshRIR = r"C:\Users\keris\Desktop\Postdoc\MeshRIR"
    path_Zea = r"C:\Users\keris\Desktop\Postdoc\DataBase_Zea"
    
    RIR, x, t = load_DB_ZEA( os.path.join(path_Zea,"BalderRIR.mat") )
    
    mic_meshrir = load_mic_MeshRIR(path_MeshRIR, 0, 20)
    t_meshrir = np.arange(mic_meshrir.size)
    
    t=t[:,0]
    x=x[0,:]
    
    mic_a = RIR[:,20]
    mic_b = RIR[:,50]
    
    
    d_fase = detect_phase_shift(t, abs(mic_a), abs(mic_b))
    
    
    
    print("dfase: ",  d_fase)
    
    fig, ax = plt.subplots()
    plt.plot(t, mic_a)
    plt.plot(t, mic_b)
    plt.xlim([0, 0.02])
    
    # %%
    fig, ax = plt.subplots()
    plt.plot(t , mic_a)
    # plt.plot(t - d_fase, mic_b)
    plt.plot(t + d_fase, mic_b)

    plt.xlim([0, 0.02])
    
    
    # %%
    two_pcorr = two_point_corr(t, mic_a, mic_b)
    
    fig, ax = plt.subplots()
    plt.plot(two_pcorr)
    # plt.plot(t - d_fase, mic_b)
    # plt.plot(t + d_fase, mic_b)

    # plt.xlim([0, 0.02])
    
    # %%
    
    print(np.argmax(two_pcorr))
    
    
    
    # %%
    
    # %%
    fig, ax = plt.subplots()
    plt.plot(t , mic_a)
    # plt.plot(t - d_fase, mic_b)
    plt.plot(t - 4*t[5], mic_b)

    plt.xlim([0, 0.02])
    
    
    
    
    



