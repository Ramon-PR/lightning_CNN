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
    
    cc = signal.correlate(x, y, mode="same")

    i_max = np.argmax(cc)
    phi_shift = np.linspace(-0.5*L, 0.5*L , N)
    delta_phi = phi_shift[i_max]
    
    fig, ax = plt.subplots(2,1)
    plt.sca(ax[0])
    plt.plot(t, x)
    plt.plot(t, y)
    # plt.xlim([0, 0.02])
    plt.title("Señales entrada")
    plt.sca(ax[1])
    plt.plot(t, x)
    plt.plot(t + delta_phi, y, 'r')
    # plt.xlim([0, 0.02])
    plt.title("Señal 2 con shift")
    plt.tight_layout()

    return delta_phi

def two_point_corr(t, x, y):
    fx = np.fft.fft(x)
    fy = np.fft.fft(y)
    
    n = fx.size
    sp = fx*np.conj(fy)/n
    two_pcorr = np.fft.ifft(sp)
    return two_pcorr.real
    
    
if __name__ == "__main__":

    # path_MeshRIR = r"C:\Users\keris\Desktop\Postdoc\MeshRIR"
    # path_Zea = r"C:\Users\keris\Desktop\Postdoc\DataBase_Zea"
    path_MeshRIR = r"/scratch/ramonpr/3NoiseModelling/MeshRIR"
    path_Zea = r"/scratch/ramonpr/3NoiseModelling/DataBase_Zea"
    
    RIR, x, t = load_DB_ZEA( os.path.join(path_Zea,"BalderRIR.mat") )
    
    mic_meshrir = load_mic_MeshRIR(path_MeshRIR, 0, 20)
    t_meshrir = np.arange(mic_meshrir.size)
    
    t=t[:,0]
    x=x[0,:]
    
    mic_a = RIR[:,20]
    mic_b = RIR[:,50]
    
    
    d_fase = detect_phase_shift(t, mic_a, mic_b)
    two_pcorr = two_point_corr(t, mic_a, mic_b)
    index = np.argmax(two_pcorr)
    d_fase2 = t[index]
    
    print(" Desfase de las dos señales de microfonos ")
    print("dfase: ",  d_fase)
    print("dfase two point corr: ",  d_fase2)
    
    fig, ax = plt.subplots(2,1)
    plt.sca(ax[0])
    plt.plot(t, mic_a)
    plt.plot(t, mic_b)
    plt.xlim([0, 0.02])
    plt.title("Señales originales")
    plt.sca(ax[1])
    plt.plot(t, mic_a)
    plt.plot(t-d_fase, mic_b, 'r')
    plt.xlim([0, 0.02])
    plt.title("Señal 2 (roja) tras aplicar desfase")
    plt.tight_layout()
    
    
    # %% Antes de usar el metodo del desfase usar valor absoluto de las señales
    
    d_fase = detect_phase_shift(t, abs(mic_a), abs(mic_b))
    two_pcorr = two_point_corr(t, abs(mic_a), abs(mic_b))
    index = np.argmax(two_pcorr)
    d_fase2 = t[index]
    
    print(" Desfase de las dos señales de microfonos ")
    print("dfase: ",  d_fase)
    print("dfase two point corr: ",  d_fase2)
    
    fig, ax = plt.subplots(2,1)
    plt.sca(ax[0])
    plt.plot(t, mic_a)
    plt.plot(t, mic_b)
    # plt.xlim([0, 0.02])
    plt.title("Señales originales")
    plt.sca(ax[1])
    plt.plot(t, mic_a)
    plt.plot(t+d_fase, mic_b, 'r')
    # plt.plot(t-d_fase2, mic_b, 'r')
    
    
    # plt.xlim([0, 0.02])
    plt.title("Señal 2 (roja) tras aplicar desfase")
    plt.tight_layout()
    
    
    # %%
    
    lags = np.arange(-(t.size-1), t.size) * (t[1]-t[0])
    
    ab = np.correlate(mic_a, mic_b, mode="full")
    ba = np.correlate(mic_b, mic_a, mode="full")
    
    fig, ax = plt.subplots(3,1)
    plt.sca(ax[0])
    plt.plot(ab)
    plt.plot(ba)
    plt.legend(['correlate(a,b)', 'correlate(b,a)'])
    
    plt.sca(ax[1])
    plt.plot(ab)
    plt.plot(ba[::-1])
    plt.legend(['correlate(a,b)', 'sym(correlate(b,a))'])

    
    ab_cv = np.convolve(mic_a, mic_b, mode="full")
    ba_cv = np.convolve(mic_b, mic_a, mode="full")
    plt.sca(ax[2])
    plt.plot(ab_cv)
    plt.plot(ba_cv)
    plt.legend(['conv(a,b)', 'conv(b,a)'])
        
    
    # %%
    
    plt.close('all')
    
    fig, ax = plt.subplots(2,1)
    plt.sca(ax[0])
    plt.plot(t, mic_a)
    plt.plot(t, mic_b)
    plt.xlim([0, 0.02])
    plt.title("Señales originales")
    plt.sca(ax[1])
    # plt.plot(lags, ab)
    plt.plot(lags, ab_cv)
    # plt.plot(mic_b[40-5:], 'r')
    # plt.xlim([-0.05, 0.05])


# %%

print(np.argmax(ab))
print(np.argmax(ba))
print(np.argmax(ab_cv))
print(np.argmax(ba_cv))

# L = t[-1]


print(lags)












