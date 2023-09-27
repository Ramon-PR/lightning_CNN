# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 15:18:45 2023

@author: Ramón Pozuelo
"""





from load_mic_MeshRIR import load_mic_MeshRIR
from module_functions_DataBases_RIR import load_DB_ZEA
import matplotlib.pyplot as plt
import os
import numpy as np

def fft_and_shift(t,f):
    fft_f = np.fft.fftshift( np.fft.fft(f) )
    frq_f = np.fft.fftshift( np.fft.fftfreq( t.size, t[1]-t[0] ) )
    
    return frq_f, fft_f

if __name__ == "__main__":
    
    # path_MeshRIR = "/scratch/ramonpr/3NoiseModelling/MeshRIR"
    path_MeshRIR = r"C:\Users\keris\Desktop\Postdoc\MeshRIR"
    path_Zea = r"C:\Users\keris\Desktop\Postdoc\DataBase_Zea"
    
    RIR, x, t = load_DB_ZEA( os.path.join(path_Zea,"BalderRIR.mat") )
    
    fig, ax = plt.subplots()
    plt.pcolor(x[0,:], t[:100,0], RIR[:100, :])
    plt.show()
    
    t_zea = t[:,0]
    mic_zea = RIR[:,20]
    
    mic_meshrir = load_mic_MeshRIR(path_MeshRIR, 0, 20)
    t_meshrir = np.arange(mic_meshrir.size)


# %% Compare ZeaRIR and MeshRIR microphones in phisical space

    fig, ax = plt.subplots(2,1)
    ax[0].plot(t_zea, mic_zea)
    ax[1].plot(t_meshrir, mic_meshrir)
    plt.show()

    
# %% Compare fft of ZeaRIR microphone with MeshRIR microphone

    f1 = np.fft.fftshift( np.fft.fft(mic_zea) )
    frqf1 = np.fft.fftshift( np.fft.fftfreq( f1.size, t_zea[1]-t_zea[0]  ) )
    
    frqf1, f1 = fft_and_shift(t_zea, mic_zea)
    frqf2, f2 = fft_and_shift(t_meshrir, mic_meshrir)
    
    fig, ax = plt.subplots(2,1)

    ax[0].plot(frqf1, np.abs(f1) )
    ax[1].plot(frqf2, np.abs(f2) )

    plt.show()


# %% FFT in time of the RIR images. 3 Different rooms
# % FFT IN TIME AXIS=0 (ROWS). Shift frequencies only in axis=0

    colormap='jet'
    RIR1, x, t = load_DB_ZEA( os.path.join(path_Zea,"BalderRIR.mat") )
    frqf1 = np.fft.fftshift( np.fft.fftfreq( t.shape[0], t[1,0]-t[0,0] ) )

    fft_RIR = np.fft.fftshift( np.fft.fft(RIR1, axis=0), 0 )    
    
    fig, ax = plt.subplots()
    plt.pcolor(x[0,:], frqf1, np.abs(fft_RIR) , cmap=colormap)
    plt.title('FFT Balder')
    plt.show()
    
    
# %
    
    colormap='jet'
    RIR2, x, t = load_DB_ZEA( os.path.join(path_Zea,"MuninRIR.mat") )
    frqf1 = np.fft.fftshift( np.fft.fftfreq( t.shape[0], t[1,0]-t[0,0] ) )

    fft_RIR = np.fft.fftshift( np.fft.fft(RIR2, axis=0), 0 )    
    
    fig, ax = plt.subplots()
    plt.pcolor(x[0,:], frqf1, np.abs(fft_RIR) , cmap=colormap)
    plt.title('FFT Munin')
    plt.show()

# %

    colormap='jet'
    RIR3, x, t = load_DB_ZEA( os.path.join(path_Zea,"FrejaRIR.mat") )
    frqf1 = np.fft.fftshift( np.fft.fftfreq( t.shape[0], t[1,0]-t[0,0] ) )

    fft_RIR = np.fft.fftshift( np.fft.fft(RIR3, axis=0), 0 )    
    
    fig, ax = plt.subplots()
    plt.pcolor(x[0,:], frqf1, np.abs(fft_RIR) , cmap=colormap)
    plt.title('FFT Freja')
    plt.show()


# %% Short time Fourier Transform of Balder, Munin & Freja

    from scipy.signal import stft

    Fs = 1.0/(t[1,0]-t[0,0])
    stft_1 = stft( RIR1[:,10], Fs, window='hann', padded=True)
    
    fig, ax = plt.subplots()
    plt.pcolormesh(stft_1[1] , stft_1[0], np.abs(stft_1[2]) , shading='gouraud')
    plt.title('STFT Balder')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.show()
    
    
    # Fs = 1.0/(t[1,0]-t[0,0])
    stft_2 = stft( RIR2[:,10], Fs, window='hann', padded=True)
    
    fig, ax = plt.subplots()
    plt.pcolormesh(stft_2[1] , stft_2[0], np.abs(stft_2[2]) , shading='gouraud')
    plt.title('STFT Munin')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.show()
    
    # Fs = 1.0/(t[1,0]-t[0,0])
    stft_3 = stft( RIR3[:,10], Fs, window='hann', padded=True)
    
    fig, ax = plt.subplots()
    plt.pcolormesh(stft_3[1] , stft_3[0], np.abs(stft_3[2]) , shading='gouraud')
    plt.title('STFT Freja')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.show()
    
# %% STFT. Compare same room different microphones.
    
    Fs = 1.0/(t[1,0]-t[0,0])
    stft_11 = stft( RIR1[:,10], Fs, window='hann', padded=True)
    stft_12 = stft( RIR1[:,90], Fs, window='hann', padded=True)

    stft_21 = stft( RIR2[:,10], Fs, window='hann', padded=True)
    stft_22 = stft( RIR2[:,90], Fs, window='hann', padded=True)
    
    fig, ax = plt.subplots(2,2)
    ax[0,0].pcolormesh(stft_11[1] , stft_11[0], np.abs(stft_11[2]) , shading='gouraud')
    ax[0,1].pcolormesh(stft_12[1] , stft_12[0], np.abs(stft_12[2]) , shading='gouraud')
    ax[1,0].pcolormesh(stft_21[1] , stft_21[0], np.abs(stft_21[2]) , shading='gouraud')
    ax[1,1].pcolormesh(stft_22[1] , stft_22[0], np.abs(stft_22[2]) , shading='gouraud')

    ax[0,0].set_xticklabels([])
    ax[0,1].set_xticklabels([])
    
    ax[0,1].set_yticklabels([])
    ax[1,1].set_yticklabels([])
    ax[1,1].set_yticklabels([])
    
    ax[0,0].set_title('STFT Balder, mic: 10')
    ax[0,1].set_title('STFT Balder, mic: 90')
    ax[1,0].set_title('STFT Munin, mic: 10')
    ax[1,1].set_title('STFT Munin, mic: 90')
    plt.show()
    
    
# %% STFT. Same room: Microphones further away. Difference in time

    Fs = 1.0/(t[1,0]-t[0,0])
    stft_11 = stft( RIR1[:,0], Fs, window='hann', padded=True)
    stft_12 = stft( RIR1[:,99], Fs, window='hann', padded=True)


    fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
    ax[0].pcolormesh(stft_11[1] , stft_11[0], np.abs(stft_11[2]) , shading='gouraud')
    ax[1].pcolormesh(stft_12[1] , stft_12[0], np.abs(stft_12[2]) , shading='gouraud')
    
    fig.suptitle('STFT Balder. mics 0 and 99')
    fig.supylabel('Frequency [Hz]')
    fig.supxlabel('Time [s]')
    
    # fig.text(0.04, 0.5, 'Frequency [Hz]', va='center', rotation='vertical')
    
    plt.xlim((0, 0.1))
    plt.tight_layout()
    plt.show()
    
    
    
    Fs = 1.0/(t[1,0]-t[0,0])
    stft_21 = stft( RIR2[:,0], Fs, window='hann', padded=True)
    stft_22 = stft( RIR2[:,99], Fs, window='hann', padded=True)

    fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
    ax[0].pcolormesh(stft_21[1] , stft_21[0], np.abs(stft_21[2]) , shading='gouraud')
    ax[1].pcolormesh(stft_22[1] , stft_22[0], np.abs(stft_22[2]) , shading='gouraud')
    
    fig.suptitle('STFT Munin. mics 0 and 99')
    fig.supylabel('Frequency [Hz]')
    fig.supxlabel('Time [s]')
    
    plt.xlim((0, 0.1))
    plt.tight_layout()
    plt.show()
    
# %% STFT & inver STFT
    
    from scipy.signal import istft

    id_mic = 20
    
    RIR1, x, t = load_DB_ZEA( os.path.join(path_Zea,"BalderRIR.mat") )
    t=t[:,0]
    signal_original = RIR1[:,id_mic]
    stft_signal_orig = stft( RIR1[:,id_mic], Fs, window='hann', padded=True)[2]
    
    # signal_istft = istft(stft_signal_orig, Fs, window='hann')
    t_inv, signal_istft = istft(stft_signal_orig, Fs)

    
    fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
    ax[0].plot(t , signal_original )
    ax[1].plot(t_inv , signal_istft)
    plt.tight_layout()
    plt.show()

# %% Compare raw signal with recovered signal after STFT & ISTFT

    frqf1 = np.fft.fftshift( np.fft.fftfreq( t.shape[0], t[1]-t[0] ) )
    fft_sig_orig = np.fft.fftshift( np.fft.fft(signal_original, axis=0), 0 )

    frqf2 = np.fft.fftshift( np.fft.fftfreq( t_inv.shape[0], t_inv[1]-t_inv[0] ) )
    fft_sig_istft = np.fft.fftshift( np.fft.fft(signal_istft, axis=0), 0 )


    fig, ax = plt.subplots(2,1)

    ax[0].plot(t , signal_original )
    ax[0].plot(t_inv , signal_istft)
    ax[0].set_xlim((0, 0.28))
    
    ax[1].plot(frqf1, np.abs(fft_sig_orig))
    ax[1].plot(frqf2, np.abs(fft_sig_istft))

    fig.suptitle('Effect of STFT & inverseSTFT')
    ax[0].set_xlabel('t[s]')
    ax[1].set_xlabel('F [1/s]')
    ax[0].legend(['raw signal', 'istft(stft(signal))'])

    ax[0].set_ylabel('signal')
    ax[1].set_ylabel('abs(fft( signal )) ')

    plt.tight_layout()
    plt.show()


# %% Wavelets
    from scipy import signal
    # scipy.signal.cwt(data, wavelet, widths, dtype=None, **kwargs)
    # Continuous wavelet transform.
    # Performs a continuous wavelet transform on data, using the wavelet function. 
    # A CWT performs a convolution with data using the wavelet function, 
    # which is characterized by a width parameter and length parameter. 
    # The wavelet function is allowed to be complex.
    
    import pywt
    
    id_mic = 20
    
    RIR1, x, t = load_DB_ZEA( os.path.join(path_Zea,"BalderRIR.mat") )
    t=t[:,0]
    
    signal_original = RIR1[:,id_mic]
    scales_stft, times_stft,  stft_signal_orig = stft( RIR1[:,id_mic], Fs, window='hann', padded=True)
    t_inv, signal_istft = istft(stft_signal_orig, Fs, window='hann')
    
    
    # fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
    # ax[0].plot(t , signal_original )
    # ax[1].plot(t_inv , signal_istft)
    # plt.tight_layout()
    # plt.show()
    
    
    
    
    # widths = np.arange(1,31)
    # cwtmatr = signal.cwt(signal_original, signal.ricker, widths)
    # cwtmatr_yflip = np.flipud(cwtmatr)
    
    # dt=time[1]-time[0]
    # coef, freq = pywt.cwt(sign al, scales, waveletname, dt)
    cwtmatr, freqs = pywt.cwt( signal_original, np.arange(1,31), "gaus1", sampling_period= )
    
    # plt.imshow(cwtmatr_yflip, extent=[-1,1,1,31], cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max() )

    plt.imshow(cwtmatr_yflip, cmap='PRGn', aspect='auto' )
    plt.show()
    
    









