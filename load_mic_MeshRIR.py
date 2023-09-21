# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:57:28 2023

@author: Ramón Pozuelo
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal

def load_mic_MeshRIR(path_MeshRIR, room_id, mic_id):
    # path_MeshRIR = "/.../MeshRIR"
    # room_id = 0 # or 1
    # mic_id = 0 to 3968 or 0 to 440
    # mic_data = load_mic_MeshRIR(path_MeshRIR, room_id, mic_id)
    Rooms = ["S1-M3969_npy", "S32-M441_npy"]
    room_path = os.path.join(path_MeshRIR, Rooms[room_id])
    mic_path = os.path.join(room_path, "ir_"+str(mic_id)+".npy")
    mic_data = np.load(mic_path)[0,:]
    return mic_data


if __name__ == "__main__":
    
    # path_MeshRIR = "/scratch/ramonpr/3NoiseModelling/MeshRIR"
    path_MeshRIR = r"C:\Users\keris\Desktop\Postdoc\MeshRIR"
    room_id = 0 # or 1
    mic_id = 0 
    mic_data_1 = load_mic_MeshRIR(path_MeshRIR, room_id, mic_id)
    mic_data_2 = load_mic_MeshRIR(path_MeshRIR, 1, 200)

    x_ax1 = np.arange(mic_data_1.size)
    x_ax2 = np.arange(mic_data_2.size)

    from numpy.fft import fft, ifft
    Fsig1 = fft(mic_data_1)
    Fsig2 = fft(mic_data_2)
    freqs = np.fft.fftfreq(Fsig1.size)
    
    freqs = np.fft.fftshift(freqs)
    Fsig1 = np.fft.fftshift(Fsig1)
    Fsig2 = np.fft.fftshift(Fsig2)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    ax1.plot(mic_data_1, 'b')
    ax2.plot(mic_data_2, 'orange')
    ax3.plot(freqs, np.absolute(Fsig1),'b')
    ax4.plot(freqs, np.absolute(Fsig2), 'orange')
    plt.show()

