# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 16:07:35 2023

@author: Ramón Pozuelo
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal


r2d = 180.0/np.pi   # conversion factor RAD-to-DEG
delta_phi_true = 50.0/r2d

def detect_phase_shift(t, x, y):
    '''detect phase shift between two signals from cross correlation maximum'''
    N = len(t)
    L = t[-1] - t[0]
    
    cc = signal.correlate(x, y, mode="same")
    i_max = np.argmax(cc)
    phi_shift = np.linspace(-0.5*L, 0.5*L , N)
    delta_phi = phi_shift[i_max]

    # print("true delta phi = {} DEG".format(delta_phi_true*r2d))
    # print("detected delta phi = {} DEG".format(delta_phi*r2d))
    # print("error = {} DEG    resolution for comparison dphi = {} DEG".format((delta_phi-delta_phi_true)*r2d, dphi*r2d))
    # print("ratio = {}".format(delta_phi/delta_phi_true))
    return delta_phi


L = np.pi*10+2     # interval length [RAD], for generality not multiple period
N = 1001   # interval division, odd number is better (center is integer)
noise_intensity = 0.5
X = 0.5   # amplitude of first signal..
Y = 2.0   # ..and second signal

phi = np.linspace(0, L, N)
dphi = phi[1] - phi[0]

'''generate signals'''
nx = noise_intensity*np.random.randn(N)*np.sqrt(dphi)   
ny = noise_intensity*np.random.randn(N)*np.sqrt(dphi)
x_raw = X*np.sin(phi) + nx
y_raw = Y*np.sin(phi+delta_phi_true) + ny

'''preprocessing signals'''
x = x_raw.copy() 
y = y_raw.copy()
window = signal.windows.hann(N)   # Hanning window 
#x -= np.mean(x)   # zero mean
#y -= np.mean(y)   # zero mean
#x /= np.std(x)    # scale
#y /= np.std(y)    # scale
x *= window       # reduce effect of finite length 
y *= window       # reduce effect of finite length 

print(" -- using raw data -- ")
delta_phi_raw = detect_phase_shift(phi, x_raw, y_raw)

print(" -- using preprocessed data -- ")
delta_phi_preprocessed = detect_phase_shift(phi, x, y)

# %%

fig, ax = plt.subplots()
ax.plot(phi, x, '-')
ax.plot(phi+0.8726646259971648, y, '--')
plt.show()

# %%

fig, ax = plt.subplots()
ax.plot(phi, x_raw, '-')
ax.plot(phi+0.8726646259971648, y_raw, '--')
plt.show()
