# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:11:20 2023

@author: Ramón Pozuelo
"""
import pymatreader as pymat
import random
import numpy as np

#-----------------------------------------------------------------------
#---------- LOAD DATABASE ------------------------------------------------
#-----------------------------------------------------------------------

def load_DB_ZEA(path):
    #Load data
    RIR=[]
    RIR = pymat.read_mat(path)["out"]["image"]
    fs = pymat.read_mat(path)["out"]["fs"] # Hz
    
    T, M = RIR.shape
    
    x0 = 0.0
    dx = 0.03 # m
    x = np.arange(0,M).reshape((1,M))*dx + x0

    t0 = 0.0
    dt = 1.0/fs # s
    t = np.arange(0,T).reshape((T,1))*dt + t0
    
    # X, Time = np.meshgrid(x, t, indexing="xy")

    return RIR, x, t #X, Time

#-----------------------------------------------------------------------
#---------- DIVIDE RIR IMAGE IN SUBSAMPLES -----------------------------
#-----------------------------------------------------------------------
def divide_RIR(RIR, maskX=None, maskT=None, kt=32, kx=32, strideT=32, strideX=32):
    # Divide RIR in small imagenes 
    # the image is a kernel of size (kt, kx) 
    # the kernel moves in strides strideX and strideT
    # if kt=strideT and kx=strideX then "no-overlap"
    # it also gives the downsampling submask for each subimage.
    
    T, M = RIR.shape
    
    if kt > T or kx > M:
        raise ValueError("The kernel size cannot be larger than the dimensions of RIR.")

    # stride = s
    # kernel = k
    # total: M
    xImags = (M - kx)//strideX + 1
    tImags = (T - kt)//strideT + 1

    nImags = (xImags)*(tImags)

    # Imags without modification
    Y = np.zeros([nImags,kt,kx])

    for imx in range(xImags):
        for imt in range(tImags):
            im = imx*tImags + imt
            Y[im, :, :] = RIR[strideT*imt  : (strideT*imt)+kt, strideX*imx  : (strideX*imx)+kx]

    if maskX is not None:
        subMaskX = np.ones([nImags,1,kx], dtype=maskX.dtype)
        for imx in range(xImags):
            for imt in range(tImags):
                im = imx*tImags + imt
                subMaskX[im, 0,:] = maskX[0, strideX*imx  : (strideX*imx)+kx]            
    else:
        subMaskX = None
    
    if maskT is not None:
        subMaskT = np.ones([nImags,kt,1], dtype=maskT.dtype)
        for imx in range(xImags):
            for imt in range(tImags):
                im = imx*tImags + imt
                subMaskT[im, :,0] = maskT[strideT*imt  : (strideT*imt)+kt, 0]
    else:
        subMaskT = None

    return Y, subMaskX, subMaskT

#-----------------------------------------------------------------------
#---------- DOWNSAMPLE RIR IMAGE and GENERATE the General Masks --------
#-----------------------------------------------------------------------


def rand_downsamp_RIR(shape, ratio_t=1, ratio_x=0.5):
    # choose a ratio of samples in time/space from RIR
    # random choice
    T, M = shape
    tsamples = int(T*ratio_t)
    xMics  = int(M*ratio_x)

    id_T = np.sort(random.sample(range(0,T), tsamples)) # rows to take
    id_X = np.sort(random.sample(range(0,M), xMics)) # cols to take

    mask_T = np.zeros([T, 1], dtype=bool)
    mask_X = np.zeros([1, M], dtype=bool)

    mask_T[id_T,0] = True
    mask_X[0,id_X] = True

    return mask_X, mask_T


def unif_downsamp_RIR(shape, ratio_t=1, ratio_x=0.5):
    # choose a ratio of samples in time/space from RIR
    # random choice
    T, M = shape
    tsamples = int(T*ratio_t)
    xMics  = int(M*ratio_x)

    deltaT = T/tsamples
    deltaX = M/xMics

    id_T = np.rint(np.arange(0,T,deltaT)).astype(int) # rows to take
    id_X = np.rint(np.arange(0,M,deltaX)).astype(int) # cols to take

    mask_T = np.zeros([T, 1], dtype=bool)
    mask_X = np.zeros([1, M], dtype=bool)

    mask_T[id_T,0] = True
    mask_X[0,id_X] = True

    return mask_X, mask_T