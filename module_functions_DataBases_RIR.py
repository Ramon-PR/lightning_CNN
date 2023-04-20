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
    # T, M = RIR.shape
    return RIR

#-----------------------------------------------------------------------
#---------- DIVIDE RIR IMAGE IN SUBSAMPLES -----------------------------
#-----------------------------------------------------------------------
def divide_RIR(RIR, maskX, kt=32, kx=32, strideT=32, strideX=32):
    # Divide RIR in small imagenes 
    # the image is a kernel of size (kt, kx) 
    # the kernel moves in strides strideX and strideT
    # if kt=strideT and kx=strideX then "no-overlap"
    # it also gives the downsampling submask for each subimage.
    
    T, M = RIR.shape
    
    # stride = s
    # kernel = k
    # total: M
    xImags = (M - kx)//strideX + 1
    tImags = (T - kt)//strideT + 1

    nImags = (xImags)*(tImags)

    # Imags without modification
    Y = np.zeros([nImags,kt,kx])
    subMask = np.ones([nImags,1,kx], dtype=bool)

    for imx in range(xImags):
        for imt in range(tImags):
            im = imx*tImags + imt
            # Y[idx] is (nx,nt) if
            Y[im, :, :] = RIR[strideT*imt  : (strideT*imt)+kt, strideX*imx  : (strideX*imx)+kx]
            subMask[im, 0,:] = maskX[0,strideX*imx  : (strideX*imx)+kx]            
    return Y, subMask

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