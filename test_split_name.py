# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 16:01:38 2023

@author: Ramón Pozuelo
"""

import os
import re

def parameters_from_string(name_file = "UNet_L_1_K_3_5_F_10_5_C_1.ckpt"):
    model_name, s_layers, s_kernel, s_nfilters, s_nchan, s_extension = re.split('_L_|_K_|_F_|_C_|\.', name_file)
    # print("file: ", name_file)
    # print("model name: ", model_name)
    # print("number of layers: ", s_layers)
    # print("size of kernels: ", s_kernel)
    # print("number of filters: ", s_nfilters)
    # print("number of channels: ", s_nchan)
    
    # Remove and split using underscores
    # Convert to int and put it in a list
    nchan = list(map(int, s_nchan.strip("_").split("_")))
    nlayers = list(map(int, s_layers.strip("_").split("_")))
    nkernel = list(map(int, s_kernel.strip("_").split("_")))
    nfilters = list(map(int, s_nfilters.strip("_").split("_")))

    # print("file: ", name_file)
    # print("model name: ", model_name)
    # print("number of layers: ", nlayers)
    # print("size of kernels: ", nkernel)
    # print("number of filters: ", nfilters)
    # print("number of channels: ", nchan)

    return model_name, nlayers, nkernel, nfilters, nchan[0]

# %%---------------------------------------------------------------------------------------------

path_ckpt = r"C:\Users\keris\Desktop\lightning_CNN\checkpoints\UNet_L_1_K_3_F_10_C_1.ckpt"
model_ckpt = path_ckpt

ruta, file = os.path.split(model_ckpt)

print(ruta)
print(file)

# parameters_from_string()

model_name, nlayers, nkernel, nfilters, nchan =  parameters_from_string(file)

import lightning.pytorch as pl
trainer = pl.Trainer()
dic_ckpt = trainer.strategy.load_checkpoint(path_ckpt)
print("Number of epochs: ", dic_ckpt['epoch'])


# %%

import torch
from model_system import CNNModule
from models import model_dict as md1
from unet_pytorch import model_dict as md2

# Create a full dictionary with all the model classes
model_dict = dict(md1)
model_dict.update(md2)


# First, instantiate a model
model = CNNModule(
    model_name=model_name,
    model_dict=model_dict,
    model_hparams={
        "n_channels" : nchan,
        "n_filters" : nfilters ,
        "conv_kern" : nkernel ,
        "act_fn_name" : "relu" ,
    },
    optimizer_name="Adam",
    optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
)


# This is needed to load into a CPU a model that was written on a GPU
checkpoint = torch.load(path_ckpt, map_location=lambda storage, loc: storage)

# Only load the state_dict. Otherwise it will complain about the existing models_dict when it was created
# This would load extra info that may not be available
# model = CNNModule.load_from_checkpoint(path_ckpt)
model.load_state_dict(checkpoint["state_dict"])


# Set the model to eval mode
model.eval()


# %% DATA to feed the model (predict).    INPUTS

inp_img={}
inp_img["DATA_DIR"] = r"C:\Users\keris\Desktop\Postdoc"
inp_img["FILE"] = "BalderRIR.mat"

# Image size
idx=0
himg=96*2
wimg=96

# Downsampling
PARAM_DOWNSAMPLING = dict(ratio_t=1, ratio_x=0.5, kernel=[32, 32], stride=[32, 32])
# PARAM_DOWNSAMPLING = dict(ratio_t=1, ratio_x=1, kernel=[32, 32], stride=[32, 32])



from module_functions_DataBases_RIR import load_DB_ZEA, unif_downsamp_RIR, rand_downsamp_RIR, divide_RIR

# f_downsamp_RIR = unif_downsamp_RIR
f_downsamp_RIR = rand_downsamp_RIR



file_path = os.path.join(inp_img["DATA_DIR"], "DataBase_Zea", inp_img["FILE"])
data = load_DB_ZEA(file_path)
RIR = data[0]
x = data[1]
t = data[2]
shape = RIR.shape


maskX, maskT = f_downsamp_RIR(shape, PARAM_DOWNSAMPLING['ratio_t'], PARAM_DOWNSAMPLING['ratio_x'])

Imgs, subMaskX, subMaskT = divide_RIR(RIR, maskX, maskT, 
                                       himg, wimg, 
                                       himg, wimg)

_, xImgs, tImgs = divide_RIR(RIR, x, t, 
                                       himg, wimg, 
                                       himg, wimg)

image = torch.from_numpy(Imgs[idx].reshape(1, himg, wimg)).float() 
x_image = torch.from_numpy(xImgs[idx].reshape(1, 1, wimg)).float()
t_image = torch.from_numpy(tImgs[idx].reshape(1, himg, 1)).float()
mask = torch.from_numpy(subMaskX[idx].reshape(1, -1, wimg)).float()

C2 = x_image.repeat(1,image.shape[-2],1)
C3 = t_image.repeat(1,1,image.shape[-1])


# Downsampling check
# imagep = image * mask
# import matplotlib.pyplot as plt
# imagep = imagep.cpu().detach().numpy().squeeze()
# fig, axs = plt.subplots(1,1, squeeze=False)
# axs[0,0].imshow(imagep)
# plt.show()


# Set up for models with 1 or 3 channels
if nchan==3 :
    image_inp = torch.cat((image*mask, C2, C3), dim=0)
else:
    image_inp = image*mask

# Put an extra dimension (batch dimension)
image_inp = torch.unsqueeze(image_inp, dim=0)

# Predict, pass forward.
image_out = model(image_inp)


import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        
# # Reference image, input
# img = dm.reference_target
# # Reference image output
# img_out = model(torch.unsqueeze(dm.reference_input, dim=0))

image_inp = image_inp[0,0,:,:].reshape(1, himg, wimg)
image_out = image_out.reshape(1, himg, wimg)

grid = torchvision.utils.make_grid([image_inp, image_out], normalize=True)

# plt.imshow(grid)
# plt.show()
show(grid)



# i_inp = image_inp.cpu().detach().numpy().squeeze()
# i_out = image_out.cpu().detach().numpy().squeeze()

# fig, axs = plt.subplots(1,2, squeeze=False)
# axs[0,0].imshow(i_inp)
# axs[0,1].imshow(i_out)
# plt.show()


#%%

# new_img = i_out

# a = i_inp.min()
# b = i_inp.max()

# print(a)
# print(b)


# Amp = b-a
# Max = b

# a = i_out.min()
# b = i_out.max()
# Amp_out = (b-a)

# new_img = i_out / Amp_out * Amp
# new_img = new_img - (new_img.max()-Max)

# print(new_img.min())
# print(new_img.max())


# %%

# fig, axs = plt.subplots(1,2, squeeze=False)
# axs[0,0].imshow(i_inp)
# axs[0,1].imshow(new_img)
# plt.show()





