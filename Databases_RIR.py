# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 17:02:13 2023

@author: Ramón Pozuelo

Here I include the functions and definition
of the classes for the RIR Databases.

Databases:
Zea
MeshRIR
"""

import config
from module_functions_DataBases_RIR import load_DB_ZEA, unif_downsamp_RIR, divide_RIR
import torch.utils.data as data
import torch
import os.path
from typing import Any, Callable, Optional, List #, TypedDict
import numpy as np



class ZeaDataset(data.Dataset):

    base_folder = "DataBase_Zea"

    def __init__(
        self,
        root: str,
        files: List[str] = config.FILES_TRAIN,
        param_downsamp  = config.PARAM_DOWNSAMPLING,
        f_downsamp_RIR: Optional[Callable] = unif_downsamp_RIR, 
        input_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:

        super().__init__()
        self.root = root
        self.Imgs: Any = []
        self.mask_col: Any = []
        self.mask_row: Any = []
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.xImgs: Any = []
        self.tImgs: Any = []


        # Parameters downsampling and division of RIR image
        self.pd = param_downsamp

        # Saveguard that files is a list
        self.files = []
        if isinstance(files, list):
            self.files = files
        else:
            self.files.append(files)

        for file_name in self.files:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            RIR, x, t = load_DB_ZEA(file_path)
            shape = RIR.shape
            maskX, maskT = f_downsamp_RIR(shape, self.pd['ratio_t'], self.pd['ratio_x'])
            Imgs, subMaskX, subMaskT = divide_RIR(RIR, maskX, maskT, self.pd['kernel'][0], self.pd['kernel'][1], self.pd['stride'][0], self.pd['stride'][1])
            _, xImgs, tImgs = divide_RIR(RIR, x, t, self.pd['kernel'][0], self.pd['kernel'][1], self.pd['stride'][0], self.pd['stride'][1])

            self.Imgs.append(Imgs)
            self.mask_col.append(subMaskX)
            self.mask_row.append(subMaskT)
            self.xImgs.append(xImgs)
            self.tImgs.append(tImgs)


        self.Imgs = np.vstack(self.Imgs).reshape(-1, 1, self.pd['kernel'][0], self.pd['kernel'][1])
        self.mask_col = np.vstack(self.mask_col).reshape(-1, 1, 1, self.pd['kernel'][1])
        self.mask_row = np.vstack(self.mask_row).reshape(-1, 1, self.pd['kernel'][0], 1)
        self.xImgs = np.vstack(self.xImgs).reshape(-1, 1, 1, self.pd['kernel'][1])
        self.tImgs = np.vstack(self.tImgs).reshape(-1, 1, self.pd['kernel'][0], 1)


    # Return the number of data in our dataset
    def __len__(self):
        return len(self.Imgs)


    # Return the element idx in the dataset
    def __getitem__(self, idx):
        image = torch.from_numpy(self.Imgs[idx]).float() 
        im_mask_col = torch.from_numpy(self.mask_col[idx]).float()
        im_mask_row = torch.from_numpy(self.mask_row[idx]).float() 
        
        x_image = torch.from_numpy(self.xImgs[idx]).float()
        t_image = torch.from_numpy(self.tImgs[idx]).float()
        
        target = torch.from_numpy(self.Imgs[idx]).float()
        
        if self.target_transform:
            C, H, W = target.size()
            # random Horizontal Flip of mask and image at the same time
            temp  = torch.cat((image, im_mask_col, x_image),dim=-2)
            temp = self.target_transform(temp)
            
            # Recuperate the transformed target and mask
            target, mask_and_xcoord = torch.split(temp, H, dim=-2)
            im_mask_col, x_image = torch.split(mask_and_xcoord, 1, dim=-2)
            
            image = target

        if self.input_transform:
            # The input is the target image but downsampled with the mask
            image = target*im_mask_col
            image = image*im_mask_row

        # Put together the image and x_coordinate and t_coordinate as 3 channels
        

        C2 = x_image.repeat(1,image.shape[-2],1)
        C3 = t_image.repeat(1,1,image.shape[-1])
        
        image_3 = torch.cat((image, C2, C3), dim=0)
            
        return image_3, target, im_mask_col, im_mask_row